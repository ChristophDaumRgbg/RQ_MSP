import os

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from datasets import load_from_disk
import evaluate


def evaluate_lora(args):
    model = args.model
    lora_adapter = args.lora_adapter
    eval_lang = args.eval_langs[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fleurs_path = "/p/scratch/westai0064/daum1/fleurs/noisy_datasets/"
    lora_path = f"/p/scratch/westai0064/daum1/thesis/models/lora/full_runs/{lora_adapter}_full_run/end_result"

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model, language=eval_lang, task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(model)
    if args.lora_adapter:  # only load LoRA if provided
        print(f"Loading LoRA adapter: {args.lora_adapter}")
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        print("No LoRA adapter provided — using **base Whisper model only**")
        model = base_model

    forced_ids = processor.get_decoder_prompt_ids(language=eval_lang, task="transcribe")

    model.config.use_cache = False
    model.generation_config.use_cache = False
    model.config.forced_decoder_ids = forced_ids
    model.generation_config.forced_decoder_ids = forced_ids
    model.generation_config.language = eval_lang
    model.generation_config.task = "transcribe"

    model.to(device).eval()

    # Load dataset
    dataset = load_from_disk(os.path.join(fleurs_path, eval_lang, "test_with_all_noise_levels")) # load fleurs from disk

    print(dataset)

    # Preprocess audio
    def prepare_example(batch):
        audios = batch["audio_dict"]

        features = processor(
            [a["array"] for a in audios],
            sampling_rate=audios[0]["sampling_rate"],
            return_tensors="pt"
        ).input_features

        batch["input_features"] = features.tolist()
        batch["reference"] = batch.get("transcription", [""] * len(audios))
        batch["gender"] = batch.get("gender", ["unknown"] * len(audios))
        return batch

    dataset = dataset.map(
        prepare_example,
        batched=True,
        batch_size=64,  # match GPU batch size
        num_proc=1
    )

    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Batch inference
    BATCH_SIZE = 64
    predictions, references, genders = [], [], []

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Running inference"):
        batch = dataset[i:i + BATCH_SIZE]
        input_features = torch.tensor(batch["input_features"], dtype=torch.float32).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred_ids = model.generate(input_features, max_new_tokens=444)

        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        predictions.extend(preds)
        references.extend(batch["reference"])
        genders.extend(batch["gender"])

    # Compute metrics
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"WER: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")
    print("========================================\n")

    results_folder = "/p/scratch/westai0064/daum1/thesis/models/lora/results"
    if args.lora_adapter is None:
        output_csv = os.path.join(results_folder, eval_lang, f"base_whisper_results.csv")
    else:
        output_csv = os.path.join(results_folder, eval_lang, f"lora_{lora_adapter}_whisper_results.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    selected_cols = [
        "id",
        "gender",
        "noise_index",
        "noise_level",
        "added_noise_file",
        "added_noise_type",
        "added_noise_type_id",
    ]
    # Extract those from the dataset (ignore missing columns gracefully)
    available_cols = [c for c in selected_cols if c in dataset.column_names]
    meta_df = pd.DataFrame(dataset)[available_cols]

    # Combine with predictions and references
    results_df = pd.DataFrame({
        "prediction": predictions,
        "reference": references,
    })

    # Align and merge
    if len(meta_df) != len(results_df):
        raise ValueError(f"Length mismatch: metadata={len(meta_df)}, preds={len(results_df)}")

    final_df = pd.concat([meta_df, results_df], axis=1)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")

    return {"WER": wer, "CER": cer}