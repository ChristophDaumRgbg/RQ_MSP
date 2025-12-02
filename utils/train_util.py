import math
import random
from functools import partial
from pathlib import Path
import evaluate
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, EarlyStoppingCallback

from utils.data_util import get_dataset

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Set seeds for reproducibility
def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

class EmptyCacheCallback(TrainerCallback):
    """Clears GPU cache every 10 evals to balance stability and speed."""
    def __init__(self, every_n_evals: int = 10):
        self.every_n_evals = every_n_evals
        self._eval_counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        self._eval_counter += 1
        if self._eval_counter % self.every_n_evals == 0:
            print(f"[cache] Clearing CUDA cache after {self._eval_counter} evals...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return control

class EnableCacheForEvalCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer.model.config.use_cache = False
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer.model.config.use_cache = True
        return control

    def on_evaluate_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer.model.config.use_cache = False
        return control

class SaveLogsToCSVCallback(TrainerCallback):
    def __init__(self):
        self._saved_rows = 0  # track how many log rows we've already saved

    def on_log(self, args, state, control, **kwargs):
        log_path = Path(args.output_dir) / "train_log_progress.csv"
        new_logs = state.log_history[self._saved_rows:]

        if new_logs:
            df = pd.DataFrame(new_logs)
            df.to_csv(
                log_path,
                mode="a",
                header=(self._saved_rows == 0),  # write header only on first write
                index=False
            )
            self._saved_rows = len(state.log_history)
        return control


def auto_schedule(total_examples: int, global_batch_size: int, aug_factor: int = 7) -> dict:
    unique_examples = total_examples // aug_factor
    steps_per_epoch_unique = math.ceil(unique_examples / global_batch_size)
    eval_steps = 250
    # Buckets by *unique* size
    # Small  : <10k unique
    # Medium : 10k–100k unique
    # Large  : >=100k unique
    if unique_examples < 10_000:
        target_epochs = 10
        max_cap       = 5_000
        min_floor     = 2000
    elif unique_examples < 100_000:
        target_epochs = 8
        max_cap       = 20_000
        min_floor     = 5_000
        eval_steps = 500
    else:
        target_epochs = 6
        max_cap       = 50_000 # max value to safely stay in 24h window
        min_floor     = 20_000
        eval_steps = 1000

    steps_target_epochs = steps_per_epoch_unique * target_epochs
    max_steps = min(max_cap, max(min_floor, steps_target_epochs))

    return {
        "unique_examples": unique_examples,
        "steps_per_epoch_unique": steps_per_epoch_unique,
        "target_epochs": target_epochs,
        "max_steps": max_steps,
        "eval_steps": eval_steps,
    }


def prepare_dataset(batch, feature_extractor, tokenizer):
    audio_arrays = [a["array"] for a in batch["audio_dict"]]
    sampling_rate = batch["audio_dict"][0]["sampling_rate"]

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=sampling_rate,
        return_attention_mask=False,
    )

    labels = tokenizer(
        batch["sentence"],
        padding="longest",
        truncation=True,
        max_length=448,
        return_tensors="np",
    ).input_ids

    return {
        "input_features": inputs["input_features"],
        "labels": labels,
    }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Sort by audio length to minimize padding per batch
        features = sorted(features, key=lambda f: len(f["input_features"][0]), reverse=True)

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def build_compute_metrics_fn(tokenizer):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}
    return compute_metrics

def train_model(args):
    print("Training model with args:", args)
    set_seed(args.seed)

    model = args.model
    language_abbr = args.train_langs[0]
    task = "transcribe"

    dataset = get_dataset(args)
    eval_dataset = load_from_disk(f"/p/scratch/westai0064/daum1/commonvoice_io/noisy_datasets/{language_abbr}/dev_with_all_noise_levels")

    # should be noisy already
    print("Loaded dataset:", dataset)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
    tokenizer = WhisperTokenizer.from_pretrained(model, language=language_abbr, task=task)
    processor = WhisperProcessor.from_pretrained(model, language=language_abbr, task=task)

    compute_metrics = build_compute_metrics_fn(tokenizer)
    # Compute base values
    world_size = torch.cuda.device_count()
    global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    sched = auto_schedule(total_examples=len(dataset), global_batch_size=global_batch, aug_factor=7)
    max_steps = sched["max_steps"]

    # Dont preprocess everything
    used_samples = int(min(len(dataset), max_steps * global_batch * 1.2))  # 20% extra

    print(f"\nDataset stats before mapping:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Using samples: {used_samples} ({used_samples / len(dataset):.1%} of total)")
    print(f"  Global batch size: {global_batch}")
    print(f"  Estimated epochs: {max_steps * global_batch / len(dataset):.2f}")

    dataset = dataset.shuffle(seed=args.seed).select(range(used_samples))
    preprocess_fn = partial(prepare_dataset, feature_extractor=feature_extractor, tokenizer=tokenizer)

    dataset = dataset.map(
        preprocess_fn,
        remove_columns=[col for col in dataset.column_names if col not in ["input_features", "labels"]],
        batched=True,
        batch_size=256,
        num_proc=12, # highest config that works with en
        keep_in_memory=False,
        writer_batch_size=2048,
        load_from_cache_file=True,
    )

    num_eval_samples = min(5000, len(eval_dataset))
    eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(num_eval_samples))

    eval_dataset = eval_dataset.map(
        preprocess_fn,
        remove_columns=[col for col in eval_dataset.column_names if col not in ["input_features", "labels"]],
        batched=True,
        batch_size=256,
        num_proc=12,
        keep_in_memory=False,
        writer_batch_size=2048,
        load_from_cache_file=True,
    )

    print("Prepared dataset columns:", dataset.column_names)
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    model = WhisperForConditionalGeneration.from_pretrained(model)
    model.config.apply_spec_augment = True
    model.generation_config.language = language_abbr
    model.generation_config.task = "transcribe"
    model.config.use_cache = False
    forced_ids = processor.get_decoder_prompt_ids(
        language=language_abbr,
        task="transcribe"
    )
    model.config.forced_decoder_ids = forced_ids
    model.generation_config.forced_decoder_ids = forced_ids

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False


    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        #exclude_modules=".*decoder.*",
        lora_dropout=args.lora_dropout,
        bias="none"
    )

    model = get_peft_model(model, config)

    model.config.use_cache = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,

        bf16=True,
        bf16_full_eval=False,
        fp16_full_eval=False,

        logging_steps=100,

        predict_with_generate=False,
        prediction_loss_only = True,
        eval_strategy="steps",
        eval_steps = sched["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_eval_batch_size = 16,
        eval_accumulation_steps=1,

        save_strategy="steps",
        save_steps=sched["eval_steps"],
        save_total_limit=10,

        remove_unused_columns=False, # required for PEFT
        label_names=["labels"],  # same as above
        ddp_find_unused_parameters=False,
        dataloader_num_workers=16,

    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=None,
        tokenizer=processor,
        callbacks=[
            EmptyCacheCallback(every_n_evals=10),
            EnableCacheForEvalCallback(),
            SaveLogsToCSVCallback(),
            EarlyStoppingCallback(early_stopping_patience=5)
        ],
    )
    
    trainer.train()
    save_path = f"/p/scratch/westai0064/daum1/thesis/models/lora/full_runs/{language_abbr}_full_run/end_result"
    trainer.model.save_pretrained(save_path) # only LoRA Params
    print(f"LoRA parameters saved to {save_path}")

    logs = trainer.state.log_history
    df = pd.DataFrame(logs)

    # Keep useful columns only
    keep_cols = ["step", "loss", "eval_loss", "eval_wer", "eval_cer", "learning_rate"]
    df = df[[c for c in keep_cols if c in df.columns]]

    log_csv_path = f"/p/scratch/westai0064/daum1/thesis/train_logs/training_log_{language_abbr}_full_run.csv"
    df.to_csv(log_csv_path, index=False)
    print(f"Training logs saved to {log_csv_path}")
