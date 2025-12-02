import os
from pathlib import Path
import random
import numpy as np
import torch
from datasets import load_dataset, Dataset as HFDataset, Audio as HFAudio, concatenate_datasets, \
    load_from_disk, Dataset, Audio
from torchaudio.transforms import Resample
import config
from utils.audio_util import get_noise_category, combine_audio
import spacy
from utils.models_util import get_snr, load_brouhaha

tokenizer_yue_hant_hk = spacy.load("zh_core_web_sm", disable=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "ner"]).tokenizer
tokenizer_th_th = spacy.blank("th")

GLOBAL_ESC50_AUDIO = None
GLOBAL_ESC50_META = None


resampler_cache = {}

def get_resampler(orig_sr, target_sr):
    key = (orig_sr, target_sr)
    if key not in resampler_cache:
        resampler_cache[key] = Resample(orig_sr, target_sr)
    return resampler_cache[key]

def set_esc50_globals(esc50_audio, esc50_meta):
    global GLOBAL_ESC50_AUDIO, GLOBAL_ESC50_META
    GLOBAL_ESC50_AUDIO = esc50_audio
    GLOBAL_ESC50_META = esc50_meta
    print("Set global ESC-50 audio and meta")

def get_esc50():
    print("Getting esc50")
    # Load dataset from Hugging Face
    dataset = load_dataset("ashraq/esc50", split='train')
    dataset = dataset.cast_column("audio", HFAudio(sampling_rate=44100))
    return dataset

def preprocess_and_save_esc50_hf(target_sr=16000, save_path="esc50_resampled"):
    esc50 = get_esc50()
    resampled_audio = []
    meta_list = []

    for ex in esc50:
        audio_np = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        if sr != target_sr:
            if sr not in resampler_cache:
                resampler_cache[sr] = get_resampler(sr, target_sr)
            audio_res = resampler_cache[sr](torch.tensor(audio_np).unsqueeze(0)).squeeze(0).numpy()
        else:
            audio_res = audio_np

        resampled_audio.append({"array": audio_res, "sampling_rate": target_sr})
        meta_list.append({
            "filename": ex.get("filename", "unknown"),
            "category": ex.get("category", "unknown"),
            "target": ex.get("target", -1)
        })

    # Combine audio + meta into HF dataset
    hf_data = [{"audio": a, **m} for a, m in zip(resampled_audio, meta_list)]
    esc50_dataset = Dataset.from_list(hf_data)

    esc50_dataset.save_to_disk(save_path)
    print(f"Preprocessed ESC-50 saved as HF dataset at {save_path}")

    return esc50_dataset

def assign_noise_indices(main_dataset, esc50_dataset, seed=42):
    random.seed(seed)
    esc50_size = len(esc50_dataset)
    noise_indices = [random.randint(0, esc50_size - 1) for _ in range(len(main_dataset))]

    return main_dataset.add_column("noise_index", noise_indices)


def inject_noise(example, esc50_audio, esc50_meta, noise_level, target_sr=16000):
    if noise_level == 0:
        arr = np.asarray(example["audio"]["array"], dtype=np.float32)
        # No noise → keep clean audio
        return {
            **example,
            "audio": {"array": arr, "sampling_rate": target_sr},
            "noise_level": 0.0,
            "added_noise_file": None,
            "added_noise_type": None,
            "added_noise_type_id": -1,
        }

    speech_dict = example["audio"]
    speech = np.asarray(speech_dict["array"], dtype=np.float32)

    noise = np.asarray(esc50_audio[example["noise_index"]], dtype=np.float32)
    noise_meta = esc50_meta[example["noise_index"]]
    
    combined = combine_audio(speech_dict={"array": speech, "sampling_rate": target_sr},
                             noise_dict={"array": noise, "sampling_rate": target_sr},
                             noise_gain=noise_level,
                             noise_flow=get_noise_category(noise_meta["category"])[1],
                             target_sr=target_sr
    )

    return {
        **example,
        "audio": {"array": combined.astype(np.float32), "sampling_rate": target_sr},
        "noise_level": noise_level,
        "added_noise_file": noise_meta["filename"],
        "added_noise_type": noise_meta["category"],
        "added_noise_type_id": noise_meta["target"],
    }

def inject_noise_batch(batch, noise_level, target_sr=16000):
    esc50_audio = GLOBAL_ESC50_AUDIO
    esc50_meta = GLOBAL_ESC50_META

    audio_dicts = []
    noise_levels = []
    added_files = []
    added_types = []
    added_type_ids = []

    for i, audio_item in enumerate(batch["audio"]):
        ex = {"audio": audio_item, "noise_index": batch["noise_index"][i]}

        new_ex = inject_noise(ex, esc50_audio, esc50_meta, noise_level, target_sr)

        arr = np.asarray(new_ex["audio"]["array"], dtype=np.float32, order="C")

        audio_dicts.append({
            "array": arr,
            "sampling_rate": target_sr
        })

        noise_levels.append(new_ex["noise_level"])
        added_files.append(new_ex["added_noise_file"])
        added_types.append(new_ex["added_noise_type"])
        added_type_ids.append(new_ex["added_noise_type_id"])
        
    return {
        "audio_dict": audio_dicts,
        "noise_level": noise_levels,
        "added_noise_file": added_files,
        "added_noise_type": added_types,
        "added_noise_type_id": added_type_ids
    }

# Lennart's method for loading commonvoice
def load_commonvoice_split(
    language: str,
    split: str = "train",
    num_proc: int = 1,
    sampling_rate: int = 16_000,
    suffix: str = ".tsv",
    use_scratch: bool = True,
) -> HFDataset:
    cv_location = Path(config.commonvoice_path) # The path in project containing metadata
    if use_scratch:
        cv_location_clips = os.path.join(Path("/p/scratch/westai0064/daum1/commonvoice_io/audios"), language)
    else:
        cv_location_clips = os.path.join(cv_location, language, "clips")

    language_dir = cv_location / language
    if not language_dir.exists():
        raise ValueError(
            f"Language {language} not found in Common Voice dataset dir {cv_location.resolve()}"
        )

    split_table = language_dir / f"{split}{suffix}"
    if not split_table.exists():
        raise ValueError(f"Split {split} not found for language {language}")

    dataset = HFDataset.from_csv(
        str(split_table.resolve()),
        sep="\t",
        num_proc=num_proc,
        keep_default_na=False,
        on_bad_lines="skip",
    )

    dataset = dataset.map(
        lambda x: {
            "path": [
                str((cv_location_clips / Path(p)).absolute()) for p in x["path"]
            ]
        },
        batched=True,
        num_proc=num_proc,
        desc="Mapping paths to real audio files...",
    )
    return dataset

def get_dataset(args):
    if args.dataset == "commonvoice":
        print("Loading Common Voice dataset")
        if args.train_langs is None or len(args.train_langs) == 0:
            raise ValueError("Training languages must be specified.")

        if args.training_mode == "monolingual":
            lang = args.train_langs[0]
            dataset = load_from_disk(f"/p/scratch/westai0064/daum1/commonvoice_io/noisy_datasets/{lang}/train_with_all_noise_levels")

        elif args.training_mode == "few-shot":
            if args.train_langs is None or len(args.train_langs) != 2:
                raise ValueError("Few-shot training requires exactly two training languages.")
            main_lang = args.train_langs[0]
            few_shot_lang = args.train_langs[1]
            dataset_main = load_commonvoice_split(main_lang)
            dataset_few_shot = load_commonvoice_split(few_shot_lang).shuffle(seed=args.seed).select(range(args.shot_count))
            dataset = concatenate_datasets([dataset_main, dataset_few_shot])

        elif args.training_mode == "joint":
            if args.train_langs is None or len(args.train_langs) < 2:
                raise ValueError("Joint training requires multiple training languages.")
            datasets = []
            lengths = []
            for lang in args.train_langs:
                ds = load_commonvoice_split(lang)
                datasets.append(ds)
                lengths.append(len(ds))
            total_len = sum(lengths)
            raw_probs = [l / total_len for l in lengths]
            adjusted_probs = [p ** (1 / args.joint_sampling_temperature) for p in raw_probs]
            norm_factor = sum(adjusted_probs)
            sampling_probs = [p / norm_factor for p in adjusted_probs]
            sampled_datasets = []
            target_size = min(lengths) * len(lengths)
            for i, ds in enumerate(datasets):
                n_samples = int(target_size * sampling_probs[i])
                sampled_ds = ds.shuffle(seed=args.seed).select(range(min(n_samples, len(ds))))
                sampled_datasets.append(sampled_ds)
            dataset = concatenate_datasets(sampled_datasets)
        else:
            raise ValueError(f"Invalid training mode: {args.training_mode}")
        return dataset
    else:
        print("Only commonvoice dataset is supported at the moment")

def prep_dataset(args):
    if args.dataset == "commonvoice":
        #prep_commonvoice_split(args.prep_langs[0])
        if args.do_snr:
            prep_commonvoice_snr(args.prep_langs[0], split=args.split)
        if args.do_noisify:
            prep_commonvoice_noise(args.prep_langs[0], split=args.split)
    elif args.dataset == "fleurs":
        if args.do_snr:
            print("Not implemented. Fleurs is usually clean enough.")
        if args.do_noisify:
            prep_fleurs_noise(args.prep_langs[0])
    else:
        raise ValueError("Only commonvoice dataset is supported at the moment")


def prep_commonvoice_split(
    language: str,
    split: str = "train",
    num_proc: int = 1,
    sampling_rate: int = 16000,
    suffix: str = ".tsv",
    noise_levels=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    cv_dir: str = Path(config.commonvoice_path),
):
    output_dir = os.path.join(cv_dir, language)
    dataset = load_commonvoice_split(
        language=language,
        split=split,
        num_proc=num_proc,
        sampling_rate=sampling_rate,
        suffix=suffix,
    )

    inference = load_brouhaha()

    def add_snr(example):
        snr, _, _ = get_snr(inference, example["path"])
        example["snr"] = snr
        return example

    tsv_path = os.path.join(output_dir, f"{split}_with_snr.tsv")
    if not os.path.exists(tsv_path):
        dataset = dataset.map(add_snr, num_proc=num_proc, desc="calc SNR")
        dataset.to_pandas().to_csv(tsv_path, sep="\t", index=False)
    else:
        print(f"File already exists: {tsv_path}")

    esc50 = get_esc50()
    dataset = assign_noise_indices(dataset, esc50)

    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", HFAudio(sampling_rate=sampling_rate))

    noisy_datasets = []
    for level in noise_levels:
        if level == 0:
            noisy_datasets.append(dataset)
            continue  # Skip noise injection for level 0

        def inject(ex):
            return inject_noise(
                ex, esc50, level, output_dir, language, target_sr=sampling_rate, save_to_disk=True
            )

        noisy = dataset.map(inject,
                            desc=f"Noise Injection: {level} dB")
        noisy.save_to_disk(os.path.join(output_dir, "noisy_datasets", f"{split}_with_{level}_noise"))
        noisy_datasets.append(noisy)

    combined_dataset = concatenate_datasets(noisy_datasets)
    noisy.save_to_disk(os.path.join(output_dir, "noisy_datasets", f"{split}_with_all_noise_levels"))

    return combined_dataset

def prep_commonvoice_snr(
    language: str,
    split: str = "train",
    num_proc: int = 1,
    sampling_rate: int = 16000,
    suffix: str = ".tsv",
    on_scratch: bool = True,
):
    cv_dir: str = Path(config.commonvoice_path)
    output_dir = os.path.join(cv_dir, language)
    dataset = load_commonvoice_split(
        language=language,
        split=split,
        num_proc=num_proc,
        sampling_rate=sampling_rate,
        suffix=suffix,
        use_scratch=on_scratch,
    )

    inference = load_brouhaha()
    
    print(dataset)

    def add_snr(example):
        # print(example)
        snr, _, _ = get_snr(inference, example["path"])
        example["snr"] = snr
        return example


    if split == "train":
        out_path = os.path.join(output_dir, "hf_dataset")
    elif split == "dev":
        out_path = os.path.join(output_dir, "dev_dataset")
    if not os.path.exists(out_path):
        dataset = dataset.map(add_snr, num_proc=num_proc, desc="calc SNR")
        dataset = dataset.rename_column("path", "audio")
        dataset = dataset.cast_column("audio", HFAudio(sampling_rate=sampling_rate))
        dataset.save_to_disk(out_path)
        print(dataset)
    else:
        print(f"File already exists: {out_path}")

def prep_commonvoice_noise(
    language: str,
    split: str = "train",
    num_proc: int = 4,
    sampling_rate: int = 16000,
    noise_levels= [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    batch_size: int = 128,
    cv_dir: str = Path(config.commonvoice_path),
    esc50_dir: str = "/p/project1/westai0064/daum1/thesis/datasets/esc_50/esc_50_resampled"
):
    if split == "train":
        dataset = load_from_disk(os.path.join(cv_dir, language, "hf_dataset"))
    elif split == "dev":
        dataset = load_from_disk(os.path.join(cv_dir, language, "dev_dataset"))

    # Convert to dict format for processing
    dataset = dataset.map(lambda ex: {
        "audio_dict": {
            "array": ex["audio"]["array"],
            "sampling_rate": ex["audio"]["sampling_rate"]
        },
    },desc="converting to dict")

    if os.path.exists(esc50_dir):
        print(f"Loading preprocessed ESC-50 from {esc50_dir}")
        esc50_dataset = load_from_disk(esc50_dir)
    else:
        esc50_dataset = preprocess_and_save_esc50_hf(target_sr=sampling_rate, save_path=esc50_dir)
    print(esc50_dataset)
    esc50_audio = [ex["audio"]["array"] for ex in esc50_dataset]
    esc50_meta = [
        {"filename": ex.get("filename", "unknown"),
         "category": ex.get("category", "unknown"),
         "target": ex.get("target", -1),
         "sampling_rate": ex["audio"]["sampling_rate"]}
        for ex in esc50_dataset
    ]
    set_esc50_globals(esc50_audio, esc50_meta)

    zero_noise_path = os.path.join(cv_dir, language, "noisy_datasets", f"{split}_with_0_noise")
    if os.path.exists(zero_noise_path):
        print(f"Loading existing noise indices from {zero_noise_path}")
        existing_dataset = load_from_disk(zero_noise_path)

        # Reuse noise_index field if present
        if "noise_index" in existing_dataset.column_names:
            dataset = dataset.add_column("noise_index", existing_dataset["noise_index"])
        else:
            print("No noise_index column found in existing dataset, reassigning...")
            dataset = assign_noise_indices(dataset, esc50_dataset)
    else:
        dataset = assign_noise_indices(dataset, esc50_dataset)

    noisy_datasets = []
    for level in noise_levels:
        save_path = os.path.join("/p/scratch/westai0064/daum1/commonvoice_io/noisy_datasets", language, f"{split}_with_{level}_noise")
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}")
            noisy = load_from_disk(save_path)
            print(noisy, len(noisy))
            noisy_datasets.append(noisy)
        else:
            noisy = dataset.map(
                lambda batch: inject_noise_batch(batch, level, sampling_rate),
                desc=f"Noise Injection: {level} dB",
                num_proc=num_proc,
                batched=True,
                batch_size=32,
            )
            print("Got out of map")
            noisy.save_to_disk(save_path)
            noisy_datasets.append(noisy)
    combined_dataset = concatenate_datasets(noisy_datasets)
    combined_dataset.save_to_disk(os.path.join("/p/scratch/westai0064/daum1/commonvoice_io/noisy_datasets", language, f"{split}_with_all_noise_levels"))

def prep_fleurs_noise(
    language: str,
    split: str = "test",
    num_proc: int = 4,
    sampling_rate: int = 16000,
    noise_levels= [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    batch_size: int = 128,
    esc50_dir: str = "/p/project1/westai0064/daum1/thesis/datasets/esc_50/esc_50_resampled",
    fleurs_dir: str = Path("/p/project1/westai0064/daum1/thesis/datasets/fleurs"),
):
    column_names = ["id", "path", "raw_transcription", "transcription", "phonemes", "num_samples", "gender"]
    lang_dir = f"/p/project1/westai0064/daum1/thesis/datasets/fleurs/{language}"
    dataset = load_dataset(
        "csv",
        data_files= os.path.join(lang_dir, "test.tsv"),
        split="train",  # "train" here because there's only one file
        delimiter="\t",
        column_names=column_names,
        header=None
    )

    audio_dir = os.path.join(lang_dir, "test")
    dataset = dataset.map(lambda x: {"audio": os.path.join(audio_dir, x["path"])})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(dataset)

    # Convert to dict format for processing
    dataset = dataset.map(lambda ex: {
        "audio_dict": {
            "array": ex["audio"]["array"],
            "sampling_rate": ex["audio"]["sampling_rate"]
        },
    }, desc="converting to dict")

    if os.path.exists(esc50_dir):
        print(f"Loading preprocessed ESC-50 from {esc50_dir}")
        esc50_dataset = load_from_disk(esc50_dir)
    else:
        esc50_dataset = preprocess_and_save_esc50_hf(target_sr=sampling_rate, save_path=esc50_dir)
    print(esc50_dataset)
    esc50_audio = [ex["audio"]["array"] for ex in esc50_dataset]
    esc50_meta = [
        {"filename": ex.get("filename", "unknown"),
         "category": ex.get("category", "unknown"),
         "target": ex.get("target", -1),
         "sampling_rate": ex["audio"]["sampling_rate"]}
        for ex in esc50_dataset
    ]
    set_esc50_globals(esc50_audio, esc50_meta)

    zero_noise_path = os.path.join(fleurs_dir, language, "noisy_datasets", f"{split}_with_0_noise")
    if os.path.exists(zero_noise_path):
        print(f"Loading existing noise indices from {zero_noise_path}")
        existing_dataset = load_from_disk(zero_noise_path)

        # Reuse noise_index field if present
        if "noise_index" in existing_dataset.column_names:
            dataset = dataset.add_column("noise_index", existing_dataset["noise_index"])
        else:
            print("No noise_index column found in existing dataset, reassigning...")
            dataset = assign_noise_indices(dataset, esc50_dataset)
    else:
        dataset = assign_noise_indices(dataset, esc50_dataset)

    noisy_datasets = []
    for level in noise_levels:
        noisy = dataset.map(
            lambda batch: inject_noise_batch(batch, level, sampling_rate),
            desc=f"Noise Injection: {level} dB",
            num_proc=num_proc,
            batched=True,
            batch_size=32,
        )
        noisy_datasets.append(noisy)
    combined_dataset = concatenate_datasets(noisy_datasets)
    combined_dataset.save_to_disk(os.path.join("/p/scratch/westai0064/daum1/fleurs/noisy_datasets", language, f"{split}_with_all_noise_levels"))