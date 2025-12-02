import argparse

def check_input_valid(args):
    if args.do_train:
        print(f'length of train_langs {len(args.train_langs)}')
        print(args.train_langs)
        if args.training_mode == "monolingual":
            if len(args.train_langs) != 1:
                raise ValueError("Monolingual training requires exactly one training language.")
        elif args.training_mode == "few-shot":
            if len(args.train_langs) != 2:
                raise ValueError("zero-shot training requires exactly two training languages.")
            if args.few_shot_sampling_strategy is None:
                raise ValueError("Few-shot training requires a sampling strategy.")
            if args.shot_count <= 0:
                raise ValueError("Few-shot training requires a positive shot count.")
        elif args.training_mode == "joint":
            if not len(args.train_langs) > 1:
                raise ValueError("Joint training requires multiple training language.")
        else:
            raise ValueError(f"Invalid training mode: {args.training_mode}")
    return True

def get_args():
    parser = argparse.ArgumentParser(description="Training and evaluation script")

    # Config file (optional alternative to CLI args)
    parser.add_argument("--config_file", type=str, default=None, help="Path to YAML/JSON config file")

    # Model & Tokenizer
    parser.add_argument("--model", type=str, help="Path to pretrained model or model identifier from huggingface.co/models")

    # Training Modes
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--training_mode", type=str, choices=["monolingual", "few-shot", "joint"], help="Choose training strategy")

    # Dataset preparation and noisification
    parser.add_argument("--do_snr", action="store_true", help="Whether to run dataset preparation")
    parser.add_argument("--do_noisify", action="store_true", help="Whether to run dataset noisification")

    # Evaluation
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    parser.add_argument("--eval_mode", type=str, choices=["monolingual", "zero-shot", "few-shot", "cross-lingual"],
                        help="Evaluation strategy")
    parser.add_argument("--lora_adapter", type=str, help="the lora adapter to use for evaluation", default=None)

    # Dataset
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=str, help="Dataset name", default="train")
    parser.add_argument("--train_langs", type=str, nargs='+', default=None,
                        help="list of training languages (e.g., [\"es\",\"pt\"])")
    parser.add_argument("--eval_langs", type=str, nargs='+', default=None,
                        help="list of evaluation languages (e.g., [\"es\",\"pt\"])")
    parser.add_argument("--min_snr", type=float, default=20)
    parser.add_argument("--prep_langs", type=str, nargs='+', default=None,
                        help="list of prep languages (e.g., [\"es\",\"pt\"])")

    # Few-shot specific
    parser.add_argument("--few_shot_sampling_strategy", type=str, default=None, choices=["random", "stratified"], help="Sampling strategy for few-shot sampling")
    parser.add_argument("--shot_count", type=int, default=0,
                        help="Number of training examples per language (for few-shot)")

    parser.add_argument("--joint_sampling_temperature", type=float, default=1, help="Temperature for joint sampling (1 for no temperature scaling, >1 flattens distribution)")

    # Training hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating model weights. ")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=100000, help="Number of total training steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["linear", "cosine", "polynomial", "cosine_with_restarts"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=512)

    # LoRA specific parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=["q_proj", "v_proj"],
                            help="Target modules for LoRA adaptation")

    # Output and logging
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"], help="When to save checkpoints.")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # Resume and hub
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)


    args = parser.parse_args()
    if not check_input_valid(args):
        raise ValueError("Invalid input arguments")
    return args
