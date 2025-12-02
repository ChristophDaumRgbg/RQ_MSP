#!/bin/bash
#SBATCH --job-name=eval-full-run-decoder
#SBATCH --account=westai0064
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # 1 GPU is ideal for evaluation
#SBATCH --cpus-per-task=64          # Enough for fast CPU-side preprocessing
#SBATCH --time=06:00:00            # Should finish under 2h even for large test sets
#SBATCH --partition=dc-hwai
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

LANG=$1
echo "Starting evaluation for language $LANG"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate environment
source /p/project1/westai0064/daum1/thesis/code/thesis_code/.venv/bin/activate

# Set important environment variables
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Actually run evaluation
python3 main.py \
  --do_eval \
  --model openai/whisper-large-v3 \
  --lora_adapter $LANG \
  --eval_langs $LANG \
  --dataset fleurs \
  --output_dir ./eval_outputs/$LANG
