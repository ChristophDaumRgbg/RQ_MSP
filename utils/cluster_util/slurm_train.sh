#!/bin/bash
#SBATCH --job-name=lora-train-full-runs
#SBATCH --account=westai0064
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

LANG=$1
echo "Starting LoRA training for language: $LANG"

# Activate Python venv
source /p/project1/westai0064/daum1/thesis/code/thesis_code/.venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Point Accelerate to custom config
export ACCELERATE_CONFIG_FILE=/p/project1/westai0064/daum1/thesis/code/thesis_code/utils/cluster_util/accelerate_config.yaml

export HF_HOME=/p/project1/westai0064/daum1/thesis/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Run training script using Accelerate
accelerate launch --num_processes 4 --mixed_precision=bf16 main.py \
  --train_langs $LANG \
  --model openai/whisper-large-v3 \
  --do_train \
  --training_mode monolingual \
  --learning_rate 1e-5 \
  --max_steps 50000 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj v_proj \
  --dataset commonvoice \
  --output_dir /p/project1/westai0064/daum1/thesis/code/thesis_code/output_${LANG}_${SLURM_JOB_ID} \
  --seed 123 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 5
