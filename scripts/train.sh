#!/bin/bash

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate xesa311

# 设置 CUDA 设备（对于 M1 Mac，我们使用 CPU）
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=.

# 运行训练
python -c "from llmtuner.train import run_exp; run_exp()" \
    --model_name_or_path "deepseek-ai/deepseek-coder-1.3b-base" \
    --output_dir "outputs/distill_cpu_v3" \
    --dataset_dir "data" \
    --dataset "data/train/train.json" \
    --template "alpaca" \
    --stage "sft" \
    --finetuning_type "lora" \
    --lora_rank 32 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --cutoff_len 2048 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --num_train_epochs 15 \
    --logging_steps 5 \
    --save_steps 50 \
    --do_train \
    --overwrite_cache \
    --report_to "none" \
    --log_level "debug" \
    --no_cuda 