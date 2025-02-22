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
    --output_dir "outputs/distill_cpu" \
    --dataset_dir "data" \
    --dataset "data/train/train.json" \
    --template "alpaca" \
    --stage "sft" \
    --finetuning_type "lora" \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target "q_proj,k_proj,v_proj,o_proj" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --cutoff_len 512 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --use_cpu \
    --do_train \
    --do_eval \
    --overwrite_cache \
    --report_to "none" \
    --log_level "debug" 