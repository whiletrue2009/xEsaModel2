#!/bin/bash

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate xesa311

# 设置 CUDA 设备和环境变量
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 运行训练
python -c "from llmtuner.train import run_exp; run_exp()" \
    --model_name_or_path "deepseek-ai/deepseek-coder-1.3b-base" \
    --output_dir "outputs/distill_gpu" \
    --dataset_dir "data" \
    --dataset "train/train_augmented.json" \
    --template "alpaca" \
    --stage "sft" \
    --finetuning_type "lora" \
    --lora_rank 32 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --cutoff_len 2048 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine_with_restarts" \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --num_train_epochs 8 \
    --logging_steps 5 \
    --save_steps 50 \
    --max_grad_norm 0.5 \
    --do_train \
    --overwrite_cache \
    --report_to "none" \
    --log_level "debug" \
    --fp16 \
    --gradient_checkpointing \
    --load_in_8bit \
    --torch_compile \
    --optim "adamw_torch" \
    --max_memory "14GB" 