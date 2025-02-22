#!/bin/bash

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dev311

# 设置 CUDA 设备（对于 M1 Mac，我们使用 MPS）
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 运行训练
python -m llmtuner.cli.run_distill \
    --config_file config/distill_config.yml \
    --do_train \
    --do_eval \
    --overwrite_cache \
    --report_to "none" 