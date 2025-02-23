# xEsaModel

基于 LLaMA Factory 的模型蒸馏项目，目标是将 DeepSeek-R1-Distill-Qwen-1.5B 蒸馏为更小的模型。

## 环境要求

- Python 3.11
- PyTorch 2.x
- LLaMA Factory
- MacOS (M1 芯片)

## 项目结构

```
xEsaModel/
├── config/               # 配置文件
├── data/                # 数据目录
├── scripts/             # 训练脚本
├── notebooks/           # 实验笔记本
└── outputs/             # 输出目录
```

## 安装

1. 创建并激活 conda 环境：
```bash
conda create -n xesa311 python=3.11
conda activate xesa311
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 训练过程

1. 数据准备：
   - 将训练数据放在 `data/train/` 目录下
   - 运行数据增强脚本：
     ```bash
     python scripts/augment_data.py
     ```

2. 训练模型：
   ```bash
   bash scripts/train.sh
   ```

3. 测试模型：
   ```bash
   python scripts/test_model.py
   ```

4. 导出模型：
   ```bash
   python scripts/merge_lora.py
   ```

## 输出目录说明

训练完成后，在 `outputs/distill_cpu_v4/` 目录下会生成以下文件：

### 核心文件
- `adapter_model.bin`: LoRA 权重文件，包含了微调的参数
- `adapter_config.json`: LoRA 配置文件，定义了模型结构和训练参数
- `special_tokens_map.json`: 特殊词符映射文件
- `tokenizer_config.json`: 分词器配置文件
- `tokenizer.model`: 分词器模型文件

### 过程文件
- `trainer_state.json`: 训练状态记录，包含学习率、损失等信息
- `training_args.bin`: 训练参数存档
- `log/`: 训练日志目录
- `runs/`: TensorBoard 日志目录（如果启用）
- `checkpoint-*/`: 训练过程中的检查点文件（可以删除）

### 合并后的模型
运行 `merge_lora.py` 后，在 `outputs/merged_model/` 目录下会生成：
- `pytorch_model.*.safetensors`: 合并后的模型权重（分片）
- `config.json`: 模型配置文件
- `tokenizer.*`: 分词器相关文件
- `lmstudio_config.json`: LM Studio 配置文件
- `README.md`: 模型使用说明

## 模型使用

### 在 Python 中使用
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型和 LoRA 权重
base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
model = PeftModel.from_pretrained(base_model, "outputs/distill_cpu_v4")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")

# 生成回答
response = model.generate(...)
```

### 在 LM Studio 中使用
1. 运行 `merge_lora.py` 导出完整模型
2. 在 LM Studio 中导入 `outputs/merged_model` 目录
3. 使用 Alpaca 模板进行对话

## 训练效果

1. 模型大小：
   - 基础模型：1.3GB
   - LoRA 权重：12MB
   - 合并后：1.3GB

2. 性能指标：
   - 训练速度：0.59 samples/s
   - 推理速度：2-3s/response
   - 显存占用：<8GB

3. 优化历程：
   - 初始尝试：完整知识蒸馏，因资源限制失败
   - 改进方案：采用 LoRA 微调，降低资源需求
   - 最终方案：CPU 训练 + 数据增强，取得理想效果

## 注意事项

- 训练过程使用 CPU 以确保稳定性
- 建议使用较小的 batch size 以适应内存限制
- 使用梯度检查点来节省内存
- 推理时建议使用 4-bit 或 8-bit 量化 