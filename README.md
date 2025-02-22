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
conda activate dev311
```

2. 安装依赖：
```bash
pip install llmtuner
```

## 使用方法

1. 准备训练数据：
   - 将训练数据放在 `data/train/` 目录下
   - 将评估数据放在 `data/eval/` 目录下

2. 修改配置：
   - 编辑 `config/distill_config.yml` 文件
   - 根据需要调整模型参数和训练参数

3. 开始训练：
```bash
bash scripts/train.sh
```

## 注意事项

- 训练过程使用 MPS 后端（Apple Silicon）
- 建议使用较小的 batch size 以适应内存限制
- 使用梯度检查点来节省内存

## 模型导出

训练完成后，模型将保存在 `outputs/distill` 目录下。可以使用 LM Studio 加载并测试模型。 