# xEsaModel 设计文档

## 项目目标
对 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 进行知识蒸馏，创建一个轻量级模型，可以在 MacBook M1 Pro 上流畅运行。

## 技术选型

### 基础环境
- Python 3.11 (conda env: dev311)
- LLaMA Factory (主要训练框架)
- PyTorch 2.x (使用 MPS 加速)
- Transformers 4.36.0+
- Git 版本控制

### 模型架构设计
1. 教师模型: DeepSeek-R1-Distill-Qwen-1.5B
2. 学生模型参数:
   - 层数: 12 layers
   - 隐藏维度: 768
   - 注意力头数: 12
   - 词表大小: 32000
   - 最大序列长度: 2048
   - 预计模型大小: ~500MB

### 训练策略 (基于 LLaMA Factory)
1. 知识蒸馏配置:
   - 使用 LLaMA Factory 的 distillation 模式
   - 启用 response_distillation
   - 使用 MPS 后端进行训练
2. 优化设置:
   - 使用 8-bit 优化器
   - 梯度检查点
   - 动态批次大小

### 优化目标
1. 模型大小: < 1GB
2. 推理速度: 在 MacBook M1 Pro 上达到 20+ tokens/s
3. 性能损失: 相比教师模型损失控制在 15% 以内

## 项目结构 (基于 LLaMA Factory)
```
xEsaModel/
├── config/
│   ├── distill_config.yml    # LLaMA Factory 训练配置
│   └── eval_config.yml       # 评估配置
├── data/
│   ├── train/               # 训练数据
│   └── eval/                # 评估数据
├── scripts/
│   ├── prepare_data.py      # 数据预处理脚本
│   ├── train.sh            # 训练启动脚本
│   └── export.sh           # 模型导出脚本
├── outputs/                 # 模型输出目录
└── notebooks/              # 实验分析笔记本
```

## 训练过程总结

### Phase 1: 初始尝试
1. 首次训练配置:
   - 使用完整的知识蒸馏
   - 启用评估功能
   - 结果: 因缺少评估数据集而失败

### Phase 2: 配置优化
1. 移除评估相关配置:
   - 删除 `--do_eval` 参数
   - 删除 `--eval_steps`
   - 结果: 训练成功完成，但模型效果不理想

### Phase 3: LoRA 微调
1. 改进策略:
   - 采用 LoRA 微调替代完整知识蒸馏
   - 配置参数:
     ```yaml
     lora:
       use_lora: true
       r: 32
       alpha: 128
       dropout: 0.05
       target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
     ```
   - 训练参数:
     - 批次大小: 1
     - 梯度累积: 32
     - 学习率: 5e-5
     - 训练轮数: 8
     - 优化器: AdamW
   - 结果: 训练速度提升，内存占用降低

### Phase 4: 数据增强
1. 实现数据增强:
   - 问题变体生成
   - 答案重组
   - 结果: 训练数据量增加，模型表现更稳定

### Phase 5: 最终优化
1. 成功的关键因素:
   - 使用 LoRA 进行轻量级微调
   - 合理的超参数设置
   - 数据增强提升泛化能力
   - CPU 训练确保稳定性

## 最终训练配置
```yaml
# 模型配置
model_name_or_path: "deepseek-ai/deepseek-coder-1.3b-base"
lora_rank: 32
lora_alpha: 128
lora_dropout: 0.05

# 训练参数
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
learning_rate: 5e-5
num_train_epochs: 8
max_steps: -1
warmup_steps: 100

# 优化器设置
optimizer: "adamw_torch"
weight_decay: 0.01
max_grad_norm: 0.5

# 其他设置
logging_steps: 5
save_steps: 50
max_seq_length: 2048
```

## 评估指标
1. 模型大小: LoRA 权重仅 12MB
2. 推理速度: CPU 上约 2-3s/response
3. 性能指标: 回答准确性和流畅度良好
4. 内存占用: 训练期间峰值 8GB
5. 用户体验: 响应速度和质量均可接受

## 后续优化方向
1. GGUF 量化优化
2. 特定场景微调
3. 推理性能优化
4. 模型压缩优化

## 风险管理
1. 性能风险: 模型过小可能导致性能不足
2. 资源风险: M1 Mac 训练资源限制
3. 兼容性风险: 模型格式转换和 LM Studio 适配

## 开发计划
1. Phase 1: 环境配置
   - 配置 conda 环境
   - 安装 LLaMA Factory
   - 准备基础配置文件

2. Phase 2: 数据准备
   - 准备训练数据集
   - 数据预处理和格式化
   - 验证数据集

3. Phase 3: 训练与优化
   - 执行知识蒸馏训练
   - 监控训练过程
   - 参数调优

4. Phase 4: 评估与部署
   - 模型评估
   - 导出 GGUF 格式
   - LM Studio 测试

## 评估指标
1. 模型大小
2. 推理速度
3. 性能指标 (困惑度、准确率等)
4. 内存占用
5. 用户体验评分

## 风险管理
1. 性能风险: 模型过小可能导致性能不足
2. 资源风险: M1 Mac 训练资源限制
3. 兼容性风险: 模型格式转换和 LM Studio 适配

## 后续优化方向
1. GGUF 量化优化
2. 特定场景微调
3. 推理性能优化
4. 模型压缩优化 