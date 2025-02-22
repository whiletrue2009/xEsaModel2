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