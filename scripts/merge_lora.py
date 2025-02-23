import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
import shutil

def merge_lora_to_base_model():
    print("正在加载基础模型...")
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        trust_remote_code=True
    )
    
    print("正在加载 LoRA 模型...")
    # 加载 LoRA 模型
    model = PeftModel.from_pretrained(base_model, "outputs/distill_cpu_v4")
    
    print("正在合并 LoRA 权重...")
    # 合并 LoRA 权重到基础模型
    merged_model = model.merge_and_unload()
    
    print("正在保存合并后的模型...")
    # 保存合并后的模型
    save_path = "outputs/merged_model"
    os.makedirs(save_path, exist_ok=True)
    
    # 保存模型和分词器
    merged_model.save_pretrained(
        save_path,
        safe_serialization=True,  # 使用 safetensors 格式
        max_shard_size="2GB"  # 分片大小
    )
    tokenizer.save_pretrained(save_path)
    
    # 创建 LM Studio 所需的配置文件
    config = {
        "name": "xEsaModel-v1",
        "language": ["zh"],
        "model_type": "llama",
        "context_length": 2048,
        "model_format": "pytorch",
        "quantization": "none",
        "repository": "local",
        "architecture": "base_model",
        "licence": "private",
        "creator": "xEsaModel Team"
    }
    
    # 保存配置文件
    import json
    with open(os.path.join(save_path, "lmstudio_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 创建模型说明文件
    model_card = """# xEsaModel v1

## 模型说明
这是一个基于 DeepSeek Coder 1.3B 模型，使用 LoRA 技术微调的中文问答模型。主要针对特定领域的问答任务进行了优化。

## 使用方法
1. 在 LM Studio 中导入模型
2. 使用 Alpaca 格式进行对话：
   ```
   Below is an instruction that describes a task. Write a response that appropriately completes the request.

   ### Instruction:
   你的问题

   ### Response:
   ```

## 模型参数
- 基础模型：DeepSeek Coder 1.3B
- 上下文长度：2048
- 词表大小：32000
- 量化方式：无

## 注意事项
- 建议使用 4-bit 或 8-bit 量化以减少内存占用
- 推理时建议使用 temperature=0.7
- 建议使用 repetition_penalty=1.1
"""
    
    with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)
    
    print(f"模型已保存到 {save_path}")
    print("模型格式: PyTorch (safetensors)")
    print("可以直接在 LM Studio 中导入使用")
    return save_path

if __name__ == "__main__":
    merge_lora_to_base_model() 