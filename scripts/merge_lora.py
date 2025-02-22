import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def merge_lora_to_base_model():
    print("正在加载基础模型...")
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        trust_remote_code=True
    )
    
    print("正在加载 LoRA 模型...")
    # 加载 LoRA 配置
    model = PeftModel.from_pretrained(base_model, "outputs/distill_cpu")
    
    print("正在合并 LoRA 权重...")
    # 合并 LoRA 权重到基础模型
    merged_model = model.merge_and_unload()
    
    print("正在保存合并后的模型...")
    # 保存合并后的模型
    save_path = "outputs/merged_model"
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"模型已保存到 {save_path}")
    return save_path

if __name__ == "__main__":
    merge_lora_to_base_model() 