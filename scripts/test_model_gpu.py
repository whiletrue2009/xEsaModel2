import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

def load_model():
    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许使用 TF32
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        torch_dtype=torch.float16,  # 使用 FP16
        device_map="auto",  # 自动选择 GPU
        load_in_8bit=True,  # 使用 8-bit 量化
        max_memory={"0": "14GB"}  # 限制显存使用
    )
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        trust_remote_code=True
    )
    
    # 加载 LoRA 配置
    peft_config = PeftConfig.from_pretrained("outputs/distill_gpu")
    
    # 加载 LoRA 模型
    model = PeftModel.from_pretrained(base_model, "outputs/distill_gpu")
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    # 构建 Alpaca 格式的输入
    prompt_template = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    
    # 编码输入
    inputs = tokenizer(prompt_template, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成回答
    with torch.no_grad(), torch.cuda.amp.autocast():  # 使用自动混合精度
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True  # 启用 KV 缓存
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

def main():
    print("正在加载模型...")
    model, tokenizer = load_model()
    
    # 测试问题列表
    test_questions = [
        "什么是服务区资金归集管理办法？",
        "资金归集的定义是什么？",
        "服务区资金归集管理办法适用于哪些单位？",
        "江苏交控营运事业部的主要职责是什么？",
        "资金归集流程中，签订聚合支付协议涉及哪些内容？"
    ]
    
    print("\n开始测试...\n")
    print(f"使用设备: {model.device}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"当前显存使用: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
        print(f"显存峰值使用: {torch.cuda.max_memory_allocated(0)/1024**2:.2f}MB")
    
    for question in test_questions:
        print(f"\n问题: {question}")
        response = generate_response(model, tokenizer, question)
        print(f"回答: {response}")
        if torch.cuda.is_available():
            print(f"当前显存使用: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")

if __name__ == "__main__":
    main() 