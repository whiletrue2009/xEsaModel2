import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model():
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
    
    # 加载 LoRA 配置
    peft_config = PeftConfig.from_pretrained("outputs/distill_cpu")
    
    # 加载 LoRA 模型
    model = PeftModel.from_pretrained(base_model, "outputs/distill_cpu")
    model.eval()
    
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
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

def main():
    print("正在加载模型...")
    model, tokenizer = load_model()
    
    # 测试问题列表
    test_questions = [
        "资金归集的定义是什么？",
        "服务区资金归集管理办法适用于哪些单位？",
        "江苏交控营运事业部的主要职责是什么？"
    ]
    
    print("\n开始测试...\n")
    for question in test_questions:
        print(f"问题: {question}")
        response = generate_response(model, tokenizer, question)
        print(f"回答: {response}\n")

if __name__ == "__main__":
    main() 