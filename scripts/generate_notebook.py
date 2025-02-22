import nbformat as nbf
import json

# 创建一个新的笔记本
nb = nbf.v4.new_notebook()

# 添加 Markdown 单元格 - 简介
nb.cells.append(nbf.v4.new_markdown_cell("""# xEsaModel GPU 训练 (Google Colab)

## 环境准备

1. 检查 GPU 类型
2. 安装依赖
3. 克隆代码仓库
4. 准备数据集"""))

# 添加代码单元格 - 检查 GPU
nb.cells.append(nbf.v4.new_code_cell("""# 检查 GPU
!nvidia-smi"""))

# 添加代码单元格 - 安装依赖
nb.cells.append(nbf.v4.new_code_cell("""# 安装依赖
!pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
!pip install transformers==4.31.0 tokenizers==0.13.3 accelerate==0.21.0
!pip install peft==0.6.0 bitsandbytes==0.41.1 flash-attn==2.3.3
!pip install datasets==3.3.2 llmtuner==0.3.0"""))

# 添加代码单元格 - 克隆代码仓库
nb.cells.append(nbf.v4.new_code_cell("""# 克隆代码仓库
!git clone https://github.com/whiletrue2009/xEsaModel2.git
%cd xEsaModel2"""))

# 添加 Markdown 单元格 - 数据准备说明
nb.cells.append(nbf.v4.new_markdown_cell("""## 数据准备

有两种方式准备数据：
1. 从 Google Drive 挂载
2. 直接上传到 Colab"""))

# 添加代码单元格 - 方式1：Google Drive
nb.cells.append(nbf.v4.new_code_cell("""# 方式1：挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 复制数据集
!cp -r /content/drive/MyDrive/xEsaModel2/data ."""))

# 添加代码单元格 - 方式2：直接上传
nb.cells.append(nbf.v4.new_code_cell("""# 方式2：直接上传
from google.colab import files
uploaded = files.upload()

# 解压数据集
!mkdir -p data/train
!mv train_augmented.json data/train/"""))

# 添加 Markdown 单元格 - 开始训练
nb.cells.append(nbf.v4.new_markdown_cell("## 开始训练"))

# 添加代码单元格 - 训练代码
training_code = """# 设置环境变量
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# 运行训练
!python -c "from llmtuner.train import run_exp; run_exp()" \\
    --model_name_or_path "deepseek-ai/deepseek-coder-1.3b-base" \\
    --output_dir "outputs/distill_gpu" \\
    --dataset_dir "data" \\
    --dataset "train/train_augmented.json" \\
    --template "alpaca" \\
    --stage "sft" \\
    --finetuning_type "lora" \\
    --lora_rank 32 \\
    --lora_alpha 128 \\
    --lora_dropout 0.05 \\
    --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \\
    --per_device_train_batch_size 16 \\
    --gradient_accumulation_steps 2 \\
    --cutoff_len 2048 \\
    --learning_rate 5e-5 \\
    --lr_scheduler_type "cosine_with_restarts" \\
    --warmup_steps 100 \\
    --weight_decay 0.01 \\
    --num_train_epochs 8 \\
    --logging_steps 5 \\
    --save_steps 50 \\
    --max_grad_norm 0.5 \\
    --do_train \\
    --overwrite_cache \\
    --report_to "none" \\
    --log_level "debug" \\
    --fp16 \\
    --gradient_checkpointing \\
    --load_in_8bit \\
    --torch_compile \\
    --optim "adamw_torch" \\
    --max_memory "14GB\""""

nb.cells.append(nbf.v4.new_code_cell(training_code))

# 添加 Markdown 单元格 - 保存模型说明
nb.cells.append(nbf.v4.new_markdown_cell("""## 保存模型

训练完成后，我们有两种方式保存模型：
1. 保存到 Google Drive
2. 直接下载到本地"""))

# 添加代码单元格 - 保存到 Drive
nb.cells.append(nbf.v4.new_code_cell("""# 方式1：保存到 Google Drive
!cp -r outputs/distill_gpu /content/drive/MyDrive/xEsaModel2/outputs/"""))

# 添加代码单元格 - 下载到本地
nb.cells.append(nbf.v4.new_code_cell("""# 方式2：打包下载
!tar -czf model.tar.gz outputs/distill_gpu/
from google.colab import files
files.download('model.tar.gz')"""))

# 添加 Markdown 单元格 - 测试模型
nb.cells.append(nbf.v4.new_markdown_cell("## 测试模型"))

# 添加代码单元格 - 测试代码
test_code = """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model():
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,
        max_memory={"0": "14GB"}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, "outputs/distill_gpu")
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    prompt_template = f\"\"\"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
\"\"\"
    
    inputs = tokenizer(prompt_template, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# 加载模型并测试
model, tokenizer = load_model()

test_questions = [
    "什么是服务区资金归集管理办法？",
    "资金归集的定义是什么？",
    "服务区资金归集管理办法适用于哪些单位？",
    "江苏交控营运事业部的主要职责是什么？",
    "资金归集流程中，签订聚合支付协议涉及哪些内容？"
]

for question in test_questions:
    print(f"\\n问题: {question}")
    response = generate_response(model, tokenizer, question)
    print(f"回答: {response}")"""

nb.cells.append(nbf.v4.new_code_cell(test_code))

# 设置笔记本元数据
nb.metadata = {
    "accelerator": "GPU",
    "colab": {
        "gpuType": "T4",
        "provenance": []
    },
    "kernelspec": {
        "display_name": "Python 3",
        "name": "python3"
    },
    "language_info": {
        "name": "python"
    }
}

# 保存笔记本
with open('notebooks/train_on_colab.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 