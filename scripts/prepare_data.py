import json
import os

def convert_to_alpaca(input_file, output_file):
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    # 转换为 Alpaca 格式
    alpaca_data = []
    for qa in qa_pairs:
        alpaca_item = {
            "instruction": qa["question"],
            "input": "",  # 对于简单问答，input 可以为空
            "output": qa["answer"],
            "history": []  # 不需要历史对话
        }
        alpaca_data.append(alpaca_item)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "data/question_answer_pairs.json"
    output_file = "data/train/train.json"
    convert_to_alpaca(input_file, output_file)
    print(f"数据已转换并保存到 {output_file}") 