import json
import random
from typing import List, Dict

def load_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data: List[Dict], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def augment_question(question: str) -> List[str]:
    """生成问题的变体"""
    templates = [
        "请问{q}",
        "能否介绍一下{q}",
        "请详细说明{q}",
        "关于{q}，能否详细解释？",
        "{q}具体是什么？"
    ]
    return [template.format(q=question.rstrip("？").rstrip("?")) + "？" 
            for template in templates]

def augment_answer(answer: str) -> List[str]:
    """生成答案的变体"""
    # 分句
    sentences = answer.split("，")
    if len(sentences) > 1:
        # 调整句子顺序
        variants = []
        for i in range(min(3, len(sentences))):
            random.shuffle(sentences)
            variants.append("，".join(sentences))
        return variants
    return [answer]

def create_augmented_data(data: List[Dict]) -> List[Dict]:
    augmented_data = []
    
    for item in data:
        # 原始数据保留
        augmented_data.append(item)
        
        # 问题增强
        aug_questions = augment_question(item["instruction"])
        # 答案增强
        aug_answers = augment_answer(item["output"])
        
        # 组合增强后的问答对
        for q in aug_questions:
            for a in aug_answers:
                if q != item["instruction"] or a != item["output"]:
                    augmented_data.append({
                        "instruction": q,
                        "input": "",
                        "output": a,
                        "history": []
                    })
    
    return augmented_data

def main():
    # 加载原始数据
    original_data = load_data("data/train/train.json")
    print(f"原始数据集大小: {len(original_data)}")
    
    # 数据增强
    augmented_data = create_augmented_data(original_data)
    print(f"增强后数据集大小: {len(augmented_data)}")
    
    # 保存增强后的数据
    save_data(augmented_data, "data/train/train_augmented.json")
    print("数据增强完成，已保存到 data/train/train_augmented.json")

if __name__ == "__main__":
    main() 