import re
import json
import os
from datasets import load_dataset

def extract_number(answer: str) -> str:
    """
    GSM8K 的正确答案在答案字符串末尾，例如:
    "He buys 4 apples per day ... #### 28"
    我们提取 '28'
    """
    m = re.search(r"####\s*(-?\d+)", answer)
    if m:
        return m.group(1)
    raise ValueError(f"无法从答案中提取数字: {answer}")


def convert_split(split, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for item in split:
            question = item["question"].strip()
            label = extract_number(item["answer"])
            record = {"input": question, "label": label}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved: {save_path}")


# === 下载 GSM8K ===
dataset = load_dataset("openai/gsm8k", "main")

train = dataset["train"]
test = dataset["test"]

# === 转换为 ELPO/APO 所需格式 ===
convert_split(train, "local_datasets/gsm8k/train.jsonl")
convert_split(test, "local_datasets/gsm8k/test.jsonl")

print("全部完成！")