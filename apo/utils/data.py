# apo/utils/data.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json
import os


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def default_dataset_paths(task: str) -> Tuple[str, str]:
    base = os.path.join("datasets", task)
    train_path = os.path.join(base, "train.jsonl")
    test_path = os.path.join(base, "test.jsonl")
    return train_path, test_path