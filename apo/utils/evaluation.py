# apo/utils/evaluation.py
from __future__ import annotations
from typing import List, Any
import numpy as np


def accuracy(preds: List[Any], labels: List[Any]) -> float:
    assert len(preds) == len(labels)
    if not preds:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / len(preds)


def macro_f1(preds: List[Any], labels: List[Any]) -> float:
    """
    简单实现的宏平均 F1。
    """
    assert len(preds) == len(labels)
    if not preds:
        return 0.0

    classes = sorted(set(labels) | set(preds))
    f1s = []

    for c in classes:
        tp = sum(int(p == c and y == c) for p, y in zip(preds, labels))
        fp = sum(int(p == c and y != c) for p, y in zip(preds, labels))
        fn = sum(int(p != c and y == c) for p, y in zip(preds, labels))
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        f1s.append(f1)

    if not f1s:
        return 0.0
    return float(np.mean(f1s))


def task_metric(task: str, preds: List[Any], labels: List[Any]) -> float:
    """
    根据任务类型选择评估指标。
    分类任务（liar, bbh, ethos, arsarcasm）使用 Macro-F1。
    推理任务（wsc, gsm8k）使用准确率（Accuracy）。
    """
    task = task.lower()
    if task in ["liar", "bbh", "ethos", "arsarcasm"]:
        return macro_f1(preds, labels)
    elif task in ["wsc", "gsm8k"]:
        return accuracy(preds, labels)
    else:
        # 默认 F1
        return macro_f1(preds, labels)


