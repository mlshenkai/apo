# apo/ensemble/voting.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import numpy as np


@dataclass
class EnsembleMember:
    prompt_text: str
    preds_on_val: List[Any]  # 在验证集上的预测
    score: Optional[float] = None


class EnsembleVoter:
    """
    集成投票器，使用加权投票策略。
    通过启发式搜索优化权重，最大化集成模型在验证集上的表现。
    """

    def __init__(self, metric_fn: Callable[[List[Any], List[Any]], float]):
        self.metric_fn = metric_fn

    def optimize_weights(
        self,
        members: List[EnsembleMember],
        val_labels: List[Any],
        w_min: float = 0.05,
        n_steps: int = 1000,
    ) -> np.ndarray:
        """
        简单随机搜索权重（sum=1, w_j>=w_min），最大化 F1/Acc。
        真实项目中可以换成更精致的优化方法。
        """
        m = len(members)
        best_w = np.ones(m) / m
        best_score = -1.0

        preds_matrix = np.array(
            [[p for p in member.preds_on_val] for member in members]
        )  # (M, N)

        for _ in range(n_steps):
            w = np.random.rand(m)
            w = w / np.sum(w)
            # 强制 w_min
            w = np.maximum(w, w_min)
            w = w / np.sum(w)

            # 聚合预测：投票权重最大标签
            # 这里假设 labels 是离散的可 hash 值
            N = preds_matrix.shape[1]
            final_preds = []
            for i in range(N):
                # 对第 i 个样本统计每个标签的权重和
                label_weights: Dict[Any, float] = {}
                for j in range(m):
                    label = preds_matrix[j, i]
                    label_weights[label] = label_weights.get(label, 0.0) + float(w[j])
                best_label = max(label_weights.items(), key=lambda kv: kv[1])[0]
                final_preds.append(best_label)

            score = self.metric_fn(final_preds, val_labels)
            if score > best_score:
                best_score = score
                best_w = w

        return best_w