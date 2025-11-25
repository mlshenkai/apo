# apo/search/mab.py
from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans


class MABPromptSelector:
    """
    基于多臂老虎机（Multi-Armed Bandit）的提示词选择器。
    通过聚类和 UCB 策略在高潜力簇中采样候选提示词。
    """

    def __init__(self, n_clusters: int = 8, c: float = 1.0, n_rounds: int = 20,
                 per_round: int = 2):
        self.n_clusters = n_clusters
        self.c = c
        self.n_rounds = n_rounds
        self.per_round = per_round

    def select(self, candidate_embs: np.ndarray,
               candidate_scores: Optional[Dict[int, float]] = None) -> List[int]:
        """
        这里假定 candidate_scores 是某些 prompt 的已观测 reward（可选）。
        简化地：使用已有score估计 cluster reward，没有则初始化为0。
        """
        n = len(candidate_embs)
        if n == 0:
            return []

        # 聚类
        k = min(self.n_clusters, n)
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = kmeans.fit_predict(candidate_embs)

        # 初始化 bandit 状态
        rewards = np.zeros(k, dtype=float)
        pulls = np.zeros(k, dtype=int) + 1  # 防止除0

        if candidate_scores is not None:
            # 将已有score聚合到 cluster 上
            for idx, score in candidate_scores.items():
                cluster_id = labels[idx]
                rewards[cluster_id] += score
                pulls[cluster_id] += 1

        selected_indices: List[int] = []
        total_pulls = int(np.sum(pulls))

        for t in range(self.n_rounds):
            # UCB
            ucb = rewards / pulls + self.c * np.sqrt(
                np.log(total_pulls + 1) / pulls
            )
            top_clusters = np.argsort(ucb)[-self.per_round:]

            for cid in top_clusters:
                # 从该 cluster 中随机选一个样本
                cluster_indices = np.where(labels == cid)[0]
                cand_idx = np.random.choice(cluster_indices)
                selected_indices.append(int(cand_idx))
                # 更新 bandit 状态（这里 reward 暂时无法更新，除非我们外部实际评估后再回写）

        # 去重
        selected_indices = list(dict.fromkeys(selected_indices))
        return selected_indices