# apo/search/bayesian.py
from __future__ import annotations
from typing import List, Sequence
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def expected_improvement(
    X: np.ndarray,
    model: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    计算每个点的 EI。
    """
    from scipy.stats import norm

    mu, sigma = model.predict(X, return_std=True)
    sigma = sigma.reshape(-1) + 1e-9
    mu = mu.reshape(-1)

    imp = mu - y_best - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei


class BayesianPromptSelector:
    """
    基于贝叶斯优化的提示词选择器。
    使用高斯过程回归和期望改进（Expected Improvement）策略选择候选提示词。
    """

    def __init__(self, xi: float = 0.01, n_select: int = 10):
        self.xi = xi
        self.n_select = n_select

    def select(
        self,
        candidate_embs: np.ndarray,
        evaluated_embs: np.ndarray,
        evaluated_scores: np.ndarray,
    ) -> List[int]:
        """
        输入：
            candidate_embs: 待选 prompts 的 embedding (M, d)
            evaluated_embs: 已评估 prompts 的 embedding (N, d)
            evaluated_scores: 已评估的分数 (N,)
        输出：
            选中的 candidate 下标列表
        """
        if len(evaluated_embs) == 0:
            # 没有历史数据，随机选
            idx = np.random.choice(
                len(candidate_embs),
                size=min(self.n_select, len(candidate_embs)),
                replace=False,
            )
            return list(map(int, idx))

        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
        gpr.fit(evaluated_embs, evaluated_scores)

        y_best = float(np.max(evaluated_scores))
        ei = expected_improvement(candidate_embs, gpr, y_best, xi=self.xi)
        n_select = min(self.n_select, len(candidate_embs))
        selected_idx = np.argsort(ei)[-n_select:][::-1]
        return list(map(int, selected_idx))