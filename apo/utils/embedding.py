# apo/utils/embedding.py
from __future__ import annotations
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class PromptEmbedder:
    """
    封装 prompt 向量化逻辑。
    默认使用 sentence-transformers；如果不可用，则退化为随机向量（仅用于结构调试）。
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dim: int = 384):
        self.dim = dim
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name)
            self.use_stub = False
        else:
            self.model = None
            self.use_stub = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.use_stub:
            # 随机向量 fallback，用于无依赖情况下测试流程
            return np.random.randn(len(texts), self.dim).astype("float32")
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype("float32")