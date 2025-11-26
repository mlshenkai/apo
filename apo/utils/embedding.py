# apo/utils/embedding.py
from __future__ import annotations
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np
import os

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# 尝试导入 OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class EmbeddingProvider(ABC):
    """Embedding provider 抽象基类"""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本列表编码为向量"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回embedding维度"""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """使用 sentence-transformers 的 Embedding Provider"""

    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers 未安装。"
                "请使用以下命令安装: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype("float32")

    @property
    def dimension(self) -> int:
        return self._dim


class OpenAIProvider(EmbeddingProvider):
    """使用 OpenAI text-embedding 模型的 Embedding Provider"""

    # OpenAI 各模型的向量维度
    DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if OpenAI is None:
            raise ImportError(
                "openai 库未安装。"
                "请使用以下命令安装: pip install openai"
            )

        self.model_name = model_name
        if model_name not in self.DIMENSIONS:
            raise ValueError(
                f"未知的 OpenAI embedding 模型: {model_name}。"
                f"支持的模型: {list(self.DIMENSIONS.keys())}"
            )

        # 使用提供的 api_key 或从环境变量读取
        # 使用提供的 base_url 或从环境变量读取
        client_kwargs = {
            "api_key": api_key or os.getenv("OPENAI_API_KEY")
        }
        if base_url or os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(**client_kwargs)
        self._dim = self.DIMENSIONS[model_name]

    def embed(self, texts: List[str]) -> np.ndarray:
        """使用 OpenAI API 生成 embeddings"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype="float32")

    @property
    def dimension(self) -> int:
        return self._dim


class RandomProvider(EmbeddingProvider):
    """随机向量 Provider，用于测试"""

    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.random.randn(len(texts), self._dim).astype("float32")

    @property
    def dimension(self) -> int:
        return self._dim


class PromptEmbedder:
    """
    封装 prompt 向量化逻辑，支持多种 embedding 后端。

    支持的 embedding 提供商:
    - sentence-transformers: 本地模型 (默认: all-MiniLM-L6-v2)
    - OpenAI: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
    - random: 随机向量 fallback (用于测试)

    使用示例:
        # 使用 sentence-transformers (默认)
        embedder = PromptEmbedder()

        # 使用 OpenAI text-embedding-3-large
        embedder = PromptEmbedder(model_name="text-embedding-3-large")

        # 使用 OpenAI 并提供 API key
        embedder = PromptEmbedder(
            model_name="text-embedding-ada-002",
            api_key="sk-..."
        )

        # 显式指定 provider 为 random
        embedder = PromptEmbedder(provider="random", dim=512)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        provider: Optional[str] = None,
        dim: int = 384,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化 PromptEmbedder。

        Args:
            model_name: 模型名称，用于自动检测 provider
            provider: 显式指定 provider ("sentence-transformers", "openai", "random")
            dim: 仅用于 random provider 的向量维度
            api_key: OpenAI API key (可选，默认从环境变量 OPENAI_API_KEY 读取)
            base_url: OpenAI API base URL (可选，默认从环境变量 OPENAI_BASE_URL 读取)
        """
        self.model_name = model_name
        self.provider_name = provider or self._detect_provider(model_name)

        # 初始化对应的 provider
        try:
            if self.provider_name == "openai":
                self._provider = OpenAIProvider(model_name, api_key, base_url)
            elif self.provider_name == "sentence-transformers":
                # 移除 "sentence-transformers/" 前缀（如果存在）
                model_path = model_name.replace("sentence-transformers/", "")
                self._provider = SentenceTransformerProvider(model_path)
            elif self.provider_name == "random":
                self._provider = RandomProvider(dim)
            else:
                raise ValueError(f"未知的 provider: {self.provider_name}")
        except (ImportError, Exception) as e:
            print(f"警告: 无法初始化 {self.provider_name} provider: {e}")
            print("退化为随机向量用于测试。")
            self._provider = RandomProvider(dim)
            self.provider_name = "random"

    @staticmethod
    def _detect_provider(model_name: str) -> str:
        """根据模型名称自动检测 embedding provider"""
        if model_name.startswith("text-embedding-"):
            return "openai"
        elif model_name == "random":
            return "random"
        else:
            # 默认使用 sentence-transformers
            return "sentence-transformers"

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为 embeddings。

        Args:
            texts: 要编码的文本列表

        Returns:
            numpy 数组，shape 为 (len(texts), embedding_dim)
        """
        return self._provider.embed(texts)

    @property
    def dim(self) -> int:
        """返回 embedding 维度"""
        return self._provider.dimension