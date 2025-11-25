# apo/config.py
"""
配置管理模块，从 .env 文件加载所有配置参数。
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


@dataclass
class OptimizerLLMConfig:
    """优化器 LLM 配置（用于生成和改进提示词）"""
    api_key: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int

    @classmethod
    def from_env(cls) -> OptimizerLLMConfig:
        return cls(
            api_key=os.getenv("OPTIMIZER_LLM_API_KEY", "your_api_key_here"),
            model=os.getenv("OPTIMIZER_LLM_MODEL", "gpt-4o"),
            base_url=os.getenv("OPTIMIZER_LLM_BASE_URL", "https://api.openai.com/v1"),
            temperature=float(os.getenv("OPTIMIZER_LLM_TEMPERATURE", "1.0")),
            max_tokens=int(os.getenv("OPTIMIZER_LLM_MAX_TOKENS", "2048")),
        )


@dataclass
class TaskModelConfig:
    """任务模型配置（用于执行实际任务）"""
    api_key: str
    model_name: str
    base_url: str
    temperature: float
    max_tokens: int

    @classmethod
    def from_env(cls) -> TaskModelConfig:
        return cls(
            api_key=os.getenv("TASK_MODEL_API_KEY", "your_api_key_here"),
            model_name=os.getenv("TASK_MODEL_NAME", "gpt-3.5-turbo"),
            base_url=os.getenv("TASK_MODEL_BASE_URL", "https://api.openai.com/v1"),
            temperature=float(os.getenv("TASK_MODEL_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("TASK_MODEL_MAX_TOKENS", "2048")),
        )


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    model_name: str

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        )


@dataclass
class OptimizationConfig:
    """优化参数配置"""
    default_rounds: int
    default_n_prompts: int
    default_n_iters: int

    @classmethod
    def from_env(cls) -> OptimizationConfig:
        return cls(
            default_rounds=int(os.getenv("DEFAULT_ROUNDS", "5")),
            default_n_prompts=int(os.getenv("DEFAULT_N_PROMPTS", "10")),
            default_n_iters=int(os.getenv("DEFAULT_N_ITERS", "3")),
        )


@dataclass
class SearchConfig:
    """搜索参数配置"""
    bayesian_n_select: int
    bayesian_xi: float
    mab_n_clusters: int
    mab_n_rounds: int

    @classmethod
    def from_env(cls) -> SearchConfig:
        return cls(
            bayesian_n_select=int(os.getenv("BAYESIAN_N_SELECT", "10")),
            bayesian_xi=float(os.getenv("BAYESIAN_XI", "0.01")),
            mab_n_clusters=int(os.getenv("MAB_N_CLUSTERS", "8")),
            mab_n_rounds=int(os.getenv("MAB_N_ROUNDS", "10")),
        )


@dataclass
class EnsembleConfig:
    """集成参数配置"""
    top_k: int
    w_min: float
    n_steps: int

    @classmethod
    def from_env(cls) -> EnsembleConfig:
        return cls(
            top_k=int(os.getenv("ENSEMBLE_TOP_K", "5")),
            w_min=float(os.getenv("ENSEMBLE_W_MIN", "0.05")),
            n_steps=int(os.getenv("ENSEMBLE_N_STEPS", "200")),
        )


@dataclass
class LoggingConfig:
    """日志配置"""
    log_level: str
    log_file: str

    @classmethod
    def from_env(cls) -> LoggingConfig:
        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "apo.log"),
        )


@dataclass
class APOConfig:
    """APO 全局配置"""
    optimizer_llm: OptimizerLLMConfig
    task_model: TaskModelConfig
    embedding: EmbeddingConfig
    optimization: OptimizationConfig
    search: SearchConfig
    ensemble: EnsembleConfig
    logging: LoggingConfig

    @classmethod
    def from_env(cls) -> APOConfig:
        """从环境变量加载所有配置"""
        return cls(
            optimizer_llm=OptimizerLLMConfig.from_env(),
            task_model=TaskModelConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            optimization=OptimizationConfig.from_env(),
            search=SearchConfig.from_env(),
            ensemble=EnsembleConfig.from_env(),
            logging=LoggingConfig.from_env(),
        )


# 全局配置实例（单例模式）
_config: Optional[APOConfig] = None


def get_config() -> APOConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = APOConfig.from_env()
    return _config


def reload_config() -> APOConfig:
    """重新加载配置（用于配置变更后刷新）"""
    global _config
    load_dotenv(override=True)
    _config = APOConfig.from_env()
    return _config
