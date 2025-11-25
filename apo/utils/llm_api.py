# apo/utils/llm_api.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class LLMConfig:
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 2048


class LLMClient(ABC):
    """抽象 LLM 客户端接口。"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """给定 prompt，返回一个文本输出。"""
        raise NotImplementedError


class DummyLLMClient(LLMClient):
    """
    用于开发和测试的模拟 LLM 客户端。
    生产环境请替换为真实的 LLM API 实现。
    """

    def generate(self, prompt: str) -> str:
        # 返回模拟输出用于测试
        return "DUMMY_PROMPT: " + prompt[:200]


class TaskModel(ABC):
    """任务模型，用来评估 prompt 在实际任务上的性能。"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def infer(self, full_prompt: str, input_text: str) -> str:
        """
        在给定 full_prompt + input_text 下产生模型输出。
        这里保持接口简单，具体解析逻辑由外部负责。
        """
        raise NotImplementedError


class DummyTaskModel(TaskModel):
    """
    用于开发和测试的模拟任务模型。
    生产环境请替换为真实的 LLM API 实现。
    """

    def infer(self, full_prompt: str, input_text: str) -> str:
        # 返回模拟输出用于测试
        return "YES"