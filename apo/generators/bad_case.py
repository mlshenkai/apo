# apo/generators/bad_case.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from apo.utils.llm_api import LLMClient


@dataclass
class Sample:
    input_text: str
    label: Any


class BadCaseReflectionGenerator:
    """
    Bad-Case Reflection 生成器。
    基于当前 prompt 的失败案例，使用优化 LLM 生成改进的 prompt。
    """

    def __init__(self, optimizer_llm: LLMClient, n_prompts: int = 10, n_iters: int = 3, debug: bool = False):
        self.optimizer_llm = optimizer_llm
        self.n_prompts = n_prompts
        self.n_iters = n_iters
        self.debug = debug

    def build_reflection_prompt(self, base_prompt: str,
                                bad_cases: List[Tuple[Sample, str]]) -> str:
        """
        构造给 GPT-4o 等优化器的 meta prompt。
        bad_cases: [(sample, model_output), ...]
        """
        parts = [
            "You are an expert prompt engineer.",
            "The following prompt is currently being used, but it fails on some examples.",
            "Your task is to analyze the failures and produce an improved prompt.",
            "",
            "=== CURRENT PROMPT ===",
            base_prompt,
            "",
            "=== FAILED EXAMPLES ===",
        ]
        # Show all bad cases in debug mode, otherwise only first 10
        limit = len(bad_cases) if self.debug else 10
        for i, (s, pred) in enumerate(bad_cases[:limit]):
            parts.append(f"Example {i+1}:")
            parts.append(f"Input: {s.input_text}")
            parts.append(f"Gold label: {s.label}")
            parts.append(f"Model prediction: {pred}")
            parts.append("")
        parts.append(
            "Please return ONLY the improved prompt text, without explanations."
        )
        return "\n".join(parts)

    def generate(self, base_prompt: str,
                 bad_cases: List[Tuple[Sample, str]]) -> List[str]:
        """
        返回多个 candidate prompts。
        使用并发生成来加速多个 prompt 的生成。
        """
        if not bad_cases:
            # 没有 bad case 时，直接返回 base_prompt 的轻微变体（可自定义）
            return [base_prompt]

        ref_prompt = self.build_reflection_prompt(base_prompt, bad_cases)

        # 使用并发批量生成，带进度条
        prompts = [ref_prompt] * self.n_prompts
        candidates = self.optimizer_llm.generate_batch(
            prompts,
            max_workers=10,
            desc=f"BadCase Reflection ({len(bad_cases)} failures)"
        )

        # 清理结果
        candidates = [c.strip() for c in candidates if c.strip()]

        return candidates