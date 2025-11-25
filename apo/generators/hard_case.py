# apo/generators/hard_case.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from apo.utils.llm_api import LLMClient
from apo.generators.bad_case import Sample


@dataclass
class HardCaseEntry:
    sample: Sample
    error_count: int = 0
    failed_prompts: List[str] = field(default_factory=list)


class HardCaseTracker:
    """
    负责全局跟踪 hard cases（出现错误次数高的样本）。
    """

    def __init__(self, max_size: int = 300):
        self.max_size = max_size
        self._storage: Dict[str, HardCaseEntry] = {}

    def _key(self, s: Sample) -> str:
        return s.input_text  # 简单用文本做 key

    def update(self, sample: Sample, prompt_text: str):
        k = self._key(sample)
        if k not in self._storage:
            self._storage[k] = HardCaseEntry(sample=sample, error_count=0,
                                             failed_prompts=[])
        entry = self._storage[k]
        entry.error_count += 1
        entry.failed_prompts.append(prompt_text)

        # 简单截断：按 error_count 排序保留 top max_size
        if len(self._storage) > self.max_size:
            items = sorted(
                self._storage.items(),
                key=lambda kv: kv[1].error_count,
                reverse=True
            )
            self._storage = dict(items[: self.max_size])

    def top_k(self, k: int = 10) -> List[HardCaseEntry]:
        items = sorted(
            self._storage.values(),
            key=lambda e: e.error_count,
            reverse=True
        )
        return items[:k]


class HardCasePromptGenerator:
    """
    Hard-Case Tracking 生成器。
    针对持续失败的难例样本生成专门优化的提示词。
    """

    def __init__(self, optimizer_llm: LLMClient, k: int = 10):
        self.optimizer_llm = optimizer_llm
        self.k = k

    def build_meta_prompt(self, hard_cases: List[HardCaseEntry]) -> str:
        parts = [
            "You are an expert prompt engineer.",
            "We have some HARD cases where many prompts failed.",
            "Your task is to design a NEW prompt that can handle these cases correctly.",
            "",
            "=== HARD CASES ===",
        ]
        for i, entry in enumerate(hard_cases):
            parts.append(
                f"Case {i+1}:\n"
                f"Input: {entry.sample.input_text}\n"
                f"Gold label: {entry.sample.label}\n"
                f"Failure times: {entry.error_count}\n"
                f"Failed prompts (examples):"
            )
            for j, p in enumerate(entry.failed_prompts[:3]):
                parts.append(f"- Prompt {j+1}: {p[:200]}...")
            parts.append("")
        parts.append(
            "Please output ONLY the new improved prompt text that is robust "
            "and generalizable to such hard cases."
        )
        return "\n".join(parts)

    def generate(self, tracker: HardCaseTracker) -> Optional[str]:
        hard_cases = tracker.top_k(self.k)
        if not hard_cases:
            return None
        meta_prompt = self.build_meta_prompt(hard_cases)
        new_prompt = self.optimizer_llm.generate(meta_prompt)
        return new_prompt.strip()