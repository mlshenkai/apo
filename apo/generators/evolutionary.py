# apo/generators/evolutionary.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from apo.utils.llm_api import LLMClient


@dataclass
class PromptCandidate:
    text: str
    score: Optional[float] = None


class EvolutionaryReflectionGenerator:
    """
    Evolutionary Reflection 生成器。
    包含变异（mutation）和零阶生成（zero-order generation）两种策略。
    """

    def __init__(self, optimizer_llm: LLMClient,
                 n_mutation: int = 5,
                 n_zero_order: int = 5):
        self.optimizer_llm = optimizer_llm
        self.n_mutation = n_mutation
        self.n_zero_order = n_zero_order

    def _build_mutation_prompt(self, prompt_text: str) -> str:
        return (
            "You are a prompt rewriting assistant.\n"
            "Please slightly modify the following prompt to improve clarity, robustness, "
            "and reasoning quality, while keeping the same task and output format.\n\n"
            "=== ORIGINAL PROMPT ===\n"
            f"{prompt_text}\n\n"
            "Return ONLY the new prompt."
        )

    def _build_zero_order_prompt(self, population: List[PromptCandidate]) -> str:
        parts = [
            "You are an expert prompt engineer.",
            "Below are several prompts that perform reasonably well on a task.",
            "Please design a NEW prompt that combines their strengths, with better reasoning, "
            "clarity, and robustness.",
            "",
            "=== EXISTING PROMPTS ===",
        ]
        for i, p in enumerate(population[:5]):
            parts.append(f"Prompt {i+1}:\n{p.text}\n")
        parts.append("Return ONLY the new prompt text.")
        return "\n".join(parts)

    def generate(self, population: List[PromptCandidate]) -> List[str]:
        if not population:
            return []

        # Mutation
        mutation_candidates: List[str] = []
        for i in range(min(self.n_mutation, len(population))):
            p = population[i]
            prompt = self._build_mutation_prompt(p.text)
            mutated = self.optimizer_llm.generate(prompt)
            mutation_candidates.append(mutated.strip())

        # Zero-order (crossover-like)
        zero_candidates: List[str] = []
        z_prompt = self._build_zero_order_prompt(population)
        for _ in range(self.n_zero_order):
            new_prompt = self.optimizer_llm.generate(z_prompt)
            zero_candidates.append(new_prompt.strip())

        return mutation_candidates + zero_candidates