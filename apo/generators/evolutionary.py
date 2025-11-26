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
                 n_zero_order: int = 5,
                 debug: bool = False):
        self.optimizer_llm = optimizer_llm
        self.n_mutation = n_mutation
        self.n_zero_order = n_zero_order
        self.debug = debug

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
        # Show all prompts in debug mode, otherwise only first 5
        limit = len(population) if self.debug else 5
        for i, p in enumerate(population[:limit]):
            parts.append(f"Prompt {i+1}:\n{p.text}\n")
        parts.append("Return ONLY the new prompt text.")
        return "\n".join(parts)

    def generate(self, population: List[PromptCandidate]) -> List[str]:
        if not population:
            return []

        # Mutation - 并发生成，带进度条
        mutation_candidates: List[str] = []
        n_mutations = min(self.n_mutation, len(population))
        if n_mutations > 0:
            mutation_prompts = [
                self._build_mutation_prompt(population[i].text)
                for i in range(n_mutations)
            ]
            mutation_results = self.optimizer_llm.generate_batch(
                mutation_prompts,
                max_workers=10,
                desc="Evolutionary Mutation"
            )
            mutation_candidates = [m.strip() for m in mutation_results if m.strip()]

        # Zero-order (crossover-like) - 并发生成，带进度条
        zero_candidates: List[str] = []
        if self.n_zero_order > 0:
            z_prompt = self._build_zero_order_prompt(population)
            zero_prompts = [z_prompt] * self.n_zero_order
            zero_results = self.optimizer_llm.generate_batch(
                zero_prompts,
                max_workers=10,
                desc="Evolutionary Zero-Order"
            )
            zero_candidates = [z.strip() for z in zero_results if z.strip()]

        return mutation_candidates + zero_candidates