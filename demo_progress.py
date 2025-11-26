#!/usr/bin/env python
"""演示并发生成的进度条功能"""
import time
from apo.utils.llm_api import LLMConfig, LLMClient
from apo.generators.bad_case import BadCaseReflectionGenerator, Sample
from apo.generators.evolutionary import EvolutionaryReflectionGenerator, PromptCandidate


class SlowDummyLLMClient(LLMClient):
    """模拟慢速 LLM 客户端，用于演示进度条效果"""

    def __init__(self, config: LLMConfig, delay: float = 0.3):
        super().__init__(config)
        self.delay = delay

    def generate(self, prompt: str) -> str:
        """模拟 API 调用延迟"""
        time.sleep(self.delay)
        return f"IMPROVED_PROMPT_{len(prompt)}"


def demo_progress_bars():
    """演示各种生成器的进度条"""

    print("=" * 80)
    print("APO 并发生成进度条演示")
    print("=" * 80)
    print("\n模拟每次 API 调用耗时 0.3 秒\n")

    # 初始化
    config = LLMConfig(model_name="demo", temperature=0.0)
    llm = SlowDummyLLMClient(config, delay=0.3)

    # ========== 演示 1: BadCaseReflectionGenerator ==========
    print("\n" + "=" * 80)
    print("演示 1: BadCase Reflection Generator (分析失败案例生成改进 prompt)")
    print("=" * 80)

    bad_gen = BadCaseReflectionGenerator(llm, n_prompts=15, debug=False)
    base_prompt = "Classify the following text as True or False.\nInput: {input}\nOutput:"
    bad_cases = [
        (Sample(input_text="The sky is green", label="False"), "True"),
        (Sample(input_text="Water is dry", label="False"), "True"),
        (Sample(input_text="Dogs can fly", label="False"), "True"),
    ]

    print(f"\n发现 {len(bad_cases)} 个失败案例，正在生成 {bad_gen.n_prompts} 个改进候选...\n")
    start = time.time()
    candidates = bad_gen.generate(base_prompt, bad_cases)
    elapsed = time.time() - start

    print(f"\n✓ 完成! 生成了 {len(candidates)} 个候选 prompts")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  串行需要: {bad_gen.n_prompts * 0.3:.1f} 秒")
    print(f"  加速比: {(bad_gen.n_prompts * 0.3) / elapsed:.1f}x")

    # ========== 演示 2: EvolutionaryReflectionGenerator (Mutation) ==========
    print("\n" + "=" * 80)
    print("演示 2: Evolutionary Reflection Generator (变异与交叉)")
    print("=" * 80)

    evo_gen = EvolutionaryReflectionGenerator(llm, n_mutation=8, n_zero_order=7, debug=False)
    population = [
        PromptCandidate(text=f"Prompt variant {i} with score {0.6 + i*0.05}", score=0.6 + i*0.05)
        for i in range(8)
    ]

    print(f"\n当前种群大小: {len(population)} 个 prompts")
    print(f"将生成 {evo_gen.n_mutation} 个变异候选 + {evo_gen.n_zero_order} 个交叉候选\n")

    start = time.time()
    candidates = evo_gen.generate(population)
    elapsed = time.time() - start

    total_prompts = evo_gen.n_mutation + evo_gen.n_zero_order
    print(f"\n✓ 完成! 生成了 {len(candidates)} 个候选 prompts")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  串行需要: {total_prompts * 0.3:.1f} 秒")
    print(f"  加速比: {(total_prompts * 0.3) / elapsed:.1f}x")

    # ========== 演示 3: 直接使用 generate_batch ==========
    print("\n" + "=" * 80)
    print("演示 3: 直接批量生成 (generate_batch API)")
    print("=" * 80)

    test_prompts = [
        "Generate a prompt for task A",
        "Generate a prompt for task B",
        "Generate a prompt for task C",
        "Generate a prompt for task D",
        "Generate a prompt for task E",
    ]

    print(f"\n批量生成 {len(test_prompts)} 个不同的 prompts...\n")

    start = time.time()
    results = llm.generate_batch(test_prompts, max_workers=5, desc="Custom Batch Generation")
    elapsed = time.time() - start

    print(f"\n✓ 完成! 生成了 {len(results)} 个 prompts")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  串行需要: {len(test_prompts) * 0.3:.1f} 秒")
    print(f"  加速比: {(len(test_prompts) * 0.3) / elapsed:.1f}x")

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("\n✓ 并发生成功能可以显著加速 prompt 优化过程")
    print("✓ tqdm 进度条提供实时反馈，方便监控生成进度")
    print("✓ 每个生成器都有自定义的进度条描述信息")
    print("✓ 在真实场景中 (API 延迟 0.5-2 秒)，加速效果更明显\n")


if __name__ == "__main__":
    demo_progress_bars()
