#!/usr/bin/env python
"""测试并发生成功能"""
import time
from apo.utils.llm_api import LLMConfig, DummyLLMClient, LLMClient
from apo.generators.bad_case import BadCaseReflectionGenerator, Sample
from apo.generators.evolutionary import EvolutionaryReflectionGenerator, PromptCandidate


class SlowDummyLLMClient(LLMClient):
    """模拟慢速 LLM 客户端，用于测试并发效果"""

    def __init__(self, config: LLMConfig, delay: float = 0.5):
        super().__init__(config)
        self.delay = delay

    def generate(self, prompt: str) -> str:
        """模拟 API 调用延迟"""
        time.sleep(self.delay)
        return f"GENERATED_PROMPT_{len(prompt)}"


def test_concurrent_bad_case():
    """测试 BadCaseReflectionGenerator 的并发生成"""
    print("=" * 60)
    print("Testing BadCaseReflectionGenerator concurrent generation")
    print("=" * 60)

    # 创建 dummy LLM client
    config = LLMConfig(model_name="dummy", temperature=0.0)
    llm = DummyLLMClient(config, debug=False)

    # 创建生成器
    generator = BadCaseReflectionGenerator(llm, n_prompts=10, debug=False)

    # 创建测试数据
    base_prompt = "You are a helpful assistant. Task: {input}"
    bad_cases = [
        (Sample(input_text="Test input 1", label="True"), "False"),
        (Sample(input_text="Test input 2", label="False"), "True"),
    ]

    # 测试串行生成时间（对比）
    print("\n[串行生成] 模拟 10 个 prompts...")
    start = time.time()
    # 模拟串行调用
    serial_results = []
    for _ in range(10):
        result = llm.generate("test prompt")
        serial_results.append(result)
    serial_time = time.time() - start
    print(f"串行生成耗时: {serial_time:.2f} 秒")

    # 测试并发生成
    print("\n[并发生成] 生成 10 个 prompts...")
    start = time.time()
    results = generator.generate(base_prompt, bad_cases)
    concurrent_time = time.time() - start
    print(f"并发生成耗时: {concurrent_time:.2f} 秒")
    print(f"生成了 {len(results)} 个候选 prompts")
    print(f"加速比: {serial_time / concurrent_time:.2f}x")


def test_concurrent_evolutionary():
    """测试 EvolutionaryReflectionGenerator 的并发生成"""
    print("\n" + "=" * 60)
    print("Testing EvolutionaryReflectionGenerator concurrent generation")
    print("=" * 60)

    # 创建 dummy LLM client
    config = LLMConfig(model_name="dummy", temperature=0.0)
    llm = DummyLLMClient(config, debug=False)

    # 创建生成器
    generator = EvolutionaryReflectionGenerator(
        llm, n_mutation=5, n_zero_order=5, debug=False
    )

    # 创建测试种群
    population = [
        PromptCandidate(text=f"Prompt {i}", score=0.5 + i * 0.1)
        for i in range(5)
    ]

    # 测试串行生成时间（对比）
    print("\n[串行生成] 模拟 10 个 prompts (5 mutation + 5 zero-order)...")
    start = time.time()
    serial_results = []
    for _ in range(10):
        result = llm.generate("test prompt")
        serial_results.append(result)
    serial_time = time.time() - start
    print(f"串行生成耗时: {serial_time:.2f} 秒")

    # 测试并发生成
    print("\n[并发生成] 生成 10 个 prompts (5 mutation + 5 zero-order)...")
    start = time.time()
    results = generator.generate(population)
    concurrent_time = time.time() - start
    print(f"并发生成耗时: {concurrent_time:.2f} 秒")
    print(f"生成了 {len(results)} 个候选 prompts")
    print(f"加速比: {serial_time / concurrent_time:.2f}x")


def test_batch_generation():
    """测试基础的批量生成功能"""
    print("\n" + "=" * 60)
    print("Testing LLMClient.generate_batch()")
    print("=" * 60)

    config = LLMConfig(model_name="dummy", temperature=0.0)
    llm = DummyLLMClient(config, debug=False)

    prompts = [f"Test prompt {i}" for i in range(10)]

    print(f"\n[批量生成] 生成 {len(prompts)} 个 prompts...")
    start = time.time()
    results = llm.generate_batch(prompts, max_workers=10)
    batch_time = time.time() - start

    print(f"批量生成耗时: {batch_time:.2f} 秒")
    print(f"成功生成: {len(results)} 个结果")
    print(f"平均每个 prompt 耗时: {batch_time / len(prompts):.3f} 秒")


def test_with_realistic_delay():
    """使用模拟延迟测试并发效果和进度条显示"""
    print("\n" + "=" * 60)
    print("Testing with realistic API delay (0.5s per call)")
    print("=" * 60)

    config = LLMConfig(model_name="slow-dummy", temperature=0.0)
    llm = SlowDummyLLMClient(config, delay=0.5)

    # 创建生成器
    bad_gen = BadCaseReflectionGenerator(llm, n_prompts=10, debug=False)
    evo_gen = EvolutionaryReflectionGenerator(llm, n_mutation=5, n_zero_order=5, debug=False)

    # 创建测试数据
    base_prompt = "You are a helpful assistant. Task: {input}"
    bad_cases = [
        (Sample(input_text="Test input 1", label="True"), "False"),
        (Sample(input_text="Test input 2", label="False"), "True"),
    ]

    print("\n[测试 1] BadCaseReflectionGenerator with progress bar:")
    start = time.time()
    results = bad_gen.generate(base_prompt, bad_cases)
    elapsed = time.time() - start
    print(f"✓ 生成完成: {len(results)} 个候选, 耗时 {elapsed:.2f}s, 加速比 {5.0/elapsed:.1f}x\n")

    print("[测试 2] EvolutionaryReflectionGenerator with progress bar:")
    population = [
        PromptCandidate(text=f"Prompt {i}", score=0.5 + i * 0.1)
        for i in range(5)
    ]
    start = time.time()
    results = evo_gen.generate(population)
    elapsed = time.time() - start
    print(f"✓ 生成完成: {len(results)} 个候选, 耗时 {elapsed:.2f}s, 加速比 {5.0/elapsed:.1f}x")


if __name__ == "__main__":
    print("开始测试并发生成功能...\n")

    # 运行所有测试
    test_batch_generation()
    test_concurrent_bad_case()
    test_concurrent_evolutionary()

    # 使用真实延迟测试
    test_with_realistic_delay()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
