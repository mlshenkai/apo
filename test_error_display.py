#!/usr/bin/env python
"""测试错误显示功能"""
import time
from apo.pipeline import evaluate_prompt_on_dataset
from apo.utils.llm_api import LLMConfig, TaskModel


class ErrorProneTaskModel(TaskModel):
    """模拟容易出错的任务模型，用于演示错误显示"""

    def __init__(self, config: LLMConfig, delay: float = 0.05, error_rate: float = 0.3):
        super().__init__(config)
        self.delay = delay
        self.error_rate = error_rate
        self.call_count = 0

    def infer(self, full_prompt: str, input_text: str) -> str:
        """模拟 API 调用，随机产生错误"""
        time.sleep(self.delay)
        self.call_count += 1

        # 简单逻辑：包含特定词返回 True，否则 False
        # 但有一定概率返回错误答案
        correct_answer = "True" if "true" in input_text.lower() else "False"

        # 模拟错误：每3个样本就故意返回错误答案
        if self.call_count % 3 == 0:
            return "False" if correct_answer == "True" else "True"

        return correct_answer


def test_error_display():
    """测试错误显示功能"""

    print("=" * 100)
    print("错误显示功能演示")
    print("=" * 100)
    print("\n模拟一个有约30%错误率的任务模型\n")

    # 创建模拟任务模型
    config = LLMConfig(model_name="error-prone", temperature=0.0)
    task_model = ErrorProneTaskModel(config, delay=0.05, error_rate=0.3)

    # 创建测试数据集 - 20个样本
    dataset = [
        {"input": "The statement is true", "label": "True"},
        {"input": "This is false", "label": "False"},
        {"input": "True statement here", "label": "True"},
        {"input": "Another false one", "label": "False"},
        {"input": "True value", "label": "True"},
        {"input": "False information", "label": "False"},
        {"input": "This is true", "label": "True"},
        {"input": "Not true at all", "label": "False"},
        {"input": "Definitely true", "label": "True"},
        {"input": "Completely false", "label": "False"},
        {"input": "True fact", "label": "True"},
        {"input": "False claim", "label": "False"},
        {"input": "True again", "label": "True"},
        {"input": "False again", "label": "False"},
        {"input": "One more true", "label": "True"},
        {"input": "One more false", "label": "False"},
        {"input": "Final true", "label": "True"},
        {"input": "Final false", "label": "False"},
        {"input": "Extra true", "label": "True"},
        {"input": "Extra false", "label": "False"},
    ]

    prompt_text = "Classify the following as True or False: {input}"

    print("测试 1: 显示所有错误 (max_error_display=20)")
    print("-" * 100)
    start = time.time()
    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=prompt_text,
        task_model=task_model,
        dataset=dataset,
        debug=False,
        show_progress=True,
        desc="Test 1: All Errors",
        print_errors=True,
        max_error_display=20
    )
    elapsed = time.time() - start
    print(f"\n耗时: {elapsed:.2f}s\n")

    print("\n" + "=" * 100)
    print("测试 2: 只显示前5个错误 (max_error_display=5)")
    print("-" * 100)
    task_model.call_count = 0  # 重置计数器
    start = time.time()
    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=prompt_text,
        task_model=task_model,
        dataset=dataset,
        debug=False,
        show_progress=True,
        desc="Test 2: First 5 Errors",
        print_errors=True,
        max_error_display=5
    )
    elapsed = time.time() - start
    print(f"\n耗时: {elapsed:.2f}s\n")

    print("\n" + "=" * 100)
    print("测试 3: 不显示错误详情 (print_errors=False)")
    print("-" * 100)
    task_model.call_count = 0  # 重置计数器
    start = time.time()
    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=prompt_text,
        task_model=task_model,
        dataset=dataset,
        debug=False,
        show_progress=True,
        desc="Test 3: No Error Details",
        print_errors=False
    )
    elapsed = time.time() - start
    print(f"\n耗时: {elapsed:.2f}s\n")

    print("\n" + "=" * 100)
    print("测试 4: 大量错误场景 (50个样本)")
    print("-" * 100)

    # 创建更大的数据集
    large_dataset = dataset * 3  # 60个样本

    task_model.call_count = 0
    start = time.time()
    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=prompt_text,
        task_model=task_model,
        dataset=large_dataset,
        debug=False,
        show_progress=True,
        desc="Test 4: Large Dataset",
        print_errors=True,
        max_error_display=10  # 只显示前10个
    )
    elapsed = time.time() - start
    print(f"\n耗时: {elapsed:.2f}s\n")

    print("\n" + "=" * 100)
    print("功能总结")
    print("=" * 100)
    print("\n错误显示特性:")
    print("✓ 实时显示每个错误的详细信息")
    print("✓ 显示期望值、实际值和输入文本")
    print("✓ 可配置最大显示错误数量")
    print("✓ 超过限制时显示提示信息")
    print("✓ 可完全关闭错误详情显示")
    print("✓ 使用 tqdm.write() 不破坏进度条")
    print("\n错误信息格式:")
    print("  [Error] Sample N: Expected 'X', Got 'Y' | Input: <text>")


if __name__ == "__main__":
    test_error_display()
