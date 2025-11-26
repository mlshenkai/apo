#!/usr/bin/env python3
"""
测试 evaluate_prompt_on_dataset 的并行实现
"""
import time
from apo.utils.llm_api import LLMConfig, TaskModel
from apo.pipeline import evaluate_prompt_on_dataset
from apo.generators.bad_case import Sample


class TimedDummyTaskModel(TaskModel):
    """模拟任务模型，每次推理延迟一定时间以测试并行性能"""

    def __init__(self, config: LLMConfig, delay: float = 0.1):
        super().__init__(config)
        self.delay = delay
        self.call_count = 0

    def infer(self, full_prompt: str, input_text: str) -> str:
        """模拟推理，添加延迟"""
        time.sleep(self.delay)
        self.call_count += 1

        # 简单规则：如果输入包含"error"则返回错误标签
        if "error" in input_text.lower():
            return "False"
        return "True"


def test_parallel_evaluation():
    """测试并行评估功能"""

    print("=" * 60)
    print("测试并行 evaluate_prompt_on_dataset")
    print("=" * 60)

    # 创建测试数据（20个样本）
    dataset = []
    for i in range(20):
        if i % 5 == 0:  # 每5个样本有一个错误
            dataset.append({
                "input": f"Test sample {i} with error keyword",
                "label": "True"  # 期望 True，但模型会返回 False
            })
        else:
            dataset.append({
                "input": f"Test sample {i} normal case",
                "label": "True"  # 期望 True，模型也会返回 True
            })

    # 创建测试模型（每次推理延迟0.1秒）
    config = LLMConfig(model_name="test-model", temperature=0.0)
    task_model = TimedDummyTaskModel(config, delay=0.1)

    # 测试 prompt
    test_prompt = "Classify the input as True or False.\nInput: {input}\nOutput:"

    # 测试串行执行（max_workers=1）
    print("\n测试 1: 串行执行 (max_workers=1)")
    print("-" * 60)
    task_model.call_count = 0
    start_time = time.time()

    score_serial, bad_cases_serial, preds_serial, labels_serial = evaluate_prompt_on_dataset(
        task="liar",  # 使用 macro-f1 metric
        prompt_text=test_prompt,
        task_model=task_model,
        dataset=dataset,
        debug=False,
        show_progress=True,
        desc="Serial Evaluation",
        print_errors=True,
        max_error_display=5,
        max_workers=1
    )

    serial_time = time.time() - start_time
    print(f"\n串行执行完成:")
    print(f"  耗时: {serial_time:.2f} 秒")
    print(f"  得分: {score_serial:.4f}")
    print(f"  错误数: {len(bad_cases_serial)}/{len(dataset)}")
    print(f"  API 调用次数: {task_model.call_count}")

    # 测试并行执行（max_workers=10）
    print("\n\n测试 2: 并行执行 (max_workers=10)")
    print("-" * 60)
    task_model.call_count = 0
    start_time = time.time()

    score_parallel, bad_cases_parallel, preds_parallel, labels_parallel = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=test_prompt,
        task_model=task_model,
        dataset=dataset,
        debug=False,
        show_progress=True,
        desc="Parallel Evaluation",
        print_errors=True,
        max_error_display=5,
        max_workers=10
    )

    parallel_time = time.time() - start_time
    print(f"\n并行执行完成:")
    print(f"  耗时: {parallel_time:.2f} 秒")
    print(f"  得分: {score_parallel:.4f}")
    print(f"  错误数: {len(bad_cases_parallel)}/{len(dataset)}")
    print(f"  API 调用次数: {task_model.call_count}")

    # 验证结果一致性
    print("\n\n测试 3: 验证结果一致性")
    print("-" * 60)

    assert score_serial == score_parallel, "得分应该相同"
    assert len(bad_cases_serial) == len(bad_cases_parallel), "错误数应该相同"
    assert preds_serial == preds_parallel, "预测结果应该相同"

    print("✓ 串行和并行结果一致")

    # 计算加速比
    speedup = serial_time / parallel_time
    print(f"\n加速比: {speedup:.2f}x")

    # 理论加速比（考虑到有20个样本，10个并发）
    theoretical_speedup = min(10, len(dataset))
    print(f"理论加速比: ~{theoretical_speedup}x")

    if speedup > 1.5:
        print(f"✓ 并行执行显著加速 ({speedup:.2f}x)")
    else:
        print(f"⚠ 加速不明显，可能需要检查实现")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_parallel_evaluation()
