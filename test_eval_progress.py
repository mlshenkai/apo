#!/usr/bin/env python
"""测试评估函数的进度条功能"""
import time
from apo.pipeline import evaluate_prompt_on_dataset
from apo.utils.llm_api import LLMConfig, TaskModel


class SlowDummyTaskModel(TaskModel):
    """模拟慢速任务模型，用于演示进度条效果"""

    def __init__(self, config: LLMConfig, delay: float = 0.1):
        super().__init__(config)
        self.delay = delay

    def infer(self, full_prompt: str, input_text: str) -> str:
        """模拟 API 调用延迟"""
        time.sleep(self.delay)
        # 简单逻辑：包含特定词返回 True，否则 False
        return "True" if "true" in input_text.lower() else "False"


def test_evaluation_progress_bar():
    """测试评估函数的进度条显示"""

    print("=" * 80)
    print("评估函数进度条演示")
    print("=" * 80)
    print("\n模拟每次推理耗时 0.1 秒\n")

    # 创建模拟任务模型
    config = LLMConfig(model_name="slow-task", temperature=0.0)
    task_model = SlowDummyTaskModel(config, delay=0.1)

    # 创建测试数据集
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

    print("测试 1: 标准评估 (显示进度条)")
    print("-" * 80)
    start = time.time()
    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=prompt_text,
        task_model=task_model,
        dataset=dataset,
        debug=False,
        show_progress=True,
        desc="Evaluating Prompt"
    )
    elapsed = time.time() - start
    print(f"\n结果: score={score:.4f}, bad_cases={len(bad_cases)}, 耗时={elapsed:.2f}s\n")

    print("\n测试 2: 自定义描述 (轮次信息)")
    print("-" * 80)
    start = time.time()
    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=prompt_text,
        task_model=task_model,
        dataset=dataset,
        debug=False,
        show_progress=True,
        desc="Round 3: Candidate 5/10"
    )
    elapsed = time.time() - start
    print(f"\n结果: score={score:.4f}, bad_cases={len(bad_cases)}, 耗时={elapsed:.2f}s\n")

    print("\n测试 3: 测试集评估")
    print("-" * 80)
    start = time.time()
    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=prompt_text,
        task_model=task_model,
        dataset=dataset,
        debug=False,
        show_progress=True,
        desc="Test: Member 1/5"
    )
    elapsed = time.time() - start
    print(f"\n结果: score={score:.4f}, bad_cases={len(bad_cases)}, 耗时={elapsed:.2f}s\n")

    print("\n测试 4: 无进度条模式")
    print("-" * 80)
    print("(适用于需要静默输出的场景)")
    start = time.time()
    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=prompt_text,
        task_model=task_model,
        dataset=dataset[:5],  # 只测试前5个
        debug=False,
        show_progress=False  # 禁用进度条
    )
    elapsed = time.time() - start
    print(f"结果: score={score:.4f}, bad_cases={len(bad_cases)}, 耗时={elapsed:.2f}s\n")

    print("=" * 80)
    print("演示完成!")
    print("=" * 80)
    print("\n进度条特性:")
    print("✓ 实时显示评估进度")
    print("✓ 动态显示当前错误数 (errors=N)")
    print("✓ 显示处理速度 (samples/s)")
    print("✓ 预估剩余时间")
    print("✓ 支持自定义描述文本")
    print("✓ 可选择开启/关闭")


if __name__ == "__main__":
    test_evaluation_progress_bar()
