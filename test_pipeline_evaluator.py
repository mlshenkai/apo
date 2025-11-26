#!/usr/bin/env python
"""
测试 pipeline 中的 evaluate_prompt_on_dataset 是否正确使用 evaluator
"""

from apo.pipeline import evaluate_prompt_on_dataset
from apo.utils.llm_api import DummyTaskModel, LLMConfig


def test_evaluate_with_gsm8k():
    """测试 GSM8K 任务的评估"""
    print("=== Testing GSM8K evaluation ===")

    # 创建测试数据
    dataset = [
        {"input": "What is 2+2?", "label": "4"},
        {"input": "What is 10-3?", "label": "7"},
        {"input": "What is 5*5?", "label": "25"},
    ]

    # 使用 DummyTaskModel（会返回包含数字的答案）
    config = LLMConfig(model_name="test", temperature=0.0)
    task_model = DummyTaskModel(config, debug=False)

    # 模拟 prompt
    prompt = "Solve: {input}\nAnswer:"

    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="gsm8k",
        prompt_text=prompt,
        task_model=task_model,
        dataset=dataset,
        debug=True,
        show_progress=False,
        max_workers=1
    )

    print(f"Score: {score:.4f}")
    print(f"Bad cases: {len(bad_cases)}/{len(dataset)}")
    print(f"Predictions: {preds}")
    print(f"Labels: {labels}")
    print()


def test_evaluate_with_binary():
    """测试 Binary 任务（liar）的评估"""
    print("=== Testing Binary (liar) evaluation ===")

    # 创建测试数据
    dataset = [
        {"input": "Is this true?", "label": "Yes"},
        {"input": "Is this false?", "label": "No"},
        {"input": "Is this correct?", "label": "Yes"},
    ]

    config = LLMConfig(model_name="test", temperature=0.0)
    task_model = DummyTaskModel(config, debug=False)

    prompt = "Answer: {input}\nResponse:"

    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="liar",
        prompt_text=prompt,
        task_model=task_model,
        dataset=dataset,
        debug=True,
        show_progress=False,
        max_workers=1
    )

    print(f"Score: {score:.4f}")
    print(f"Bad cases: {len(bad_cases)}/{len(dataset)}")
    print(f"Predictions: {preds}")
    print(f"Labels: {labels}")
    print()


def test_evaluate_with_wsc():
    """测试 WSC 任务的评估"""
    print("=== Testing WSC evaluation ===")

    # 创建测试数据
    dataset = [
        {"input": "Choose A or B", "label": "A"},
        {"input": "Select option", "label": "B"},
        {"input": "Pick one", "label": "A"},
    ]

    config = LLMConfig(model_name="test", temperature=0.0)
    task_model = DummyTaskModel(config, debug=False)

    prompt = "Question: {input}\nAnswer:"

    score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
        task="wsc",
        prompt_text=prompt,
        task_model=task_model,
        dataset=dataset,
        debug=True,
        show_progress=False,
        max_workers=1
    )

    print(f"Score: {score:.4f}")
    print(f"Bad cases: {len(bad_cases)}/{len(dataset)}")
    print(f"Predictions: {preds}")
    print(f"Labels: {labels}")
    print()


if __name__ == "__main__":
    test_evaluate_with_gsm8k()
    test_evaluate_with_binary()
    test_evaluate_with_wsc()
    print("All pipeline evaluator tests completed! ✓")
