#!/usr/bin/env python
"""
测试 API 连接是否正常工作
"""
from apo.config import get_config
from apo.utils.llm_api import OpenAILLMClient, OpenAITaskModel, LLMConfig

def test_optimizer_llm():
    """测试优化器 LLM"""
    print("=" * 60)
    print("Testing Optimizer LLM (GPT-4o)...")
    print("=" * 60)

    config = get_config()

    client = OpenAILLMClient(
        LLMConfig(
            model_name=config.optimizer_llm.model,
            temperature=config.optimizer_llm.temperature,
            max_tokens=512  # 使用较小的 token 限制进行测试
        ),
        api_key=config.optimizer_llm.api_key,
        base_url=config.optimizer_llm.base_url
    )

    test_prompt = """You are a prompt engineer. Generate an improved version of this prompt:

Original: "Solve this math problem: {input}"

Make it more effective for mathematical reasoning."""

    try:
        response = client.generate(test_prompt)
        print("\n✅ Optimizer LLM Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        return True
    except Exception as e:
        print(f"\n❌ Optimizer LLM Failed: {e}")
        return False


def test_task_model():
    """测试任务模型"""
    print("\n" + "=" * 60)
    print("Testing Task Model (DeepSeek-V3)...")
    print("=" * 60)

    config = get_config()

    client = OpenAITaskModel(
        LLMConfig(
            model_name=config.task_model.model_name,
            temperature=config.task_model.temperature,
            max_tokens=256  # 使用较小的 token 限制进行测试
        ),
        api_key=config.task_model.api_key,
        base_url=config.task_model.base_url
    )

    test_prompt = """Solve this math problem and output ONLY the final numerical answer.

Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Answer:"""

    test_input = "dummy_input"  # Not used in this case

    try:
        response = client.infer(test_prompt, test_input)
        print("\n✅ Task Model Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        return True
    except Exception as e:
        print(f"\n❌ Task Model Failed: {e}")
        return False


def main():
    print("\n🔧 API Connection Test")
    print("=" * 60)

    # 显示配置
    config = get_config()
    print(f"Optimizer Model: {config.optimizer_llm.model}")
    print(f"Optimizer Base URL: {config.optimizer_llm.base_url}")
    print(f"Task Model: {config.task_model.model_name}")
    print(f"Task Model Base URL: {config.task_model.base_url}")
    print()

    # 测试两个模型
    optimizer_ok = test_optimizer_llm()
    task_model_ok = test_task_model()

    # 总结
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Optimizer LLM: {'✅ PASS' if optimizer_ok else '❌ FAIL'}")
    print(f"Task Model: {'✅ PASS' if task_model_ok else '❌ FAIL'}")

    if optimizer_ok and task_model_ok:
        print("\n🎉 All tests passed! You can now run the full pipeline.")
        print("\nTo run with real LLMs:")
        print("  python run_apo.py --task gsm8k --rounds 2")
        print("\nTo run with dummy models (no API calls):")
        print("  python run_apo.py --task gsm8k --rounds 2 --no-use_real_llm")
    else:
        print("\n⚠️ Some tests failed. Please check your API configuration in .env")

    print("=" * 60)


if __name__ == "__main__":
    main()
