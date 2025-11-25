#!/usr/bin/env python
"""
小规模测试运行 - 使用少量数据快速验证完整流程
"""
import json
from apo.pipeline import run_apo_pipeline

def create_small_dataset(task="gsm8k", n_train=50, n_test=10):
    """创建小规模测试数据集"""
    import os
    from apo.utils.data import load_jsonl

    # 读取完整数据集
    train_path = f"local_datasets/{task}/train.jsonl"
    test_path = f"local_datasets/{task}/test.jsonl"

    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    # 采样
    import random
    random.seed(42)
    train_sample = random.sample(train_data, min(n_train, len(train_data)))
    test_sample = random.sample(test_data, min(n_test, len(test_data)))

    # 创建临时目录
    temp_dir = f"local_datasets/{task}_small"
    os.makedirs(temp_dir, exist_ok=True)

    # 保存采样数据
    train_small_path = f"{temp_dir}/train.jsonl"
    test_small_path = f"{temp_dir}/test.jsonl"

    with open(train_small_path, "w", encoding="utf-8") as f:
        for item in train_sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_small_path, "w", encoding="utf-8") as f:
        for item in test_sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Created small dataset:")
    print(f"   Train: {len(train_sample)} samples → {train_small_path}")
    print(f"   Test: {len(test_sample)} samples → {test_small_path}")

    return train_small_path, test_small_path


def main():
    print("=" * 70)
    print("🧪 APO Small-Scale Test Run")
    print("=" * 70)
    print()
    print("This test will run APO with:")
    print("  - 50 training samples (instead of 7,473)")
    print("  - 10 test samples (instead of 1,319)")
    print("  - 1 optimization round (instead of 5)")
    print("  - Real LLM API calls")
    print()
    print("Estimated API calls:")
    print("  - Task model: ~50-100 calls")
    print("  - Optimizer LLM: ~10-20 calls")
    print()

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return

    # 创建小数据集
    print("\n📦 Preparing small dataset...")
    train_path, test_path = create_small_dataset(task="gsm8k", n_train=50, n_test=10)

    # 运行 APO
    print("\n🚀 Starting APO pipeline...")
    print("=" * 70)

    try:
        run_apo_pipeline(
            task="gsm8k",
            n_rounds=1,  # 只运行 1 轮
            train_path=train_path,
            test_path=test_path,
            use_real_llm=True  # 使用真实 LLM
        )

        print("\n" + "=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)
        print("\nCheck results:")
        print("  - Result file: results/gsm8k_ensemble_result.json")
        print("  - Generated prompts should show real improvements (not DUMMY_PROMPT)")
        print()
        print("Next steps:")
        print("  1. Review the results to see if prompts improved")
        print("  2. If satisfied, run full-scale optimization:")
        print("     python run_apo.py --task gsm8k --rounds 5")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Test failed!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
