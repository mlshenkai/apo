#!/usr/bin/env python3
"""
测试 OpenAI text-embedding-3-large 模型
"""

import os
import sys
from dotenv import load_dotenv
from apo.utils.embedding import PromptEmbedder

# 加载 .env 文件，覆盖已存在的环境变量
load_dotenv(override=True)


def test_text_embedding_3_large():
    """测试 text-embedding-3-large embedding"""
    print("="*70)
    print("测试 OpenAI text-embedding-3-large")
    print("="*70)

    # 检查配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("EMBEDDING_MODEL_NAME")

    # 如果未设置环境变量，使用默认值
    if not model_name:
        model_name = "text-embedding-3-large"
        print("\n⚠️  未设置 EMBEDDING_MODEL_NAME，使用默认值: text-embedding-3-large")

    print(f"\n配置信息:")
    print(f"  模型: {model_name}")
    print(f"  API Key: {'已设置 (' + api_key[:20] + '...)' if api_key else '未设置'}")
    print(f"  Base URL: {base_url if base_url else '默认'}")

    if not api_key:
        print("\n❌ 错误: 未设置 OPENAI_API_KEY")
        print("请在 .env 文件中设置 OPENAI_API_KEY")
        return False

    try:
        print(f"\n初始化 PromptEmbedder...")
        embedder = PromptEmbedder(model_name=model_name)

        print(f"✓ Provider: {embedder.provider_name}")
        print(f"✓ 模型: {embedder.model_name}")
        print(f"✓ 维度: {embedder.dim}")

        # 测试用的中文文本
        test_texts = [
            "这是一个优秀的提示词，可以提高模型的性能",
            "我们需要优化这个算法以获得更好的结果",
            "深度学习模型在自然语言处理任务中表现出色",
            "贝叶斯优化是一种高效的超参数搜索方法"
        ]

        print(f"\n测试文本 ({len(test_texts)} 条):")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")

        print(f"\n正在生成 embeddings...")
        embeddings = embedder.encode(test_texts)

        print(f"\n✓ 成功生成 embeddings!")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Expected: ({len(test_texts)}, {embedder.dim})")
        print(f"  Dtype: {embeddings.dtype}")

        # 验证形状
        assert embeddings.shape == (len(test_texts), embedder.dim), \
            f"形状不匹配! 期望 ({len(test_texts)}, {embedder.dim}), 实际 {embeddings.shape}"

        # 计算相似度矩阵
        print(f"\n计算余弦相似度...")
        from numpy.linalg import norm
        import numpy as np

        def cosine_similarity(a, b):
            return np.dot(a, b) / (norm(a) * norm(b))

        print(f"\n相似度矩阵:")
        print("      ", end="")
        for i in range(len(test_texts)):
            print(f"  文本{i+1} ", end="")
        print()

        for i in range(len(test_texts)):
            print(f"文本{i+1}", end="")
            for j in range(len(test_texts)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                print(f"  {sim:.3f} ", end="")
            print()

        # 显示一些统计信息
        print(f"\nEmbedding 统计:")
        print(f"  最小值: {embeddings.min():.4f}")
        print(f"  最大值: {embeddings.max():.4f}")
        print(f"  平均值: {embeddings.mean():.4f}")
        print(f"  标准差: {embeddings.std():.4f}")

        # 检查向量范数
        print(f"\n向量范数 (L2 norm):")
        for i in range(len(test_texts)):
            vector_norm = norm(embeddings[i])
            print(f"  文本{i+1}: {vector_norm:.4f}")

        print("\n" + "="*70)
        print("✓ text-embedding-3-large 测试通过!")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_encoding():
    """测试批量编码"""
    print("\n" + "="*70)
    print("测试批量编码")
    print("="*70)

    try:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
        embedder = PromptEmbedder(model_name=model_name)

        # 测试不同大小的批次
        batch_sizes = [1, 5, 10]

        for batch_size in batch_sizes:
            texts = [f"测试文本 {i}" for i in range(batch_size)]
            print(f"\n批次大小: {batch_size}")

            embeddings = embedder.encode(texts)
            print(f"  ✓ Shape: {embeddings.shape}")
            assert embeddings.shape[0] == batch_size
            assert embeddings.shape[1] == embedder.dim

        print("\n✓ 批量编码测试通过!")
        return True

    except Exception as e:
        print(f"\n❌ 批量编码测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("OpenAI text-embedding-3-large 测试套件")
    print("="*70)

    results = []

    # 测试基本功能
    results.append(("text-embedding-3-large 基本测试", test_text_embedding_3_large()))

    # 测试批量编码
    results.append(("批量编码测试", test_batch_encoding()))

    # 打印总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*70)
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
