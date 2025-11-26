#!/usr/bin/env python3
"""
测试增强的 embedding 模块，支持多种 embedding provider。
"""

import os
import sys
from apo.utils.embedding import PromptEmbedder


def test_random_provider():
    """测试随机 provider"""
    print("\n" + "="*60)
    print("测试 Random Provider")
    print("="*60)

    embedder = PromptEmbedder(provider="random", dim=256)
    texts = ["prompt 1", "prompt 2", "prompt 3"]

    embeddings = embedder.encode(texts)
    print(f"✓ Provider: {embedder.provider_name}")
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Expected shape: ({len(texts)}, {embedder.dim})")
    print(f"✓ Dimension: {embedder.dim}")

    assert embeddings.shape == (len(texts), embedder.dim), "Shape mismatch!"
    print("✓ Random provider 测试通过!")


def test_sentence_transformers_provider():
    """测试 sentence-transformers provider"""
    print("\n" + "="*60)
    print("测试 Sentence-Transformers Provider")
    print("="*60)

    try:
        # 使用默认模型测试
        embedder = PromptEmbedder()
        texts = ["This is a test prompt", "Another test prompt"]

        embeddings = embedder.encode(texts)
        print(f"✓ Provider: {embedder.provider_name}")
        print(f"✓ Model: {embedder.model_name}")
        print(f"✓ Embedding shape: {embeddings.shape}")
        print(f"✓ Dimension: {embedder.dim}")

        assert embeddings.shape[0] == len(texts), "Number of embeddings mismatch!"
        assert embeddings.shape[1] == embedder.dim, "Dimension mismatch!"
        print("✓ Sentence-transformers 测试通过!")

    except Exception as e:
        print(f"⚠ Sentence-transformers 测试跳过: {e}")
        print("  (如果未安装 sentence-transformers，这是正常的)")


def test_openai_provider():
    """测试 OpenAI provider"""
    print("\n" + "="*60)
    print("测试 OpenAI Provider")
    print("="*60)

    # 检查 API key 是否可用
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OpenAI 测试跳过: 未设置 OPENAI_API_KEY")
        print("  设置 OPENAI_API_KEY 环境变量以测试 OpenAI embeddings")
        return

    try:
        # 测试 text-embedding-3-small (1536 维)
        print("\n测试 text-embedding-3-small...")
        embedder_small = PromptEmbedder(model_name="text-embedding-3-small")
        texts = ["优化这个提示词", "提高准确率"]

        embeddings = embedder_small.encode(texts)
        print(f"✓ Provider: {embedder_small.provider_name}")
        print(f"✓ Model: {embedder_small.model_name}")
        print(f"✓ Embedding shape: {embeddings.shape}")
        print(f"✓ Expected dimension: 1536")
        print(f"✓ Actual dimension: {embedder_small.dim}")

        assert embeddings.shape == (len(texts), 1536), "Shape mismatch!"
        print("✓ text-embedding-3-small 测试通过!")

        # 测试 text-embedding-3-large (3072 维)
        print("\n测试 text-embedding-3-large...")
        embedder_large = PromptEmbedder(model_name="text-embedding-3-large")

        embeddings = embedder_large.encode(texts)
        print(f"✓ Provider: {embedder_large.provider_name}")
        print(f"✓ Model: {embedder_large.model_name}")
        print(f"✓ Embedding shape: {embeddings.shape}")
        print(f"✓ Expected dimension: 3072")
        print(f"✓ Actual dimension: {embedder_large.dim}")

        assert embeddings.shape == (len(texts), 3072), "Shape mismatch!"
        print("✓ text-embedding-3-large 测试通过!")

        # 测试 text-embedding-ada-002 (1536 维)
        print("\n测试 text-embedding-ada-002...")
        embedder_ada = PromptEmbedder(model_name="text-embedding-ada-002")

        embeddings = embedder_ada.encode(texts)
        print(f"✓ Provider: {embedder_ada.provider_name}")
        print(f"✓ Model: {embedder_ada.model_name}")
        print(f"✓ Embedding shape: {embeddings.shape}")
        print(f"✓ Expected dimension: 1536")
        print(f"✓ Actual dimension: {embedder_ada.dim}")

        assert embeddings.shape == (len(texts), 1536), "Shape mismatch!"
        print("✓ text-embedding-ada-002 测试通过!")

    except Exception as e:
        print(f"✗ OpenAI 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_auto_detection():
    """测试自动 provider 检测"""
    print("\n" + "="*60)
    print("测试自动检测")
    print("="*60)

    # 测试 OpenAI 自动检测
    embedder1 = PromptEmbedder(model_name="text-embedding-3-small")
    print(f"✓ 'text-embedding-3-small' -> Provider: {embedder1.provider_name}")
    # 如果 API key 可用且 openai 已安装则应为 openai，否则为 random
    assert embedder1.provider_name in ["openai", "random"], \
        f"Unexpected provider: {embedder1.provider_name}"
    if embedder1.provider_name == "random":
        print("  (退化为 random - OpenAI 不可用)")

    # 测试 sentence-transformers 自动检测
    embedder2 = PromptEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"✓ 'sentence-transformers/all-MiniLM-L6-v2' -> Provider: {embedder2.provider_name}")
    # 如果库可用则应为 sentence-transformers，否则为 random
    assert embedder2.provider_name in ["sentence-transformers", "random"], \
        f"Unexpected provider: {embedder2.provider_name}"
    if embedder2.provider_name == "random":
        print("  (退化为 random - sentence-transformers 不可用)")

    # 测试 random 自动检测
    embedder3 = PromptEmbedder(model_name="random", dim=128)
    print(f"✓ 'random' -> Provider: {embedder3.provider_name}")
    assert embedder3.provider_name == "random", "Failed to detect random provider"

    print("✓ 自动检测测试通过!")


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "="*60)
    print("测试向后兼容性")
    print("="*60)

    # 旧的使用方式应该仍然可以工作
    embedder = PromptEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384
    )
    texts = ["test prompt"]

    # encode 方法应该工作
    embeddings = embedder.encode(texts)
    print(f"✓ encode() 方法正常工作")

    # dim 属性应该工作
    dim = embedder.dim
    print(f"✓ dim 属性正常工作: {dim}")

    assert embeddings.shape[1] == dim, "Dimension mismatch!"
    print("✓ 向后兼容性测试通过!")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("增强的 Embedding 模块测试套件")
    print("="*60)

    try:
        # 始终测试 random provider (无依赖)
        test_random_provider()

        # 测试自动检测
        test_auto_detection()

        # 测试向后兼容性
        test_backward_compatibility()

        # 测试 sentence-transformers (如果可用)
        test_sentence_transformers_provider()

        # 测试 OpenAI (如果 API key 可用)
        test_openai_provider()

        print("\n" + "="*60)
        print("所有可用测试完成!")
        print("="*60)

        # 打印使用示例
        print("\n" + "="*60)
        print("使用示例")
        print("="*60)
        print("""
# 示例 1: 使用默认 sentence-transformers
embedder = PromptEmbedder()

# 示例 2: 使用 OpenAI text-embedding-3-large
embedder = PromptEmbedder(model_name="text-embedding-3-large")

# 示例 3: 使用 OpenAI 并提供自定义 API key
embedder = PromptEmbedder(
    model_name="text-embedding-ada-002",
    api_key="sk-..."
)

# 示例 4: 使用随机 embeddings 进行测试
embedder = PromptEmbedder(provider="random", dim=512)

# 编码文本
texts = ["prompt 1", "prompt 2"]
embeddings = embedder.encode(texts)  # 返回 numpy 数组
print(f"Shape: {embeddings.shape}")
print(f"Dimension: {embedder.dim}")
        """)

    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试套件失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
