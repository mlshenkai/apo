#!/usr/bin/env python3
"""
简化的 OpenAI embedding 测试，包含错误处理
"""

import os
from dotenv import load_dotenv
from apo.utils.embedding import PromptEmbedder

# 加载 .env 文件
load_dotenv(override=True)


def main():
    print("="*70)
    print("OpenAI text-embedding-3-large 快速测试")
    print("="*70)

    # 读取配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")

    print(f"\n配置:")
    print(f"  模型: {model_name}")
    print(f"  API Key: {api_key[:25]}..." if api_key else "  API Key: 未设置")
    print(f"  Base URL: {base_url}")

    if not api_key:
        print("\n❌ 错误: 未设置 OPENAI_API_KEY")
        return

    # 尝试不同的 base_url 配置
    test_configs = [
        ("原始 base_url", base_url),
        ("添加 /v1", base_url + "/v1" if base_url and not base_url.endswith("/v1") else None),
        ("官方 API", "https://api.openai.com/v1"),
    ]

    for config_name, test_base_url in test_configs:
        if test_base_url is None:
            continue

        print(f"\n" + "-"*70)
        print(f"尝试配置: {config_name}")
        print(f"  Base URL: {test_base_url}")

        try:
            # 创建embedder
            embedder = PromptEmbedder(
                model_name=model_name,
                base_url=test_base_url
            )

            print(f"  ✓ Provider: {embedder.provider_name}")
            print(f"  ✓ 维度: {embedder.dim}")

            # 测试简单的embedding
            test_texts = ["测试文本1", "测试文本2"]
            print(f"  正在生成 embeddings...")

            embeddings = embedder.encode(test_texts)

            print(f"  ✓ 成功!")
            print(f"  ✓ Shape: {embeddings.shape}")
            print(f"  ✓ 向量范数: {embeddings[0][:5]} ...")

            print(f"\n{'='*70}")
            print(f"✓ 成功! 使用配置: {config_name}")
            print(f"{'='*70}")
            return

        except Exception as e:
            print(f"  ✗ 失败: {type(e).__name__}")
            print(f"  错误信息: {str(e)[:100]}")
            continue

    print(f"\n{'='*70}")
    print("❌ 所有配置都失败了")
    print("="*70)
    print("\n建议:")
    print("1. 检查 OPENAI_BASE_URL 是否正确")
    print("2. 检查网络连接")
    print("3. 检查 API Key 是否有效")
    print("4. 尝试使用官方 API: https://api.openai.com/v1")


if __name__ == "__main__":
    main()
