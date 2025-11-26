# apo/utils/llm_api.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


@dataclass
class LLMConfig:
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 2048


class LLMClient(ABC):
    """抽象 LLM 客户端接口。"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """给定 prompt，返回一个文本输出。"""
        raise NotImplementedError

    def generate_batch(self, prompts: List[str], max_workers: int = 10,
                       desc: str = "Generating", show_progress: bool = True) -> List[str]:
        """
        并发生成多个 prompts。

        Args:
            prompts: 要生成的 prompt 列表
            max_workers: 最大并发线程数
            desc: 进度条描述文本
            show_progress: 是否显示进度条

        Returns:
            生成的文本列表，顺序与输入 prompts 一致
        """
        if not prompts:
            return []

        results: List[str] = [""] * len(prompts)  # 预分配结果列表

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，保存 future -> index 的映射
            future_to_idx = {
                executor.submit(self.generate, prompt): idx
                for idx, prompt in enumerate(prompts)
            }

            # 收集结果，使用 tqdm 显示进度
            progress_bar = tqdm(
                total=len(prompts),
                desc=desc,
                unit="prompt",
                disable=not show_progress,
                ncols=100
            )

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] Batch generation failed for prompt {idx}: {e}")
                    # 失败时使用空字符串
                    results[idx] = ""

                progress_bar.update(1)

            progress_bar.close()

        return results


class DummyLLMClient(LLMClient):
    """
    用于开发和测试的模拟 LLM 客户端。
    生产环境请替换为真实的 LLM API 实现。
    """

    def __init__(self, config: LLMConfig, debug: bool = False):
        super().__init__(config)
        self.debug = debug

    def generate(self, prompt: str) -> str:
        # 返回模拟输出用于测试
        print(f"[DEBUG] DummyLLMClient.generate() called with prompt length: {len(prompt)}")
        if self.debug:
            print(f"[DEBUG] Full prompt:\n{prompt}")
        else:
            print(f"[DEBUG] Prompt preview: {prompt[:100]}...")
        return "DUMMY_PROMPT: " + prompt[:200]


class OpenAILLMClient(LLMClient):
    """
    真实的 OpenAI API 客户端实现（支持 OpenAI 兼容接口）。
    用于生成和改进提示词的优化器 LLM。
    """

    def __init__(self, config: LLMConfig, api_key: str, base_url: str):
        super().__init__(config)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print(f"[INFO] Initialized OpenAI client: model={config.model_name}, base_url={base_url}")

    def generate(self, prompt: str) -> str:
        """调用 OpenAI API 生成文本"""
        try:
            print(f"[DEBUG] OpenAILLMClient.generate() calling API with prompt length: {len(prompt)}")

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            generated_text = response.choices[0].message.content
            print(f"[DEBUG] API response received, length: {len(generated_text)}")
            return generated_text

        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {e}")
            raise


class TaskModel(ABC):
    """任务模型，用来评估 prompt 在实际任务上的性能。"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def infer(self, full_prompt: str, input_text: str) -> str:
        """
        在给定 full_prompt + input_text 下产生模型输出。
        这里保持接口简单，具体解析逻辑由外部负责。
        """
        raise NotImplementedError

    def infer_batch(self, full_prompts: List[str], input_texts: List[str],
                    max_workers: int = 10, desc: str = "Inferring",
                    show_progress: bool = True) -> List[str]:
        """
        并发推理多个样本。

        Args:
            full_prompts: 完整的 prompt 列表
            input_texts: 输入文本列表
            max_workers: 最大并发线程数
            desc: 进度条描述文本
            show_progress: 是否显示进度条

        Returns:
            推理结果列表，顺序与输入一致
        """
        if not full_prompts:
            return []

        if len(full_prompts) != len(input_texts):
            raise ValueError(f"full_prompts and input_texts must have same length, "
                           f"got {len(full_prompts)} and {len(input_texts)}")

        results: List[str] = [""] * len(full_prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.infer, full_prompts[idx], input_texts[idx]): idx
                for idx in range(len(full_prompts))
            }

            # 收集结果，使用 tqdm 显示进度
            progress_bar = tqdm(
                total=len(full_prompts),
                desc=desc,
                unit="sample",
                disable=not show_progress,
                ncols=100
            )

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] Batch inference failed for sample {idx}: {e}")
                    # 失败时使用空字符串
                    results[idx] = ""

                progress_bar.update(1)

            progress_bar.close()

        return results


class DummyTaskModel(TaskModel):
    """
    用于开发和测试的模拟任务模型。
    生产环境请替换为真实的 LLM API 实现。
    """

    def __init__(self, config: LLMConfig, debug: bool = False):
        super().__init__(config)
        self.debug = debug

    def infer(self, full_prompt: str, input_text: str) -> str:
        # 返回模拟输出用于测试
        print(f"[DEBUG] DummyTaskModel.infer() called")
        print(f"[DEBUG] Full prompt length: {len(full_prompt)}, Input text length: {len(input_text)}")
        if self.debug:
            print(f"[DEBUG] Full input:\n{input_text}")
        else:
            print(f"[DEBUG] Input preview: {input_text[:100]}...")
        return "YES"


class OpenAITaskModel(TaskModel):
    """
    真实的 OpenAI API 任务模型实现。
    用于执行实际的任务推理。
    """

    def __init__(self, config: LLMConfig, api_key: str, base_url: str):
        super().__init__(config)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print(f"[INFO] Initialized OpenAI task model: model={config.model_name}, base_url={base_url}")

    def infer(self, full_prompt: str, input_text: str) -> str:
        """调用 API 执行任务推理"""
        try:
            # 构建完整的推理消息
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            content = response.choices[0].message.content
            result = content.strip() if content else ""
            return result

        except Exception as e:
            print(f"[ERROR] Task model API call failed: {e}")
            # 返回空字符串以避免中断整个流程
            return ""