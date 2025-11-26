# apo/pipeline.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import os
import json
import numpy as np
from tqdm import tqdm

from apo.config import get_config
from apo.utils.llm_api import (
    LLMConfig,
    OpenAILLMClient,
    OpenAITaskModel,
    TaskModel,
    LLMClient,
)
from apo.utils.embedding import PromptEmbedder
from apo.utils.evaluation import task_metric
from apo.utils.data import load_jsonl, default_dataset_paths
from apo.generators.bad_case import BadCaseReflectionGenerator, Sample
from apo.generators.evolutionary import EvolutionaryReflectionGenerator, PromptCandidate
from apo.generators.hard_case import HardCaseTracker, HardCasePromptGenerator
from apo.search.bayesian import BayesianPromptSelector
from apo.search.mab import MABPromptSelector
from apo.ensemble.voting import EnsembleVoter, EnsembleMember
from apo.evaluators import get_evaluator


@dataclass
class PromptRecord:
    text: str
    score: Optional[float] = None


def evaluate_prompt_on_dataset(
    task: str,
    prompt_text: str,
    task_model: TaskModel,
    dataset: List[Dict[str, Any]],
    debug: bool = False,
    show_progress: bool = True,
    desc: str = "Evaluating",
    print_errors: bool = True,
    max_error_display: int = 10,
    max_workers: int = 10
) -> Tuple[float, List[Tuple[Sample, str]], List[Any], List[Any]]:
    """
    给定一个 prompt，在整个训练集上并行评估得到指标 + bad cases。
    返回：
      - score
      - bad_cases: List[(Sample, pred)]
      - preds
      - labels

    Args:
        task: 任务名称
        prompt_text: 待评估的 prompt 文本
        task_model: 任务模型
        dataset: 数据集
        debug: 是否启用调试模式
        show_progress: 是否显示进度条
        desc: 进度条描述
        print_errors: 是否打印错误详情
        max_error_display: 最多显示的错误数量
        max_workers: 并发推理的最大线程数
    """
    if not dataset:
        return 0.0, [], [], []

    # 获取对应任务的 evaluator
    evaluator = get_evaluator(task)

    # 准备批量推理的输入
    input_texts = [item["input"] for item in dataset]
    labels = [item["label"] for item in dataset]

    # 构建完整的 prompt（假设 prompt_text 中有 {input} 占位符）
    full_prompts = [prompt_text.replace("{input}", input_text) for input_text in input_texts]

    # 并行推理
    raw_outputs = task_model.infer_batch(
        full_prompts=full_prompts,
        input_texts=input_texts,
        max_workers=max_workers,
        desc=desc,
        show_progress=show_progress
    )

    # 使用 evaluator 解析预测结果和标准化标签
    preds = [evaluator.parse_pred(output) for output in raw_outputs]
    normalized_labels = [evaluator.normalize_label(label) for label in labels]

    # 收集 bad cases 和调试信息
    bad_cases: List[Tuple[Sample, str]] = []
    error_count = 0

    for i, (pred, label, raw_output, input_text) in enumerate(zip(preds, normalized_labels, raw_outputs, input_texts)):
        if debug and i < 3:  # Only print first 3 samples in debug mode
            tqdm.write(f"[DEBUG] Sample {i}: raw_output={raw_output}, parsed_pred={pred}, normalized_label={label}")

        if pred != label:
            bad_cases.append((Sample(input_text=input_text, label=label), pred))
            error_count += 1

            # 打印错误详情
            if print_errors and error_count <= max_error_display:
                error_msg = (
                    f"[Error] Sample {i}: "
                    f"Expected '{label}', Got '{pred}' (raw: '{raw_output[:50]}...') | "
                    f"Input: {input_text[:100]}{'...' if len(input_text) > 100 else ''}"
                )
                tqdm.write(error_msg)

    score = task_metric(task, preds, normalized_labels)

    # 打印评估摘要
    if show_progress:
        tqdm.write(f"✓ Evaluation complete: score={score:.4f}, bad_cases={len(bad_cases)}/{len(dataset)}")

        if print_errors and len(bad_cases) > max_error_display:
            tqdm.write(f"[Info] Showing first {max_error_display} errors. "
                      f"Total errors: {len(bad_cases)}")

    return score, bad_cases, preds, normalized_labels


def run_apo_pipeline(
    task: str,
    n_rounds: Optional[int] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    debug: bool = False,
):
    # 加载配置
    config = get_config()

    # 使用配置中的默认值
    if n_rounds is None:
        n_rounds = config.optimization.default_rounds

    # 1. 数据路径
    print(f"[DEBUG] Starting APO pipeline for task: {task}, rounds: {n_rounds}")
    if train_path is None or test_path is None:
        train_path, test_path = default_dataset_paths(task)

    print(f"[DEBUG] Loading data from train: {train_path}, test: {test_path}")
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    print(f"[DEBUG] Loaded {len(train_data)} train samples, {len(test_data)} test samples")

    # 2. 初始化 LLM（使用配置）
    print(f"[DEBUG] Initializing optimizer LLM and task model")

    # 使用真实的 OpenAI API 客户端
    optimizer_llm: LLMClient = OpenAILLMClient(
        LLMConfig(
            model_name=config.optimizer_llm.model,
            temperature=config.optimizer_llm.temperature,
            max_tokens=config.optimizer_llm.max_tokens
        ),
        api_key=config.optimizer_llm.api_key,
        base_url=config.optimizer_llm.base_url
    )
    task_model: TaskModel = OpenAITaskModel(
        LLMConfig(
            model_name=config.task_model.model_name,
            temperature=config.task_model.temperature,
            max_tokens=config.task_model.max_tokens
        ),
        api_key=config.task_model.api_key,
        base_url=config.task_model.base_url
    )

    print(f"[DEBUG] Models initialized successfully")

    # 3. 读取 initial prompt（这里只是示例，真实应从 prompts/initial/<task>.txt 读取）
    init_prompt_path = os.path.join("prompts", "initial", f"{task}.txt")
    if os.path.exists(init_prompt_path):
        with open(init_prompt_path, "r", encoding="utf-8") as f:
            base_prompt_text = f.read()
        print(f"[DEBUG] Loaded initial prompt from {init_prompt_path}")
    else:
        base_prompt_text = "## Task\nSolve the problem.\n## Prediction\nInput: {input}\nOutput:"
        print(f"[DEBUG] Using default initial prompt")

    if debug:
        print(f"[DEBUG] Initial prompt:\n{base_prompt_text}")
    else:
        print(f"[DEBUG] Initial prompt preview: {base_prompt_text[:100]}...")
    prompt_population: List[PromptRecord] = [PromptRecord(text=base_prompt_text)]
    hard_tracker = HardCaseTracker(max_size=300)

    # 4. 构建生成器 & 搜索器（使用配置）
    bad_gen = BadCaseReflectionGenerator(
        optimizer_llm,
        n_prompts=config.optimization.default_n_prompts,
        n_iters=config.optimization.default_n_iters,
        debug=debug
    )
    evo_gen = EvolutionaryReflectionGenerator(optimizer_llm, n_mutation=5, n_zero_order=5, debug=debug)
    hard_gen = HardCasePromptGenerator(optimizer_llm, k=10, debug=debug)

    embedder = PromptEmbedder(model_name=config.embedding.model_name)
    bayes_selector = BayesianPromptSelector(
        xi=config.search.bayesian_xi,
        n_select=config.search.bayesian_n_select
    )
    mab_selector = MABPromptSelector(
        n_clusters=config.search.mab_n_clusters,
        c=1.0,
        n_rounds=config.search.mab_n_rounds,
        per_round=2
    )

    # 5. 历史已评估 prompt embedding / score（供 Bayesian 使用）
    evaluated_embs: List[np.ndarray] = []
    evaluated_scores: List[float] = []

    # 6. 优化轮次
    for round_id in range(n_rounds):
        print(f"=== Round {round_id} ===")
        # 6.1 选择当前 best prompt
        best_prompt = max(
            prompt_population,
            key=lambda pr: pr.score if pr.score is not None else -1.0
        )

        # 如果是第一轮，先评估一次 base prompt
        if best_prompt.score is None:
            score, bad_cases, _, _ = evaluate_prompt_on_dataset(
                task, best_prompt.text, task_model, train_data, debug,
                desc=f"Round {round_id}: Initial Eval"
            )
            best_prompt.score = score

            # 更新 hard-case tracker
            for s, pred in bad_cases:
                hard_tracker.update(s, best_prompt.text)

            # 更新 evaluated emb / score
            emb = embedder.encode([best_prompt.text])[0]
            evaluated_embs.append(emb)
            evaluated_scores.append(score)

        # 6.2 生成 candidate prompts
        # 6.2.1 Bad-case Reflection 使用上一轮 best prompt + 最新 bad cases
        # 这里简单重算一次 bad cases，你也可以缓存
        _, bad_cases, _, _ = evaluate_prompt_on_dataset(
            task, best_prompt.text, task_model, train_data, debug,
            desc=f"Round {round_id}: Find Bad Cases"
        )
        bad_candidates = bad_gen.generate(best_prompt.text, bad_cases)
        print(f"[Round {round_id}] Generated {len(bad_candidates)} bad-case candidates")

        # 6.2.2 Evolutionary Reflection 从当前 population 生成
        print(f"[Round {round_id}] Stage: Evolutionary Reflection")
        evo_candidates = evo_gen.generate(
            [PromptCandidate(text=pr.text, score=pr.score) for pr in prompt_population]
        )
        print(f"[Round {round_id}] Generated {len(evo_candidates)} evolutionary candidates")

        # 6.2.3 Hard-case 跟踪生成
        print(f"[Round {round_id}] Stage: Hard-case Tracking")
        hard_prompt = hard_gen.generate(hard_tracker)
        hard_candidates = [hard_prompt] if hard_prompt is not None else []
        print(f"[Round {round_id}] Generated {len(hard_candidates)} hard-case candidates")

        candidate_texts = bad_candidates + evo_candidates + hard_candidates
        print(f"[Round {round_id}] Generated {len(candidate_texts)} candidates.")

        # 去掉重复
        existing_texts = {pr.text for pr in prompt_population}
        candidate_texts = [t for t in candidate_texts if t not in existing_texts]

        if not candidate_texts:
            print(f"[Round {round_id}] No new candidates, stopping.")
            break

        # 6.3 搜索：Bayesian + MAB 选出需要评估的 subset
        print(f"[Round {round_id}] Stage: Search (Bayesian + MAB)")
        cand_embs = embedder.encode(candidate_texts)

        if evaluated_embs:
            eval_emb_arr = np.stack(evaluated_embs, axis=0)
            eval_score_arr = np.array(evaluated_scores, dtype=float)
        else:
            eval_emb_arr = np.zeros((0, cand_embs.shape[1]), dtype=float)
            eval_score_arr = np.zeros((0,), dtype=float)

        bayes_idx = bayes_selector.select(cand_embs, eval_emb_arr, eval_score_arr)
        print(f"[Round {round_id}] Bayesian selected {len(bayes_idx)} candidates")
        mab_idx = mab_selector.select(cand_embs)
        print(f"[Round {round_id}] MAB selected {len(mab_idx)} candidates")
        selected_idx = sorted(set(bayes_idx + mab_idx))
        print(f"[Round {round_id}] Total selected: {len(selected_idx)} candidates for evaluation.")

        # 6.4 对选中 prompts 在训练集上评估
        for i, idx in enumerate(selected_idx):
            text = candidate_texts[idx]
            score, bad_cases, _, _ = evaluate_prompt_on_dataset(
                task, text, task_model, train_data, debug,
                desc=f"Round {round_id}: Candidate {i+1}/{len(selected_idx)}"
            )
            prompt_population.append(PromptRecord(text=text, score=score))
            emb = cand_embs[idx]
            evaluated_embs.append(emb)
            evaluated_scores.append(score)

            for s, pred in bad_cases:
                hard_tracker.update(s, text)

        # 6.5 打印当前最优
        best_now = max(
            prompt_population,
            key=lambda pr: pr.score if pr.score is not None else -1.0
        )
        print(f"[Round {round_id}] Best score so far: {best_now.score:.4f}")

    # 7. 构建 Ensemble（使用配置）
    print("=== Building ensemble ===")
    prompt_population_sorted = sorted(
        prompt_population,
        key=lambda pr: pr.score if pr.score is not None else -1.0,
        reverse=True
    )
    top_k = config.ensemble.top_k
    ensemble_members: List[EnsembleMember] = []

    # 在 train 上做一份"验证"预测（这里粗暴复用全部 train）
    for i, pr in enumerate(prompt_population_sorted[:top_k]):
        _, _, preds, _ = evaluate_prompt_on_dataset(
            task, pr.text, task_model, train_data, debug,
            desc=f"Ensemble: Member {i+1}/{top_k} on Train"
        )
        ensemble_members.append(
            EnsembleMember(prompt_text=pr.text, preds_on_val=preds, score=pr.score)
        )

    # 使用任务对应的评估指标优化权重（使用配置）
    from apo.utils.evaluation import task_metric

    labels_train = [item["label"] for item in train_data]
    metric_fn = lambda preds, labels: task_metric(task, preds, labels)
    voter = EnsembleVoter(metric_fn)
    weights = voter.optimize_weights(
        ensemble_members,
        labels_train,
        w_min=config.ensemble.w_min,
        n_steps=config.ensemble.n_steps
    )

    print("Ensemble weights:", weights)

    # 8. 在测试集上评价 Ensemble
    print("=== Evaluating ensemble on test set ===")
    # 构造每个成员在 test 上的预测
    member_preds_test: List[List[Any]] = []
    labels_test = [item["label"] for item in test_data]
    for i, member in enumerate(ensemble_members):
        _, _, preds_t, _ = evaluate_prompt_on_dataset(
            task, member.prompt_text, task_model, test_data, debug,
            desc=f"Test: Member {i+1}/{len(ensemble_members)}"
        )
        member_preds_test.append(preds_t)
    preds_matrix = np.array(member_preds_test)  # (M, N)

    final_preds: List[Any] = []
    M, N = preds_matrix.shape
    for i in range(N):
        label_weights: Dict[Any, float] = {}
        for j in range(M):
            label = preds_matrix[j, i]
            label_weights[label] = label_weights.get(label, 0.0) + float(weights[j])
        best_label = max(label_weights.items(), key=lambda kv: kv[1])[0]
        final_preds.append(best_label)

    final_score = task_metric(task, final_preds, labels_test)
    print(f"[TEST] Ensemble score on task {task}: {final_score:.4f}")

    # 9. 保存结果
    os.makedirs("results", exist_ok=True)
    result_path = os.path.join("results", f"{task}_ensemble_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": task,
                "score": final_score,
                "weights": list(map(float, weights)),
                "members": [
                    {"score": m.score, "prompt": m.prompt_text} for m in ensemble_members
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved ensemble result to {result_path}")