# 并发生成功能说明

## 概述

APO 现已支持**并发 prompt 生成**，可将生成速度提升 **7-10 倍**。所有生成器都已优化，并配备实时进度条。

## 主要特性

### 1. 评估函数进度条

`evaluate_prompt_on_dataset()` 在评估 prompt 时显示实时进度:

```python
score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
    task="liar",
    prompt_text=prompt_text,
    task_model=task_model,
    dataset=dataset,
    debug=False,
    show_progress=True,  # 显示进度条
    desc="Round 3: Candidate 5/10"  # 自定义描述
)
```

**进度条显示**:
```
Round 3: Candidate 5/10: 100%|███████| 20/20 [00:02<00:00, 9.60sample/s, errors=1]
✓ Evaluation complete: score=0.9499, bad_cases=1
```

**特性**:
- 实时显示评估进度百分比
- 动态显示当前错误数 (`errors=N`)
- 显示处理速度 (`samples/s`)
- 预估剩余时间
- 支持自定义描述文本
- 可选择开启/关闭
- **实时打印错误详情**:
  - 使用 `[Error]` 前缀标识错误
  - 显示期望值 vs 实际值
  - 显示输入文本 (自动截断过长文本)
  - 可配置最大显示数量 (`max_error_display`)
  - 可完全关闭错误打印 (`print_errors=False`)

### 2. 并发批量生成 API

`LLMClient.generate_batch()` 提供了核心的并发生成能力:

```python
# 批量并发生成多个 prompts
prompts = ["prompt 1", "prompt 2", "prompt 3", ...]
results = llm.generate_batch(
    prompts,
    max_workers=10,  # 最大并发数
    desc="生成描述",   # 进度条标题
    show_progress=True  # 是否显示进度条
)
```

**实现细节**:
- 使用 `ThreadPoolExecutor` 实现并发
- 默认 10 个并发 worker
- 保证结果顺序与输入一致
- 单个失败不影响整体执行

### 3. 自动进度条显示

所有生成器和评估函数都自动显示 tqdm 进度条:

**生成器进度条**:
```
BadCase Reflection (15 failures): 100%|██████████| 15/15 [00:00<00:00, 25.78prompt/s]
Evolutionary Mutation: 100%|████████████████████| 8/8 [00:00<00:00, 26.28prompt/s]
Evolutionary Zero-Order: 100%|██████████████████| 7/7 [00:00<00:00, 23.27prompt/s]
```

**评估函数进度条**:
```
Round 0: Initial Eval: 100%|████████████| 100/100 [00:10<00:00, 9.60sample/s, errors=15]
[Error] Sample 3: Expected 'True', Got 'False' | Input: The sky is blue...
[Error] Sample 7: Expected 'False', Got 'True' | Input: Water is dry...
[Error] Sample 12: Expected 'True', Got 'False' | Input: Dogs can fly...
...
Round 1: Candidate 1/5: 100%|████████| 100/100 [00:10<00:00, 9.55sample/s, errors=12]
Test: Member 1/5: 100%|██████████████| 50/50 [00:05<00:00, 9.62sample/s, errors=3]
✓ Evaluation complete: score=0.8500, bad_cases=15/100
```

**进度条信息**:
- 完成进度百分比
- 可视化进度条
- 已完成/总数
- 预计剩余时间
- 当前速度 (prompts/s 或 samples/s)
- 额外信息 (如错误数)

### 4. 优化的生成器

所有三个主要生成器都已优化:

#### BadCaseReflectionGenerator
```python
# 自动使用并发生成
candidates = bad_gen.generate(base_prompt, bad_cases)
# 进度条: "BadCase Reflection (N failures)"
```

#### EvolutionaryReflectionGenerator
```python
# Mutation 和 Zero-order 分别并发生成
candidates = evo_gen.generate(population)
# 进度条: "Evolutionary Mutation" 和 "Evolutionary Zero-Order"
```

#### HardCasePromptGenerator
```python
# 单个 prompt 生成，暂不使用并发
hard_prompt = hard_gen.generate(tracker)
```

## 性能对比

### 串行 vs 并发

假设单次 API 调用耗时 **0.5 秒**:

| 生成数量 | 串行耗时 | 并发耗时 (10 workers) | 加速比 |
|---------|---------|---------------------|--------|
| 10 个   | 5.0 秒  | 0.5 秒              | 10x    |
| 15 个   | 7.5 秒  | 0.8 秒              | 9.4x   |
| 20 个   | 10.0 秒 | 1.0 秒              | 10x    |

### 实际测试结果

在模拟 API 延迟 0.3 秒的测试中:

```
BadCase (15 prompts):
  串行需要: 4.5 秒
  并发耗时: 0.62 秒
  加速比: 7.3x

Evolutionary (15 prompts):
  串行需要: 4.5 秒
  并发耗时: 0.61 秒
  加速比: 7.4x
```

## 使用示例

### 运行演示

查看并发生成和进度条效果:

```bash
python demo_progress.py
```

### 运行测试

验证并发生成功能:

```bash
python test_concurrent.py
```

测试评估进度条:

```bash
python test_eval_progress.py
```

测试错误显示功能:

```bash
python test_error_display.py
```

### 在管道中使用

无需修改代码，管道自动使用并发:

```bash
python run_apo.py --task liar --rounds 5
```

所有 prompt 生成都会自动并发执行，并显示进度条。

## 配置选项

### 调整并发数

修改生成器代码中的 `max_workers` 参数:

```python
candidates = self.optimizer_llm.generate_batch(
    prompts,
    max_workers=20,  # 增加并发数到 20
    desc="Custom Description"
)
```

### 禁用进度条

如果不需要进度条显示:

```python
candidates = self.optimizer_llm.generate_batch(
    prompts,
    show_progress=False  # 禁用进度条
)
```

### 错误显示配置

控制错误信息的显示:

```python
# 显示所有错误
score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
    task=task,
    prompt_text=prompt,
    task_model=model,
    dataset=data,
    print_errors=True,
    max_error_display=100  # 最多显示100个错误
)

# 只显示前5个错误
score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
    task=task,
    prompt_text=prompt,
    task_model=model,
    dataset=data,
    print_errors=True,
    max_error_display=5  # 只显示前5个
)

# 完全关闭错误打印
score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
    task=task,
    prompt_text=prompt,
    task_model=model,
    dataset=data,
    print_errors=False  # 不显示错误详情
)
```

## 错误处理

并发生成具有良好的容错性:

- **单个失败**: 返回空字符串，不中断其他请求
- **错误日志**: 使用 `tqdm.write()` 输出，不破坏进度条
- **保序返回**: 结果顺序始终与输入一致

示例错误输出:
```
[ERROR] Batch generation failed for prompt 3: API timeout
```

## 技术实现

### 核心代码结构

```python
# apo/utils/llm_api.py
def generate_batch(self, prompts: List[str], max_workers: int = 10) -> List[str]:
    results: List[str] = [""] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(self.generate, prompt): idx
            for idx, prompt in enumerate(prompts)
        }

        with tqdm(total=len(prompts), desc=desc) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                pbar.update(1)

    return results
```

### 关键设计

1. **线程池**: 使用 `ThreadPoolExecutor` 管理并发
2. **Future 映射**: 保持结果顺序
3. **as_completed**: 实时更新进度条
4. **预分配列表**: 高效的结果存储

## 依赖要求

确保安装了 tqdm:

```bash
pip install tqdm>=4.65.0
```

或使用项目 requirements:

```bash
pip install -r requirements.txt
```

## 最佳实践

1. **合理设置并发数**:
   - 通常 10-20 个 worker 即可
   - 过高可能导致 API 限流

2. **监控 API 限制**:
   - 注意 API 提供商的速率限制
   - 根据限制调整 `max_workers`

3. **处理超时**:
   - 考虑在 `generate()` 方法中添加超时处理
   - 避免长时间等待单个请求

4. **日志记录**:
   - 使用 `tqdm.write()` 而非 `print()`
   - 保持进度条显示整洁

## 故障排除

### 进度条不显示

检查 `show_progress` 参数是否为 `True`:

```python
llm.generate_batch(prompts, show_progress=True)
```

### 并发速度不理想

可能原因:
- API 本身有限流
- 网络延迟较高
- 并发数设置过低

解决方案:
- 增加 `max_workers`
- 检查网络连接
- 查看 API 限制文档

### 内存占用过高

如果一次性生成大量 prompts:
- 减少 `max_workers`
- 分批次生成
- 优化 prompt 大小

## 未来改进

可能的优化方向:

1. **自适应并发**: 根据 API 响应时间动态调整并发数
2. **异步 I/O**: 使用 asyncio 替代 ThreadPoolExecutor
3. **重试机制**: 自动重试失败的请求
4. **缓存机制**: 缓存重复的 prompt 生成结果
5. **批处理优化**: 支持 batch API (如 OpenAI batch endpoints)

## 反馈与贡献

如有问题或建议，欢迎提交 Issue 或 Pull Request。
