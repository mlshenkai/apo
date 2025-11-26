# Parallel Evaluation Implementation

This document describes the parallel evaluation feature added to the APO pipeline.

## Overview

The `evaluate_prompt_on_dataset` function has been enhanced to support concurrent evaluation of prompts across datasets, providing significant performance improvements (~10x speedup).

## Implementation Details

### Changes Made

1. **TaskModel.infer_batch()** (`apo/utils/llm_api.py:151-205`)
   - Added batch inference method to `TaskModel` base class
   - Uses ThreadPoolExecutor for concurrent API calls
   - Maintains result ordering consistent with input
   - Includes error handling and progress monitoring
   - Default: 10 concurrent workers

2. **evaluate_prompt_on_dataset()** (`apo/pipeline.py:36-118`)
   - Refactored from serial loop to batch processing
   - Prepares all prompts upfront
   - Calls `task_model.infer_batch()` for parallel inference
   - Post-processes results to collect bad cases
   - Added `max_workers` parameter (default: 10)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  evaluate_prompt_on_dataset()                           │
├─────────────────────────────────────────────────────────┤
│  1. Prepare batch inputs:                               │
│     - input_texts = [item["input"] for item in dataset] │
│     - full_prompts = [prompt.replace(...) for ...]      │
│                                                          │
│  2. Parallel inference:                                 │
│     - preds = task_model.infer_batch(...)               │
│       └──> ThreadPoolExecutor (10 workers)              │
│            ├─> task_model.infer(prompt[0], input[0])    │
│            ├─> task_model.infer(prompt[1], input[1])    │
│            ├─> ...                                       │
│            └─> task_model.infer(prompt[N], input[N])    │
│                                                          │
│  3. Post-process results:                               │
│     - Collect bad cases                                 │
│     - Calculate metrics                                 │
└─────────────────────────────────────────────────────────┘
```

## Performance Comparison

### Test Results (20 samples, 0.1s per sample)

| Execution Mode | Time (s) | Speedup |
|---------------|----------|---------|
| Serial (max_workers=1) | 2.08 | 1.00x |
| Parallel (max_workers=10) | 0.21 | 9.83x |

### Expected Impact on Full Pipeline

For a typical APO run with:
- 5 rounds
- 20 candidate prompts per round
- 100 training samples
- 0.5s per API call

**Before**: ~20 * 100 * 0.5 = 1000s per round = **~1.4 hours** total
**After**: ~20 * 100 * 0.5 / 10 = 100s per round = **~8 minutes** total

**Total speedup**: ~10x faster pipeline

## Usage

### Basic Usage

```python
from apo.pipeline import evaluate_prompt_on_dataset
from apo.utils.llm_api import OpenAITaskModel, LLMConfig

# Initialize task model
task_model = OpenAITaskModel(
    LLMConfig(model_name="gpt-4", temperature=0.0),
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)

# Evaluate with parallel inference (default)
score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
    task="liar",
    prompt_text="Your prompt with {input} placeholder",
    task_model=task_model,
    dataset=train_data,
    max_workers=10  # 10 concurrent workers
)
```

### Adjusting Concurrency

```python
# More aggressive parallelism (if API rate limits allow)
score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
    ...,
    max_workers=20  # 20 concurrent workers
)

# Conservative parallelism (for rate-limited APIs)
score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
    ...,
    max_workers=5  # 5 concurrent workers
)

# Serial execution (for debugging)
score, bad_cases, preds, labels = evaluate_prompt_on_dataset(
    ...,
    max_workers=1  # Single-threaded
)
```

## Testing

Run the test suite to verify the implementation:

```bash
python test_parallel_eval.py
```

Expected output:
- ✓ Serial and parallel results are identical
- ✓ Parallel execution achieves ~10x speedup
- ✓ Progress bars display correctly
- ✓ Error handling works properly

## Considerations

### API Rate Limits

When using external APIs (OpenAI, etc.), consider:
- **Rate limits**: Adjust `max_workers` to stay within API rate limits
- **Token limits**: Monitor token usage with parallel requests
- **Cost**: Parallel requests may increase costs due to higher throughput

### Memory Usage

- Batch processing prepares all prompts in memory
- For very large datasets (>10,000 samples), consider chunking
- Memory overhead: ~O(N) where N is dataset size

### Error Handling

- Individual inference failures return empty string
- Errors are logged but don't interrupt the batch
- Failed samples may affect metrics calculation

## Future Improvements

Potential enhancements:

1. **Adaptive concurrency**: Auto-adjust `max_workers` based on API response times
2. **Chunked processing**: Process datasets in chunks for better memory efficiency
3. **Retry logic**: Automatic retry for failed requests with exponential backoff
4. **Caching**: Cache inference results to avoid re-evaluation
5. **Batch API support**: Use native batch APIs when available (e.g., OpenAI Batch API)

## Related Files

- `apo/utils/llm_api.py` - TaskModel interface and implementations
- `apo/pipeline.py` - Main pipeline with evaluate_prompt_on_dataset
- `test_parallel_eval.py` - Test suite for parallel evaluation
- `CLAUDE.md` - Project documentation (Performance Features section)
