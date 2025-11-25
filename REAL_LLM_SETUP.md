# 真实 LLM API 配置完成 ✅

## 配置概览

您的 APO 系统现已成功配置为使用真实的 LLM API：

- **优化器模型**: `openai/gpt-4o` (通过 LiteLLM 代理)
- **任务模型**: `deepseek-v3` (通过 LiteLLM 代理)
- **API 基础地址**: `https://litellm.iyunquna.com/`

## 测试结果

✅ **API 连接测试通过**

```bash
python test_api.py
```

两个模型均正常响应：
- GPT-4o (优化器): 成功生成改进的提示词
- DeepSeek-V3 (任务执行): 成功解决数学问题 (72)

## 使用方法

### 1. 使用真实 LLM 运行 (默认)

```bash
# 完整运行 5 轮优化
python run_apo.py --task gsm8k --rounds 5

# 快速测试 2 轮
python run_apo.py --task gsm8k --rounds 2
```

### 2. 使用 Dummy 模式测试 (不调用 API)

```bash
python run_apo.py --task gsm8k --rounds 2 --no-use_real_llm
```

### 3. 支持的任务

- `liar` - 谎言检测
- `bbh` - Big-Bench Hard
- `ethos` - 道德分类
- `arsarcasm` - 讽刺检测
- `wsc` - Winograd Schema Challenge
- `gsm8k` - 数学问题求解 ✅

## 重要说明

### ⚠️ 数据集大小与成本

GSM8K 数据集：
- **训练集**: 7,473 个样本
- **测试集**: 1,319 个样本

每轮优化会：
1. 在训练集上评估多个 prompt (7,473 次推理)
2. 生成新的 prompt 候选 (~10-20 次生成)
3. 重复 n 轮

**预估 API 调用量 (5 轮优化)**:
- 任务模型调用: ~40,000+ 次 (DeepSeek-V3)
- 优化器调用: ~50-100 次 (GPT-4o)

### 💡 建议

1. **首次运行建议使用小数据集或少轮次**:
   ```bash
   python run_apo.py --task gsm8k --rounds 1
   ```

2. **使用采样数据进行快速实验**:
   - 可以修改代码限制训练集大小
   - 或创建采样版本的数据集

3. **监控 API 成本**:
   - 查看 LiteLLM 代理的使用统计
   - 根据预算调整轮次和数据量

## 代码改动说明

### 新增文件

1. **`test_api.py`**: API 连接测试脚本
2. **`REAL_LLM_SETUP.md`**: 本文档
3. **`TEST_REPORT.md`**: 系统测试报告

### 修改文件

1. **`apo/utils/llm_api.py`**:
   - 新增 `OpenAILLMClient` - 真实优化器 LLM 客户端
   - 新增 `OpenAITaskModel` - 真实任务模型客户端
   - 保留 `DummyLLMClient` 和 `DummyTaskModel` 用于测试

2. **`apo/pipeline.py`**:
   - 新增 `use_real_llm` 参数
   - 根据参数选择使用真实 API 或 Dummy 模式
   - 添加详细的调试输出

3. **`run_apo.py`**:
   - 新增 `--use_real_llm` / `--no-use_real_llm` 参数
   - 默认使用真实 LLM (use_real_llm=True)

## 调试功能

系统已添加详细的调试输出：

```python
[DEBUG] Starting APO pipeline for task: gsm8k, rounds: 5
[DEBUG] Initializing optimizer LLM and task model (use_real_llm=True)
[INFO] Initialized OpenAI client: model=openai/gpt-4o
[INFO] Initialized OpenAI task model: model=deepseek-v3
[DEBUG] OpenAILLMClient.generate() calling API...
[DEBUG] API response received, length: 353
```

这些输出帮助您追踪：
- 每次 API 调用
- 数据加载状态
- 各个优化阶段的执行
- 候选 prompt 的生成和评估

## 下一步

1. **测试 API 连接**:
   ```bash
   python test_api.py
   ```

2. **小规模测试运行**:
   ```bash
   python run_apo.py --task gsm8k --rounds 1
   ```

3. **查看结果**:
   ```bash
   cat results/gsm8k_ensemble_result.json
   ```

4. **完整优化运行** (确认成本后):
   ```bash
   python run_apo.py --task gsm8k --rounds 5
   ```

## 故障排除

### API 调用失败

如果遇到错误：
```
[ERROR] OpenAI API call failed: ...
```

检查：
1. `.env` 文件中的 API key 是否正确
2. LiteLLM 代理地址是否可访问
3. 网络连接是否正常

### 切换回 Dummy 模式

如果需要测试不调用 API：
```bash
python run_apo.py --task gsm8k --rounds 2 --no-use_real_llm
```

## 技术架构

```
用户输入
    ↓
run_apo.py (--task, --rounds, --use_real_llm)
    ↓
pipeline.py
    ↓
    ├─ OpenAILLMClient (GPT-4o) → 生成改进的 prompts
    │   └─ Bad-case Reflection
    │   └─ Evolutionary Reflection
    │   └─ Hard-case Tracking
    │
    └─ OpenAITaskModel (DeepSeek-V3) → 执行任务推理
        └─ 评估每个 prompt 在数据集上的表现
            ↓
        搜索策略选择候选
        (Bayesian + MAB)
            ↓
        Ensemble 构建
            ↓
        结果保存到 results/
```

---

**状态**: ✅ 系统已就绪，可以开始真实的 prompt 优化实验！
