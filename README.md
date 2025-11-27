# ELPO - 基于集成学习的提示词优化框架

**ELPO (Ensemble Learning based Prompt Optimization)** 是一个创新的大语言模型提示词自动优化框架。该系统结合集成学习思想，通过多种生成策略和搜索算法的协同工作，实现对提示词的高效优化。

> 📄 **论文**: Zhang, Q., Xu, B., Zhang, X., et al. (2025). "ELPO: Ensemble Learning Based Prompt Optimization for Large Languages Models". *arXiv:2511.16122* [[PDF]](docs/2511.16122v1.pdf)

## 研究背景

本项目基于 ByteDance 和香港大学的联合研究成果。与现有的单一优化算法方法不同，ELPO 采用集成学习策略，结合多个生成器和搜索方法，在多个基准数据集上显著超越了现有技术（如 APE、ProTeGi、OPRO、PromptBreeder 等）。

**核心创新**：
- 🎯 **多策略集成**：融合错误案例反思、进化式优化和难例追踪三种生成策略
- 🔍 **高效搜索**：首次将贝叶斯优化与多臂老虎机算法结合应用于提示词优化
- 🏆 **集成投票**：通过加权投票机制聚合多个高质量提示词，提升鲁棒性和泛化能力
- ⚡ **高性能执行**：并发生成和并行评估实现 7-10 倍加速

## 特性

- **多策略提示词生成**：结合错误案例反思、进化式反思和难例追踪三种生成策略
- **智能搜索算法**：使用贝叶斯优化和多臂老虎机算法高效选择候选提示词
- **集成学习**：通过加权投票集成多个优质提示词，提升最终性能
- **灵活的任务支持**：支持多种 NLP 任务（分类、推理等）
- **⚡ 高性能并行执行**：
  - 并发提示词生成：7-10倍加速
  - 并行评估推理：~10倍加速
  - 默认10个并发worker，可自定义调整

## 项目结构

```
APO/
├── apo/                           # 核心模块
│   ├── ensemble/                  # 集成方法
│   │   └── voting.py              # 加权投票
│   ├── generators/                # 提示词生成器
│   │   ├── bad_case.py            # 错误案例反思生成器
│   │   ├── evolutionary.py        # 进化式生成器
│   │   └── hard_case.py           # 难例追踪生成器
│   ├── search/                    # 搜索策略
│   │   ├── bayesian.py            # 贝叶斯优化
│   │   └── mab.py                 # 多臂老虎机
│   ├── utils/                     # 工具函数
│   │   ├── data.py                # 数据加载
│   │   ├── embedding.py           # 向量嵌入
│   │   ├── evaluation.py          # 评估指标
│   │   └── llm_api.py             # LLM API 接口 (支持并行推理)
│   ├── config.py                  # 配置管理
│   └── pipeline.py                # 主流程 (支持并行评估)
├── local_datasets/                # 数据集目录
│   ├── bbh/                       # BBH 数据集
│   ├── gsm8k/                     # GSM8K 数据集
│   ├── gsm8k_small/               # GSM8K 小规模测试集
│   └── liar/                      # LIAR 数据集
├── prompts/                       # 提示词目录
│   ├── generated/                 # 生成的提示词
│   └── initial/                   # 初始提示词
│       ├── arsarcasm.txt
│       ├── bbh.txt
│       ├── ethos.txt
│       ├── gsm8k.txt
│       ├── liar.txt
│       └── wsc.txt
├── tools/                         # 工具脚本
│   └── convert_gsm8k.py           # GSM8K 数据转换
├── checkpoints/                   # 检查点保存
├── models/                        # 模型保存
├── results/                       # 结果输出
├── CLAUDE.md                      # 项目开发指南 (Claude Code)
├── CONCURRENT_GENERATION.md       # 并发生成文档
├── PARALLEL_EVALUATION.md         # 并行评估文档
├── requirements.txt               # 依赖列表
├── run_apo.py                     # 主入口脚本
├── test_api.py                    # API 连接测试
├── test_small_run.py              # 小规模测试
├── test_parallel_eval.py          # 并行评估测试
└── demo_progress.py               # 并发生成演示
```

## 实验结果

ELPO 在 6 个基准数据集上的表现显著优于现有方法：

| 数据集 | 指标 | ELPO | GPO | EvoPrompt | PromptBreeder | OPRO | ProTeGi | APE | CoT | 提升幅度 |
|--------|------|------|-----|-----------|---------------|------|---------|-----|-----|----------|
| **LIAR** | F1 | **72.1** | 56.6 | 52.3 | 51.8 | 52.1 | 60.3 | 51.2 | 46.0 | +11.8 |
| **BBH** | F1 | **91.1** | 75.0 | 76.4 | 75.7 | 75.0 | 73.6 | 74.3 | 81.9 | +9.2 |
| **ETHOS** | F1 | **98.4** | 95.5 | 94.3 | 95.7 | 94.8 | 97.0 | 93.2 | 84.5 | +1.4 |
| **ArSarcasm** | F1 | **92.3** | 83.8 | 83.9 | 84.5 | 84.7 | 84.1 | 84.3 | 83.7 | +7.6 |
| **WSC** | Acc. | **95.9** | 84.0 | 78.8 | 80.0 | 83.3 | 80.0 | 79.3 | 81.3 | +11.9 |
| **GSM8K** | Acc. | **96.0** | 90.3 | 90.7 | 91.7 | 90.7 | 91.0 | 91.3 | 89.0 | +4.3 |

**关键发现**：
- ✅ ELPO 在所有 6 个数据集上均取得最优性能
- ✅ 相比次优方法平均提升 **7.9 个百分点**
- ✅ 在复杂推理任务（如 BBH、WSC）上提升尤为显著（>9 分）
- ✅ 通过消融实验验证了各组件的有效性

详细实验结果和分析请参阅论文。

## 安装

安装所需依赖：

```bash
pip install numpy scikit-learn scipy sentence-transformers
```

## 快速开始

### 1️⃣ 测试 API 连接

首先确保你的 LLM API 配置正确（已在 `.env` 中配置）：

```bash
python test_api.py
```

如果看到 `✅ All tests passed!`，则可以继续。

### 2️⃣ 测试并行评估性能（可选）

测试并行推理功能和性能提升：

```bash
python test_parallel_eval.py
```

预期输出：
- ✓ 串行和并行结果一致
- ✓ 并行执行显著加速（~10x）
- 性能对比数据

### 3️⃣ 小规模测试运行

建议先使用小数据集测试完整流程（~50 样本，1 轮优化）：

```bash
python test_small_run.py
```

这将：
- 创建一个 50 样本的训练集和 10 样本的测试集
- 运行 1 轮优化
- 验证真实 LLM API 工作正常
- 预估 API 调用量：~100 次

### 4️⃣ 完整优化运行

确认测试通过后，运行完整优化：

```bash
# 使用真实 LLM API (默认)
python run_apo.py --task gsm8k --rounds 5

# 快速测试 2 轮
python run_apo.py --task gsm8k --rounds 2

# 使用 dummy 模式测试（不调用 API）
python run_apo.py --task gsm8k --rounds 2 --no-use_real_llm
```

### 参数说明

- `--task`: 任务名称（必需）
  - 支持的任务：`liar`, `bbh`, `ethos`, `arsarcasm`, `wsc`, `gsm8k`
- `--rounds`: 优化轮数（默认：从 .env 读取，通常为 5）
- `--train_path`: 训练集路径（可选，默认使用 `local_datasets/<task>/train.jsonl`）
- `--test_path`: 测试集路径（可选，默认使用 `local_datasets/<task>/test.jsonl`）
- `--use_real_llm`: 使用真实 LLM API（默认：True）
- `--no-use_real_llm`: 使用 dummy 模式测试，不调用 API

### 更多示例

```bash
# 在 LIAR 数据集上运行 10 轮优化
python run_apo.py --task liar --rounds 10

# 使用自定义数据集路径
python run_apo.py --task bbh --rounds 5 \
    --train_path /path/to/train.jsonl \
    --test_path /path/to/test.jsonl

# 使用 dummy 模式快速测试流程（不消耗 API）
python run_apo.py --task gsm8k --rounds 1 --no-use_real_llm
```

## 工作原理

### ELPO 优化流程

ELPO 框架采用**共享生成-差异化搜索-集成投票**的三阶段优化策略：

#### 第一阶段：初始化
   - 加载训练/测试数据集（JSONL 格式）
   - 加载初始提示词（从 `prompts/initial/<task>.txt`）
   - 初始化**优化器 LLM**（如 GPT-4o，用于生成改进的提示词）
   - 初始化**任务模型**（如 DeepSeek-V3，用于执行实际任务）
   - 初始化 Hard-Case Tracker（全局难例追踪器）

#### 第二阶段：迭代优化（重复 n 轮）

每轮优化包含以下步骤：

1. **并行评估**
   - 在训练集上并行评估当前所有提示词（使用 `infer_batch()`）
   - 收集每个提示词的错误案例、性能分数和预测结果
   - 更新 Hard-Case Tracker 记录

2. **共享生成**（Abundant Prompt Generation）

   三种生成器**并发**生成候选提示词：

   - **Bad-Case Reflection**（算法 1）：
     - 采样失败案例 → 生成反思提示 → 迭代优化 → 添加 Few-shot 示例
     - 生成 10 个候选提示词

   - **Evolutionary Reflection**（算法 2）：
     - Direct Mutation：对 top 提示词进行语义变异（生成 5 个）
     - Zero-order Generation：综合种群特征生成全新提示词（生成 3 个）

   - **Hard-Case Tracking**（算法 3）：
     - 从全局 tracker 选择 top-k 难例 → 构建元提示 → 生成专门优化的提示词
     - 生成 1 个候选提示词

3. **高效搜索**（Efficient Prompt Search）

   使用两种搜索算法**并行**选择候选：

   - **Bayesian Search**（算法 4）：
     - 嵌入所有候选 → 用 GPR 拟合性能函数 → 计算 EI 值 → 选择 top-N

   - **Multi-Armed Bandit Search**（算法 5）：
     - K-means 聚类候选 → 每个簇视为一个臂 → 使用 UCB 策略 → 迭代采样

4. **评估与更新**
   - 并行评估搜索选中的候选提示词
   - 更新提示词种群（保留高性能提示词）
   - 更新 Hard-Case Tracker

#### 第三阶段：集成投票

1. **构建集成**（Ensemble Voting，算法 6）
   - 通过聚类和排序选择 top-M 个多样化的高性能提示词
   - 在验证集上优化加权投票权重：
     ```
     min_w { -F1_macro(w) + λ||w||² }
     s.t. Σw_j = 1, w_j ≥ w_min
     ```

2. **集成预测**
   - 对每个测试样本，每个提示词独立预测
   - 加权投票决定最终预测：
     ```
     ŷ(x) = argmax_y Σ w_j · I{f_j(x) = y}
     ```

3. **结果保存**
   - 保存集成得分、权重、成员提示词到 `results/<task>_ensemble_result.json`

### 三层 LLM 架构

- **优化器 LLM**：元级别模型，用于生成和改进提示词（如 GPT-4o）
- **任务模型**：执行提示词并完成实际任务的模型（如 DeepSeek-V3）
- **实现方式**：
  - ✅ 真实 API 模式：`OpenAILLMClient` + `OpenAITaskModel`（通过 LiteLLM 代理）
  - 🧪 测试模式：`DummyLLMClient` + `DummyTaskModel`（不调用 API，用于开发测试）

## 数据格式

数据集使用 JSONL 格式，每行一个 JSON 对象：

```json
{"input": "输入文本...", "label": "标签"}
```

### 数据集目录结构

```
local_datasets/<task>/
├── train.jsonl  # 训练集
└── test.jsonl   # 测试集
```

## 评估指标

系统根据任务类型自动选择评估指标：

- **Macro-F1**：用于 `liar`, `bbh`, `ethos`, `arsarcasm`（分类任务）
- **Accuracy**：用于 `wsc`, `gsm8k`（推理任务）

## 配置说明

### LLM API 配置

✅ **已完成真实 LLM API 集成**

系统支持通过 `.env` 文件配置真实的 LLM API：

```env
# 优化器 LLM (用于生成改进的提示词)
OPTIMIZER_LLM_API_KEY=your_api_key
OPTIMIZER_LLM_MODEL=openai/gpt-4o
OPTIMIZER_LLM_BASE_URL=https://litellm.iyunquna.com/
OPTIMIZER_LLM_TEMPERATURE=1.0
OPTIMIZER_LLM_MAX_TOKENS=2048

# 任务模型 (用于执行实际任务)
TASK_MODEL_API_KEY=your_api_key
TASK_MODEL_NAME=deepseek-v3
TASK_MODEL_BASE_URL=https://litellm.iyunquna.com/
TASK_MODEL_TEMPERATURE=0.0
TASK_MODEL_MAX_TOKENS=2048
```

支持的实现：
- `OpenAILLMClient`: 兼容 OpenAI API 格式的客户端
- `OpenAITaskModel`: 兼容 OpenAI API 格式的任务模型
- 通过 LiteLLM 代理支持多种 LLM 提供商

查看完整配置说明：`REAL_LLM_SETUP.md`

### 提示词模板

- 初始提示词存放在 `prompts/initial/<task>.txt`
- 提示词需包含 `{input}` 占位符用于动态输入注入
- 示例：
  ```
  ## Task
  判断以下陈述的真实性。

  ## Prediction
  Input: {input}
  Output:
  ```

## 核心模块说明

### 生成器 (generators/)

- **bad_case.py**：分析失败案例，生成针对性改进
- **evolutionary.py**：通过变异和交叉操作进化提示词
- **hard_case.py**：追踪难例（多次失败的样本），生成专门优化的提示词

### 搜索器 (search/)

- **bayesian.py**：基于高斯过程的贝叶斯优化，计算期望改进（EI）选择候选
- **mab.py**：将候选聚类后使用 UCB 策略平衡探索与利用

### 集成器 (ensemble/)

- **voting.py**：加权投票机制，通过随机搜索优化权重

### 工具 (utils/)

- **llm_api.py**：LLM 客户端抽象接口
  - `LLMClient.generate_batch()`: 并发提示词生成（7-10x加速）
  - `TaskModel.infer_batch()`: 并行推理接口（~10x加速）
- **embedding.py**：提示词向量化（使用 sentence-transformers）
- **evaluation.py**：任务指标计算（F1、准确率等）
- **data.py**：数据加载工具

### 性能优化

- **并发生成**：所有生成器使用 `generate_batch()` 进行并发 API 调用
- **并行评估**：`evaluate_prompt_on_dataset()` 使用 `infer_batch()` 并行推理
- **默认配置**：10个并发worker（可通过 `max_workers` 参数调整）
- **加速效果**：
  - 提示词生成：7-10倍加速
  - 评估推理：~10倍加速
  - 完整pipeline：预计~10倍加速
- **进度监控**：实时显示进度条和完成速度

测试并行功能：
```bash
# 测试并行评估
python test_parallel_eval.py

# 测试并发生成
python demo_progress.py
```

## 扩展开发

### 添加新任务

1. 在 `local_datasets/` 下创建任务目录，放置 `train.jsonl` 和 `test.jsonl`
2. 在 `prompts/initial/` 下创建初始提示词文件 `<task>.txt`
3. 在 `run_apo.py` 的 `choices` 参数中添加任务名称
4. 如需自定义指标，在 `apo/utils/evaluation.py` 的 `task_metric()` 中添加逻辑

### 自定义生成策略

继承 `apo/generators/` 中的基类，实现 `generate()` 方法，然后在 `pipeline.py` 中注册。

### 自定义搜索策略

实现包含 `select()` 方法的选择器类，输入候选嵌入和历史数据，输出选中的索引列表。

## 输出结果

优化完成后，结果保存在 `results/<task>_ensemble_result.json`：

```json
{
  "task": "liar",
  "score": 0.8523,
  "weights": [0.25, 0.30, 0.20, 0.15, 0.10],
  "members": [
    {
      "score": 0.85,
      "prompt": "提示词文本..."
    }
  ]
}
```

## 注意事项与成本预估

### ⚠️ API 调用成本

**GSM8K 完整数据集**（5 轮优化）：
- 训练集：7,473 样本
- 测试集：1,319 样本
- **预估调用量**：
  - 任务模型（DeepSeek-V3）：~40,000+ 次
  - 优化器（GPT-4o）：~50-100 次

### 💡 建议

1. **首次运行使用小规模测试**：
   ```bash
   python test_small_run.py  # 仅 50 样本，1 轮
   ```

2. **逐步增加规模**：
   ```bash
   python run_apo.py --task gsm8k --rounds 1  # 全数据，1 轮
   python run_apo.py --task gsm8k --rounds 2  # 全数据，2 轮
   ```

3. **使用 dummy 模式验证流程**：
   ```bash
   python run_apo.py --task gsm8k --rounds 5 --no-use_real_llm
   ```

4. **监控成本**：查看 LiteLLM 代理的使用统计

### 其他注意事项

- ✅ 真实 LLM API 已集成，可直接使用
- 如果未安装 `sentence-transformers`，嵌入模块会降级为随机向量（仅用于测试）
- 建议在小规模数据上先测试流程，再进行大规模优化
- 优化轮数越多，API 调用成本越高，建议根据预算合理设置

### 📁 新增文件

- `test_api.py`: API 连接测试脚本
- `test_small_run.py`: 小规模测试运行脚本
- `test_parallel_eval.py`: 并行评估测试脚本（验证10倍加速）
- `demo_progress.py`: 并发生成演示脚本
- `REAL_LLM_SETUP.md`: LLM API 配置详细说明
- `TEST_REPORT.md`: 系统测试报告
- `CONCURRENT_GENERATION.md`: 并发生成实现文档
- `PARALLEL_EVALUATION.md`: 并行评估实现文档（架构、性能、用法）

## 作者与贡献

本项目基于以下研究成果：

**研究团队**：
- Qing Zhang*, Bing Xu*, Xudong Zhang*, Yifan Shi* (ByteDance, China)
- Yang Li, Yijie Chen, Hong Dai, Xiansen Chen, Mian Zhang (ByteDance, China)
- Chen Zhang, Yik Chung Wu, Ngai Wong (The University of Hong Kong)

(*Equal contribution)

**联系方式**：
- Qing Zhang: zhangqing.thomas@bytedance.com

如果你在研究中使用了本项目，请引用我们的论文：

```bibtex
@article{zhang2025elpo,
  title={ELPO: Ensemble Learning Based Prompt Optimization for Large Languages Models},
  author={Zhang, Qing and Xu, Bing and Zhang, Xudong and Shi, Yifan and Li, Yang and Zhang, Chen and Wu, Yik Chung and Wong, Ngai and Chen, Yijie and Dai, Hong and Chen, Xiansen and Zhang, Mian},
  journal={arXiv preprint arXiv:2511.16122},
  year={2025}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 技术特点与创新

### 核心算法

ELPO 框架的技术创新主要体现在以下几个方面：

1. **多策略提示词生成**
   - **Bad-Case Reflection**：深度分析失败案例，通过反思机制生成针对性改进
   - **Evolutionary Reflection**：借鉴遗传算法，通过变异（Mutation）和零阶生成（Zero-order Generation）进化提示词
   - **Hard-Case Tracking**：全局追踪跨多个提示词反复失败的样本，生成专门优化的提示词

2. **高效搜索策略**
   - **Bayesian Search**：使用高斯过程回归（GPR）建模性能landscape，通过期望改进（Expected Improvement, EI）准则选择候选
   - **Multi-Armed Bandit (MAB) Search**：将候选聚类后视为臂，使用 UCB (Upper Confidence Bound) 策略平衡探索与利用
   - 首次将贝叶斯优化与 MAB 结合应用于提示词优化领域

3. **集成投票机制**
   - 选择 top-K 多样化的高性能提示词构建集成
   - 通过优化加权投票权重最大化验证集性能
   - 显著提升模型鲁棒性和泛化能力

4. **并发执行优化**
   - 所有生成器使用 `generate_batch()` 实现并发 API 调用（7-10x 加速）
   - 评估流程使用 `infer_batch()` 实现并行推理（~10x 加速）
   - 整体 pipeline 效率提升约 10 倍

### 理论基础

- **"No Free Lunch" 定理**：单一优化算法无法在所有任务上表现最优，因此 ELPO 采用集成策略
- **集成学习理论**：通过聚合多个"好但不稳定"的模型（提示词）提升整体性能
- **贝叶斯优化**：基于概率模型的全局优化方法，高效探索高维搜索空间
- **强化学习（MAB）**：通过试错学习最优策略，适应性分配评估资源

详细技术说明请参阅论文和 `CLAUDE.md`。
