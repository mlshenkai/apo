# APO - 自动提示词优化框架

APO (Automatic Prompt Optimization) 是一个基于迭代反思和搜索的 LLM 提示词自动优化框架。该系统通过评估提示词在任务上的表现，基于失败案例分析生成改进版本，并使用贝叶斯优化和多臂老虎机搜索策略高效探索提示词空间。

## 特性

- **多策略提示词生成**：结合错误案例反思、进化式反思和难例追踪三种生成策略
- **智能搜索算法**：使用贝叶斯优化和多臂老虎机算法高效选择候选提示词
- **集成学习**：通过加权投票集成多个优质提示词，提升最终性能
- **灵活的任务支持**：支持多种 NLP 任务（分类、推理等）

## 项目结构

```
APO/
├── datasets/              # 数据集目录
│   ├── liar/             # LIAR 数据集
│   ├── bbh/              # BBH 数据集
│   └── ...
├── prompts/              # 提示词目录
│   ├── initial/          # 初始提示词
│   └── generated/        # 生成的提示词
├── apo/                  # 核心模块
│   ├── generators/       # 提示词生成器
│   │   ├── bad_case.py   # 错误案例反思生成器
│   │   ├── evolutionary.py  # 进化式生成器
│   │   └── hard_case.py  # 难例追踪生成器
│   ├── search/           # 搜索策略
│   │   ├── bayesian.py   # 贝叶斯优化
│   │   └── mab.py        # 多臂老虎机
│   ├── ensemble/         # 集成方法
│   │   └── voting.py     # 加权投票
│   ├── utils/            # 工具函数
│   │   ├── llm_api.py    # LLM API 接口
│   │   ├── embedding.py  # 向量嵌入
│   │   ├── evaluation.py # 评估指标
│   │   └── data.py       # 数据加载
│   └── pipeline.py       # 主流程
├── checkpoints/          # 检查点保存
├── models/               # 模型保存
├── results/              # 结果输出
└── run_apo.py           # 主入口脚本
```

## 安装

安装所需依赖：

```bash
pip install numpy scikit-learn scipy sentence-transformers
```

## 快速开始

### 基本用法

```bash
python run_apo.py --task liar --rounds 5
```

### 参数说明

- `--task`: 任务名称（必需）
  - 支持的任务：`liar`, `bbh`, `ethos`, `arsarcasm`, `wsc`, `gsm8k`
- `--rounds`: 优化轮数（默认：5）
- `--train_path`: 训练集路径（可选，默认使用 `datasets/<task>/train.jsonl`）
- `--test_path`: 测试集路径（可选，默认使用 `datasets/<task>/test.jsonl`）

### 示例

```bash
# 在 LIAR 数据集上运行 10 轮优化
python run_apo.py --task liar --rounds 10

# 使用自定义数据集路径
python run_apo.py --task bbh --rounds 5 \
    --train_path /path/to/train.jsonl \
    --test_path /path/to/test.jsonl
```

## 工作原理

### 优化流程

1. **初始化**
   - 加载数据集和初始提示词（从 `prompts/initial/<task>.txt`）
   - 初始化优化器 LLM（如 GPT-4o）和任务 LLM（如 Doubao-pro）

2. **迭代优化**（重复 n 轮）
   - **评估**：在训练集上评估当前提示词，收集错误案例
   - **生成**：使用三种策略生成候选提示词
     - 错误案例反思：分析失败样本，生成改进版本
     - 进化式反思：对现有提示词进行变异和交叉
     - 难例追踪：针对持续失败的样本生成特殊提示词
   - **搜索**：使用贝叶斯优化和 MAB 选择最有潜力的候选
   - **更新**：评估选中的候选，更新提示词种群

3. **集成构建**
   - 选择 top-K 表现最好的提示词
   - 优化加权投票的权重（在训练集上最大化指标）
   - 在测试集上评估集成效果

4. **结果保存**
   - 保存最终得分、权重和集成成员到 `results/` 目录

### 三层 LLM 架构

- **优化器 LLM**：元级别模型，用于生成和改进提示词
- **任务模型**：执行提示词并完成实际任务的模型
- **当前实现**：使用 `DummyLLMClient` 和 `DummyTaskModel` 作为开发占位符

## 数据格式

数据集使用 JSONL 格式，每行一个 JSON 对象：

```json
{"input": "输入文本...", "label": "标签"}
```

### 数据集目录结构

```
datasets/<task>/
├── train.jsonl  # 训练集
└── test.jsonl   # 测试集
```

## 评估指标

系统根据任务类型自动选择评估指标：

- **Macro-F1**：用于 `liar`, `bbh`, `ethos`, `arsarcasm`（分类任务）
- **Accuracy**：用于 `wsc`, `gsm8k`（推理任务）

## 配置说明

### 替换为真实 LLM API

在生产环境中，需要：

1. 在 `apo/utils/llm_api.py` 中实现真实的 API 调用
2. 继承 `LLMClient` 和 `TaskModel` 抽象类
3. 在 `apo/pipeline.py` 中替换 `DummyLLMClient` 和 `DummyTaskModel`

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
- **embedding.py**：提示词向量化（使用 sentence-transformers）
- **evaluation.py**：任务指标计算（F1、准确率等）
- **data.py**：数据加载工具

## 扩展开发

### 添加新任务

1. 在 `datasets/` 下创建任务目录，放置 `train.jsonl` 和 `test.jsonl`
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

## 注意事项

- 当前实现使用虚拟 LLM 客户端，需替换为真实 API 才能正常工作
- 如果未安装 `sentence-transformers`，嵌入模块会降级为随机向量（仅用于测试）
- 建议在小规模数据上先测试流程，再进行大规模优化
- 优化轮数越多，API 调用成本越高，建议根据预算合理设置

## 许可证

（待添加）

## 技术特点

本项目结合了反思学习、进化算法和贝叶斯优化等多种技术，实现了高效的自动提示词优化。
