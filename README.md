# MiniLLM
For MATH_X405 final report and presentation

**English submission version:** [Open in Colab](https://colab.research.google.com/github/Bubble0128/MiniLLM/blob/main/MiniLLM_Submission.ipynb) · [Notebook](MiniLLM_Submission.ipynb) · [Report](Report_English.md)

英文版保留提交所需的代码、实验展示和导出功能。
下方为原中文版学习入口。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bubble0128/MiniLLM/blob/main/MiniLLM_Colab.ipynb)

基于 [angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch) 教程实现的
字符级 Shakespeare GPT。用于从零训练、独白续写和可追溯的课程实验。

## 在 Colab 开始

1. 点击 **Open in Colab**，选择“文件 → 在云端硬盘中保存副本”。
2. “运行时 → 更改运行时类型 → T4 GPU”。
3. 选择 tiny、small、medium 之一。默认读取已有结果，上传相应实验目录即可展示。
4. 需要训练时开启 `RUN_TRAINING=True`，相同配置可从 last.pt 恢复。
5. 三组采用 dropout=0.1，每 100 步评估一次，连续五次未改善时早停，最多 10000 步。

Notebook 会自动下载 `.py` 程序和固定版本的数据，无需手动上传脚本。
默认把结果保存至 `MyDrive/MiniLLM/runs/`；挂载 Drive 需要在 Colab 中授权。
同名实验再次运行会恢复最近的 checkpoint；修改配置时使用新的 `RUN_TAG`。
若显存不足，各组一致降低 batch size，并重新命名实验，保留公平比较。

完整实验默认上限 10000 步，实际耗时取决于 Colab GPU；先观察前 100 步耗时。

## 文件与输出

| 文件 | 作用 |
|---|---|
| `MiniLLM_Colab.ipynb` | 中文说明、参数入口、训练、图表、生成、导出 |
| `model.py` | GPTConfig、causal attention、MLP、Transformer blocks、GPT |
| `data.py` | 固定来源下载与校验、字符编码、90/10 数据划分 |
| `train.py` | AdamW、warmup/cosine、梯度裁剪、固定窗口评估、保存和恢复 |
| `generate.py` | 独白首行续写、temperature、top-k、多种子样本 |
| `analysis.py` | loss 曲线、实验 CSV、attention 等变性的数值演示 |
| `tests/test_pipeline.py` | 因果性、学习能力、断点恢复、采样等行为检查 |
| `build_notebook.py` | 从可读的单元格源文件重新生成 Notebook |

每次训练保存：

```text
runs/baseline_v1/
├── config.json          # 参数、数据指纹、软件/GPU、代码版本与哈希
├── history.json         # 按完成步数对齐的 train/val loss
├── metrics.csv
├── summary.json         # 最佳/最后 loss、参数量、时间与训练 token 数
├── best.pt              # 验证 loss 最低的已观测 checkpoint
├── last.pt              # 模型、优化器、RNG：用于继续训练
├── loss_curve.png
└── samples_topk_40/     # 每份原始生成文本与 samples.json
```

权重、数据与实验输出由 `.gitignore` 排除，避免提交大文件。
Notebook 最后一步导出代码与本次结果的 ZIP。
需要包含运行输出时，请另外从 Colab 下载 `.ipynb`；ZIP 内的 Notebook 是源版本。

## 基线和实验的边界

资料版本：[`63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9`](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9)。
当前上游主要包含教程与数据，并未实际提供文档提到的 `colab.ipynb`。
本项目将教程第 1–5 部分整理为可运行模块和 Colab 入口。

| 设置 | tiny | small | medium |
|---|---:|---:|---:|
| 层数 / 头数 / embedding 维度 | 2 / 2 / 128 | 4 / 4 / 256 | 6 / 6 / 384 |
| 上下文长度 | 256 | 256 | 256 |
| dropout | 0.1 | 0.1 | 0.1 |
| batch size / 最大步数 | 64 / 10000 | 64 / 10000 | 64 / 10000 |
| 最大 / 最小学习率 | 1e-3 / 1e-4 | 同左 | 同左 |
| warmup / weight decay / gradient clip | 100 / 0.01 / 1.0 | 同左 | 同左 |
| early stopping patience | 5 | 5 | 5 |

每 100 步验证一次，严格降低 CE 才重置计数。早停后使用 best.pt 生成，last.pt 保存停止状态。
当前规模实验的余弦学习率调度按 10000 步规划，每组从随机初始化训练。
此前 5000 步规模实验保存在带 scale_d01_p5_5080_v1 后缀的目录。

历史 baseline 使用 6/6/384、dropout=0、5000 步且无早停，原始结果位于 runs/baseline_5080_v1。
三组规模实验同时改变深度、头数和宽度，实际步数和 token 数由停止时刻决定。

- **教程已有**：字符级词表、learned positions、pre-LN、GELU、weight tying、causal SDPA、
  AdamW、warmup/cosine、梯度裁剪、temperature 和 top-k；不可当作本项目新增设计。
- **模型实验**：可配置 embedding/attention/residual dropout；使用教程建议的小模型作容量对比。
  small 同时改变层数、头数、宽度，不能单独归因于某一个变量。
- **工程补充**：固定评估窗口和独立采样 RNG、对齐指标、最佳模型选择、完整恢复状态、
  配置与数据一致性检查、未知字符报错、采样参数校验、结果和代码指纹导出。
- baseline 保留教程的模型/优化配置，但上述评估与保存修正也应用于基线，
  所以它不是对教程训练循环逐行复刻。三组使用同一套评估基础设施。

评估：训练前 90%、验证后 10%，窗口不跨边界；词表沿用教程由整个数据的字符集合构成。
每次 train/val 都在 eval mode 下用固定种子评估 20 个 batch。报告中的验证 loss 是采样估计，
不是完整语料分数；验证集用于选模型，不是独立测试集。报告和比较表统一使用字符级交叉熵（CE），单位 nats/character。
修改上下文/批量等配置会改变评估窗口或 token 预算，比较时须注明。

默认完整生成比较使用 Hamlet 首行 `To be, or not to be, that is the question:`，
追加换行后续写 800 个字符，temperature 为 0.6/0.8/1.0，top-k=40，种子为 42/123/2026。
保留每份样本而非只展示最好的一份。生成不是逐字还原原作，也不是训练质量保证。
修改 prompt/采样网格时请更换样本目录名，避免覆盖原始对比材料。

## 本地检查与运行

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --experiment tiny
python train.py --experiment small
python train.py --experiment medium
python run_size_experiments.py --run-tag scale_d01_p5_10k_5080_v1 --max-steps 10000
```

源代码变动后，分别用 `python build_notebook.py` 和 `python build_submission_notebook.py` 重新生成两个 Notebook。
`last.pt` 保存 optimizer 和 RNG；同 CPU 后端的恢复行为有自动化检查。
跨设备/软件版本恢复允许继续训练，但不承诺逐位相同的结果。

已在 CPU 环境验证：完整 Notebook 的 10 个代码单元格执行、真实 Shakespeare 数据的
20 步短训练、loss 绘图、模型加载与续写、ZIP 导出；自动化检查覆盖因果性、有限梯度、
合成序列学习、dropout 训练的精确恢复、采样可复现性及等变性演示。
正式训练在本地 NVIDIA RTX 5080 上完成；这些检查不代表最终文本质量或改进效果。

## 理论任务与报告

Notebook 含数值演示和证明提示，**数值检查不是一般性数学证明**。
无位置编码且无固定 causal mask 的 self-attention 满足置换等变性。
固定绝对位置编码一般破坏这一性质；如果同时置换位置编码，则基础 attention 仍然等变。
实际 GPT 的固定 causal mask 也破坏任意置换等变性，应与课堂理论假设区分。

完整证明见 [Report_English.md](Report_English.md) 第 4 节。
本仓库代码经 AI 辅助实现，教程来源见上文。
