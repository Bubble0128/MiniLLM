"""Build the Colab notebook from readable cells using only the standard library."""
import json
from pathlib import Path
import textwrap

cells = []


def markdown(source):
    """Append an explanatory Markdown cell."""
    cells.append(dict(cell_type="markdown", metadata={},
                      source=textwrap.dedent(source).strip() + "\n"))


def code(source):
    """Append an unexecuted Python cell with a stable cell identifier."""
    cells.append(dict(cell_type="code", execution_count=None, outputs=[], metadata={},
                      source=textwrap.dedent(source).strip() + "\n"))


markdown("""
# MiniLLM：从零训练 Shakespeare GPT

**MATH_X405 project — Colab 实验入口**

按顺序运行本 Notebook。首次使用：**运行时 → 更改运行时类型 → T4 GPU**，
然后选择 **文件 → 在云端硬盘中保存副本**。辅助代码在仓库的 `.py` 文件中。

本程序按 [angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch)
教程第 1–5 部分实现，资料版本固定为
`63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9`。
训练数据也固定为该版本的 `data/shakespeare.txt`，并校验内容。
无预训练权重；词表为字符级；预测下一个字符。

**首次默认运行 `smoke`（20 步）只检查流程，不是最终实验。**
跑通后把第 3 步的 `EXPERIMENT` 改为 `baseline` 进行完整基线训练。
再单独运行 `dropout` 或 `small`，对比 loss 与生成文本。
结果取决于实际训练；此 Notebook 不预填任何性能结论。

作业交付：代码、生成文本、报告（结构分析、改进讨论、数学证明）。
这里提供代码与实验记录工具，报告仍需依据真实结果完成。
""")
markdown("""
## 1. 下载程序并检查环境

Colab 首次运行会从你的 GitHub 仓库下载程序。已经存在的目录会保留，避免覆盖修改；
需要获取仓库新版本时，请使用新的 Colab 运行时。每次训练会记录实际代码哈希。
只安装所需库，不强制升级 Colab 自带的 PyTorch。
""")
code('''
from pathlib import Path
import os
import subprocess
import sys

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

PROJECT_DIR = Path("/content/MiniLLM") if IN_COLAB else Path.cwd()
if not (PROJECT_DIR / "model.py").exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Bubble0128/MiniLLM.git", str(PROJECT_DIR)], check=True)
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

import torch
from train import get_device
DEVICE = get_device()
print("Project:", PROJECT_DIR)
print("PyTorch:", torch.__version__, "| device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name())
else:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print("CPU/MPS 可检查流程；Colab 完整实验请切换 GPU。")
''')
markdown("""
## 2. 选择保存位置

推荐挂载 Google Drive，保存断点、参数、loss 和生成文本。
`USE_DRIVE=False` 时只保存在当前运行时，Colab 重置后会丢失。
挂载 Drive 可能需要你在 Colab 中完成一次 Google 授权。
""")
code('''
USE_DRIVE = True #@param {type:"boolean"}
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RUNS_ROOT = Path("/content/drive/MyDrive/MiniLLM/runs")
else:
    RUNS_ROOT = PROJECT_DIR / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
print("Results will be saved to:", RUNS_ROOT)
''')
markdown("""
## 3. 集中配置实验

| EXPERIMENT | 模型 | 训练步数 | 用途 |
|---|---|---:|---|
| `smoke` | 2 层 / 2 头 / 64 维，context 32 | 20 | 检查代码，不能用作正式基线 |
| `baseline` | 6 层 / 6 头 / 384 维，context 256 | 5000 | 教程默认架构与训练预算，dropout=0 |
| `dropout` | 与 baseline 相同，dropout=0.1 | 5000 | 正则化实验，其余配置保持一致 |
| `small` | 4 层 / 4 头 / 256 维，context 256 | 5000 | 教程建议的容量实验，dropout=0 |

完整实验使用 batch size 64、AdamW、lr=1e-3、100 步 warmup、cosine decay、梯度裁剪。
默认保存最佳验证 checkpoint，但不自动提前停止；这样各组的训练预算一致。
temperature/top-k 属于生成设置，不改变 checkpoint 的标准验证 loss。

**修改参数后使用新的 `RUN_TAG`**。相同目录默认从 `last.pt` 恢复，若参数或数据改变则报错。
如显存不足，可把 batch size 改为 32；正式比较的各组应一致修改，并在报告说明。
完整训练时间由分配到的 GPU 决定；先观察前 100 步耗时。
""")
code('''
from dataclasses import asdict, replace
import json
from train import experiment_config
from generate import DEFAULT_PROMPT

EXPERIMENT = "smoke" #@param ["smoke", "baseline", "dropout", "small"]
RUN_TAG = "v1" #@param {type:"string"}
RESUME_IF_AVAILABLE = True #@param {type:"boolean"}
PROMPT = DEFAULT_PROMPT  # Hamlet 独白首行，可改成另一段独白的首行

model_config, train_config = experiment_config(EXPERIMENT)
# 自定义示例（取消注释后请更换 RUN_TAG）：
# train_config = replace(train_config, batch_size=32)
# model_config = replace(model_config, block_size=128)

if not RUN_TAG or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in RUN_TAG):
    raise ValueError("RUN_TAG 只能包含英文字母、数字、下划线、短横线")
RUN_DIR = RUNS_ROOT / f"{EXPERIMENT}_{RUN_TAG}"
MAX_NEW_TOKENS = 120 if EXPERIMENT == "smoke" else 800
SAMPLE_SEEDS = (42,) if EXPERIMENT == "smoke" else (42, 123, 2026)
TEMPERATURES = (0.8,) if EXPERIMENT == "smoke" else (0.6, 0.8, 1.0)
TOP_K = 40
print(json.dumps({"model": asdict(model_config), "training": asdict(train_config)}, indent=2))
print("Run directory:", RUN_DIR)
print("Prompt:", repr(PROMPT))
''')
markdown("""
## 4. 数据、词表与前向检查

文本前 90% 用于训练，后 10% 用于验证；采样窗口不会跨越分割边界。
词表沿用教程，取整份文本的字符集合，但只用训练片段更新模型。
验证集用于调参和 checkpoint 选择，不能称作独立测试集。
""")
code('''
from data import CharacterData, download_shakespeare
from model import GPT

DATA_PATH = download_shakespeare(PROJECT_DIR / "data/shakespeare.txt")
dataset = CharacterData(DATA_PATH)
model_config = replace(model_config, vocab_size=len(dataset.chars))
assert dataset.decode(dataset.encode(PROMPT)) == PROMPT
print("Vocabulary:", len(dataset.chars))
print("Train chars:", len(dataset.splits["train"]), "| validation chars:", len(dataset.splits["val"]))
print("Data SHA256:", dataset.sha256)

check_model = GPT(model_config).to(DEVICE)
x, y = dataset.batch("train", min(16, model_config.block_size), 2, DEVICE,
                     torch.Generator().manual_seed(999))
with torch.no_grad():
    logits, loss = check_model(x, y)
print("Input:", tuple(x.shape), "| logits:", tuple(logits.shape), "| loss:", loss.item())
print("Parameters:", sum(p.numel() for p in check_model.parameters()))
del check_model, x, y, logits, loss
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
''')
markdown("""
## 5. 训练并保存最佳模型

结构：token embedding + learned position embedding → pre-LN Transformer blocks
（causal multi-head attention + residual；4× GELU MLP + residual）→ LayerNorm → tied output head。

记录的 train/val loss 都在 eval mode 下计算，每次使用固定评估窗口，避免 dropout
及不同采样窗口干扰曲线；训练采样使用独立 RNG。
`step` 是已经完成的参数更新次数。`best.pt` 依据验证 loss 选择，`last.pt` 用于断点恢复。
断开后重新运行前面的单元格即可恢复到最近一次保存点（通常每 100 步）。
同后端、同软件下可尽量复现；跨 GPU/软件版本不承诺逐位一致。
""")
code('''
from train import train

summary = train(
    DATA_PATH, RUN_DIR, model_config, train_config, device=DEVICE,
    resume=RESUME_IF_AVAILABLE and (RUN_DIR / "last.pt").exists(),
)
print(json.dumps(summary, indent=2))
if EXPERIMENT == "smoke":
    print("流程检查完成。下一步将 EXPERIMENT 改为 baseline，从第 3 步重新运行。")
''')
markdown("""
## 6. Loss 曲线与最佳 checkpoint

观察验证 loss 是否下降，以及 train/val 的差距是否扩大。不要只看训练 loss。
最佳 checkpoint 是在本次评估间隔中观测到的最优点。
""")
code('''
import matplotlib.pyplot as plt
from analysis import plot_history

figure = plot_history(RUN_DIR)
plt.show()
plt.close(figure)
print("Best checkpoint:", RUN_DIR / "best.pt")
''')
markdown("""
## 7. 用独白首行续写并比较 temperature

固定首行、长度与种子列表，保留所有样本。完整实验默认生成 3 个 temperature × 3 个种子。
本轮固定 top-k=40；要研究 top-k，请保留相同模型/temperature，并另存样本目录。
模型学习 Shakespeare 风格，不保证逐字还原 Hamlet 原文。
关注拼写、句法、连贯性、重复和风格，并在报告引用实际片段。
""")
code('''
from generate import save_samples

SAMPLES_DIR = RUN_DIR / f"samples_topk_{TOP_K}"
samples = save_samples(
    RUN_DIR / "best.pt", SAMPLES_DIR, device=DEVICE, prompt=PROMPT,
    temperatures=TEMPERATURES, top_k=TOP_K, seeds=SAMPLE_SEEDS,
    max_new_tokens=MAX_NEW_TOKENS,
)
for sample in samples:
    print(f"\\nTemperature={sample['temperature']} | seed={sample['seed']} | checkpoint step={sample['checkpoint_step']}")
    print(sample["text"])
print("All generated texts and metadata:", SAMPLES_DIR)
''')
markdown("""
## 8. 跨实验比较表

运行 baseline、dropout、small 后，比较表会汇总这些目录。默认排除 smoke。
优先对比相同 RUN_TAG 的实验，避免混入不同预算。请检查 CSV 中的配置列。
small 同时改变层数、头数和宽度，是整体容量对比，不能把结果单独归因于某一个参数。
dropout 是否有帮助，需要实际结果支持；更小模型也不保证更好的 loss。
""")
code('''
from analysis import compare_runs

experiment_dirs = [RUNS_ROOT / f"{name}_{RUN_TAG}" for name in ("baseline", "dropout", "small")]
experiment_dirs = [path for path in experiment_dirs if (path / "summary.json").exists()]
if experiment_dirs:
    rows = compare_runs(experiment_dirs, RUNS_ROOT / f"comparison_{RUN_TAG}")
    for row in rows:
        print(row["run"], "best val:", round(row["best_val_loss"], 4),
              "best step:", row["best_step"], "complete:", row["complete"])
    print("Saved:", RUNS_ROOT / f"comparison_{RUN_TAG}" / "comparison.csv")
else:
    print("目前仅有流程检查；完成 baseline 后这里会生成正式实验比较表。")
''')
markdown(r"""
## 9. 理论部分：数值演示与证明边界

作业要求书面证明，而不是只运行数值实验。下面只是帮助检查理解。
令 $A(X)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$，$P$ 为置换矩阵。
对于**没有位置编码、没有固定 causal mask**的 self-attention，证明目标是
$A(PX)=PA(X)$，可使用按行 softmax 的恒等式
$\mathrm{softmax}(PSP^\top)=P\mathrm{softmax}(S)P^\top$。

对固定的绝对位置编码 $E$，比较 $A(PX+E)$ 与 $PA(X+E)$：
通常 $PX+E\ne P(X+E)$，因此等变性不再普遍成立。
但仅指出输入不同，不足以证明输出必然不同；书面论证应给反例或证明一般性失败，
不要宣称每个位置编码、每个置换、每组参数都必然破坏等变性。
若把内容和位置编码一起置换，基础 attention 的等变性仍成立。

本项目的 GPT 使用**固定 causal mask**，它本身就破坏任意置换下的等变性。
请按课程符号区分理论问题里的 unmasked attention 与实际 GPT。
""")
code('''
from analysis import attention_equivariance_demo

theory_checks = attention_equivariance_demo()
for name, error in theory_checks.items():
    print(f"{name}: {error:.3e}")
print("接近零表示此例吻合；数值演示不能替代一般性数学证明。")
''')
markdown("""
## 10. 导出代码和本次实验结果

压缩包包含 `.py`、原始 Notebook、参数、曲线、生成文本、比较表（若已存在）。
默认不打包体积较大的模型权重；权重仍保留在 Drive。
**另用 Colab 的“文件 → 下载 → .ipynb”保存带运行输出的 Notebook**；
压缩包里的 Notebook 是 GitHub 的未执行版本。报告需单独完成并加入最终提交材料。
""")
code('''
from zipfile import ZipFile, ZIP_DEFLATED

INCLUDE_CHECKPOINTS = False #@param {type:"boolean"}
DOWNLOAD_ZIP = False #@param {type:"boolean"}
EXPORT_DIR = RUNS_ROOT.parent / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
archive_path = EXPORT_DIR / f"{RUN_DIR.name}_results.zip"
with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
    for name in ("model.py", "data.py", "train.py", "generate.py", "analysis.py",
                 "requirements.txt", "README.md", "MiniLLM_Colab.ipynb"):
        path = PROJECT_DIR / name
        if path.exists():
            archive.write(path, f"code/{name}")
    for path in RUN_DIR.rglob("*"):
        if path.is_file() and (INCLUDE_CHECKPOINTS or path.suffix != ".pt") and path.suffix != ".tmp":
            archive.write(path, f"results/{RUN_DIR.name}/{path.relative_to(RUN_DIR)}")
    comparison = RUNS_ROOT / f"comparison_{RUN_TAG}" / "comparison.csv"
    if comparison.exists():
        archive.write(comparison, "results/comparison.csv")
print("Saved:", archive_path)
if IN_COLAB and DOWNLOAD_ZIP:
    from google.colab import files
    files.download(str(archive_path))
''')
markdown("""
## 报告记录清单与来源

- 描述字符级 tokenization、上下文长度、causal attention、位置 embedding、层数/头数/宽度。
- 标明基线、每项改动的动机、参数、训练预算、随机种子、GPU 与代码版本。
- 引用 train/val 曲线和真实生成片段；讨论语言自然度、重复、连贯性及局限。
- temperature/top-k 已在原教程中介绍；不能声称为新增模型结构。
- dropout 是实验性修改；best checkpoint、日志、恢复和固定评估是工程改进。
- 用课程符号完成数学证明。此 Notebook 的演示不构成完整证明。
- 本程序使用 AI 辅助实现；保留来源说明，并按课程要求披露使用情况。

来源：[README](https://github.com/angelos-p/llm-from-scratch/blob/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9/README.md)，
[Part 1–5](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9/docs)。
原仓库 Part 6 的诗歌竞赛属于另一项活动，不代替本课程截图里的作业要求。
""")

for index, cell in enumerate(cells):
    cell["id"] = f"minillm-{index:02d}"
    cell["metadata"]["id"] = cell["id"]

notebook = dict(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"name": "MiniLLM_Colab.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}, nbformat=4, nbformat_minor=5)

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "MiniLLM_Colab.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)
