"""Build the concise English coursework notebook; keep the learning version intact."""
import json
from pathlib import Path
from textwrap import dedent

cells = []


def markdown(text):
    """Append a Markdown cell."""
    cells.append(dict(cell_type="markdown", metadata={}, source=dedent(text).strip() + "\n"))


def code(text):
    """Append an unexecuted Python cell."""
    cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                      outputs=[], source=dedent(text).strip() + "\n"))


markdown("""
# MiniLLM: Shakespeare Text Generation
**MATH_X405 Project**

Implementation and experiments based on
[angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch/tree/63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9).
The accompanying [report](https://github.com/Bubble0128/MiniLLM/blob/main/Report_English.md)
contains the methods, experiment discussion sections, and attention proof.

Select a GPU runtime and run the cells in order. Run `baseline` first, then
`dropout` and/or `small`. Results and generated text are produced by execution.
""")
markdown("""
## 1. Setup
Auxiliary functions are maintained in separate Python modules. Google Drive stores
checkpoints and results across Colab sessions.
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

PROJECT_DIR = Path("/content/MiniLLM_submission") if IN_COLAB else Path.cwd()
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
USE_DRIVE = True #@param {type:"boolean"}
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RUNS_ROOT = Path("/content/drive/MyDrive/MiniLLM/runs/submission")
else:
    RUNS_ROOT = PROJECT_DIR / "runs/submission"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 1))
print("Device:", DEVICE, "| PyTorch:", torch.__version__)
print("Results:", RUNS_ROOT)
''')
markdown("""
## 2. Experimental configuration

| Experiment | Layers | Heads | Embedding dimension | Dropout | Purpose |
|---|---:|---:|---:|---:|---|
| baseline | 6 | 6 | 384 | 0.0 | Tutorial reference model |
| dropout | 6 | 6 | 384 | 0.1 | Test regularization |
| small | 4 | 4 | 256 | 0.0 | Test reduced model capacity |

All experiments use context length 256, batch size 64, 5,000 updates, AdamW,
100 warmup steps, cosine learning-rate decay (0.001 to 0.0001), weight decay
0.01, and gradient clipping at 1.0. Evaluation uses 20 fixed batches per split
every 100 updates. Change `RUN_TAG` when changing parameters.
""")
code('''
from dataclasses import asdict, replace
import json
from train import experiment_config

EXPERIMENT = "baseline" #@param ["baseline", "dropout", "small"]
RUN_TAG = "v1" #@param {type:"string"}
RESUME = True #@param {type:"boolean"}

model_config, train_config = experiment_config(EXPERIMENT)
# Apply any shared training changes consistently across compared experiments.
# train_config = replace(train_config, batch_size=32)

PROMPT = "To be, or not to be, that is the question:\\n"  # Hamlet, Act 3, Scene 1
MAX_NEW_TOKENS = 800
TEMPERATURES = (0.6, 0.8, 1.0)
TOP_K = 40
SEEDS = (42, 123, 2026)
SAMPLE_TAG = "hamlet_v1"  # Change this when modifying the prompt or sampling settings.

for label in (RUN_TAG, SAMPLE_TAG):
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label):
        raise ValueError("Tags must contain only letters, digits, underscores or hyphens")
RUN_DIR = RUNS_ROOT / f"{EXPERIMENT}_{RUN_TAG}"
print(json.dumps({"model": asdict(model_config), "training": asdict(train_config)}, indent=2))
''')
markdown("""
## 3. Data and model
The pinned Shakespeare corpus is encoded at character level and split contiguously
into 90% training and 10% validation data. Targets are inputs shifted by one character.
The model combines token and learned positional embeddings, pre-normalized causal
attention blocks, residual connections, a 4x-width GELU MLP, and a tied output head.
""")
code('''
from data import CharacterData, download_shakespeare
from model import GPT

DATA_PATH = download_shakespeare(PROJECT_DIR / "data/shakespeare.txt")
dataset = CharacterData(DATA_PATH)
model_config = replace(model_config, vocab_size=len(dataset.chars))
dataset.encode(PROMPT)
print("Vocabulary:", len(dataset.chars))
print("Training characters:", len(dataset.splits["train"]))
print("Validation characters:", len(dataset.splits["val"]))
print("Data SHA256:", dataset.sha256)
parameter_model = GPT(model_config)
print("Parameters:", sum(p.numel() for p in parameter_model.parameters()))
del parameter_model
''')
markdown("""
## 4. Training
Training and validation losses are measured in evaluation mode on fixed sampled
windows. The lowest observed validation loss selects `best.pt`; `last.pt` stores
the optimizer and random states for resuming training.
""")
code('''
from train import train

summary = train(DATA_PATH, RUN_DIR, model_config, train_config, device=DEVICE,
                resume=RESUME and (RUN_DIR / "last.pt").exists())
print("Best validation loss:", summary["best_val_loss"])
print("Best checkpoint step:", summary["best_step"])
''')
markdown("""
## 5. Loss curves
Loss is cross-entropy in nats per character. Validation data are used for model
selection, rather than as an independent test set.
""")
code('''
import matplotlib.pyplot as plt
from analysis import plot_history

figure = plot_history(RUN_DIR)
plt.show()
plt.close(figure)
''')
markdown("""
## 6. Soliloquy continuation
Each sample continues the first line of Hamlet's soliloquy using the best checkpoint.
The prompt, generation length, seed list and top-k are fixed across models.
Temperature is varied independently of training; all samples are retained.
""")
code('''
from generate import save_samples

samples = save_samples(
    RUN_DIR / "best.pt", RUN_DIR / f"samples_{SAMPLE_TAG}", device=DEVICE,
    prompt=PROMPT, temperatures=TEMPERATURES, top_k=TOP_K, seeds=SEEDS,
    max_new_tokens=MAX_NEW_TOKENS,
)
for sample in samples:
    print(f"\\nTemperature: {sample['temperature']} | Seed: {sample['seed']}")
    print(sample["text"])
''')
markdown("""
## 7. Experiment comparison
The table includes completed experiments with the selected `RUN_TAG`. Compare loss,
training cost and generated text. The report should connect observed spelling,
grammar, coherence and repetition to model and sampling choices.
""")
code('''
from analysis import compare_runs
from IPython.display import Markdown, display

completed_runs = []
for name in ("baseline", "dropout", "small"):
    path = RUNS_ROOT / f"{name}_{RUN_TAG}"
    summary_file = path / "summary.json"
    if summary_file.exists() and json.loads(summary_file.read_text())["complete"]:
        completed_runs.append(path)

if completed_runs:
    rows = compare_runs(completed_runs, RUNS_ROOT / f"comparison_{RUN_TAG}")
    table = "| Experiment | Parameters | Best step | Best val loss | Final val loss | Minutes |\\n"
    table += "|---|---:|---:|---:|---:|---:|\\n"
    for row in rows:
        table += (f"| {row['run']} | {row['parameters']:,} | {row['best_step']} | "
                  f"{row['best_val_loss']:.4f} | {row['last_val_loss']:.4f} | "
                  f"{row['elapsed_minutes']:.1f} |\\n")
    display(Markdown(table))
else:
    print("No completed experiments available.")
''')
markdown("""
## 8. Submission files
The archive contains Python modules, the English source notebook, the English report
draft, and results from the completed experiments above. Complete the report using
these results, and separately download this executed notebook from Colab to retain
its displayed outputs. Model checkpoints remain in the results directory.
""")
code('''
from zipfile import ZipFile, ZIP_DEFLATED

DOWNLOAD_ARCHIVE = False #@param {type:"boolean"}
archive_path = RUNS_ROOT / f"MiniLLM_{RUN_TAG}.zip"
with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
    for name in ("model.py", "data.py", "train.py", "generate.py", "analysis.py",
                 "requirements.txt", "MiniLLM_Submission.ipynb"):
        archive.write(PROJECT_DIR / name, f"code/{name}")
    archive.write(PROJECT_DIR / "Report_English.md", "report/Report_English.md")
    for run_dir in completed_runs:
        for path in run_dir.rglob("*"):
            if path.is_file() and path.suffix not in (".pt", ".tmp"):
                archive.write(path, f"results/{run_dir.name}/{path.relative_to(run_dir)}")
    comparison = RUNS_ROOT / f"comparison_{RUN_TAG}" / "comparison.csv"
    if completed_runs and comparison.exists():
        archive.write(comparison, "results/comparison.csv")
print("Archive:", archive_path)
if IN_COLAB and DOWNLOAD_ARCHIVE:
    from google.colab import files
    files.download(str(archive_path))
''')

for index, cell in enumerate(cells):
    cell["id"] = f"submission-{index:02d}"
    cell["metadata"]["id"] = cell["id"]

notebook = dict(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"name": "MiniLLM_Submission.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}, nbformat=4, nbformat_minor=5)

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "MiniLLM_Submission.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(path)
