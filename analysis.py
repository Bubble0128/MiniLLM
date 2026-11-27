"""Plots, experiment tables, and a numerical illustration for the theory task."""
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_history(run_dir):
    """Plot aligned train/validation evaluation losses at completed steps."""
    run_dir = Path(run_dir)
    rows = json.loads((run_dir / "history.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([r["step"] for r in rows], [r["train_loss"] for r in rows], label="Train (eval mode)")
    ax.plot([r["step"] for r in rows], [r["val_loss"] for r in rows], label="Validation")
    best = min(rows, key=lambda r: r["val_loss"])
    ax.axvline(best["step"], color="grey", linestyle="--", alpha=0.5, label="Best validation")
    summary = json.loads((run_dir / "summary.json").read_text())
    unit = summary.get("loss_unit", "nats/character")
    ax.set(xlabel="Completed optimizer steps", ylabel=f"Cross-entropy ({unit})",
           title=run_dir.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=160)
    return fig


def compare_runs(run_dirs, output_dir):
    """Export factual metrics; do not infer text quality from loss alone.

    Include settings needed to spot unequal compute or evaluation budgets.
    """
    rows = []
    for run_dir in map(Path, run_dirs):
        s = json.loads((run_dir / "summary.json").read_text())
        m, t = s["model_config"], s["train_config"]
        rows.append(dict(run=run_dir.name, complete=s["complete"],
                         tokenizer_type=s.get("tokenizer_type", "character"),
                         loss_unit=s.get("loss_unit", "nats/character"),
                         actual_steps=s["step"], patience=t.get("patience", 0),
                         stop_reason=s.get("stop_reason", "max_steps"),
                         parameters=s["parameters"], n_layer=m["n_layer"],
                         n_head=m["n_head"], n_embd=m["n_embd"],
                         dropout=m["dropout"], block_size=m["block_size"],
                         batch_size=t["batch_size"], max_lr=t["max_lr"],
                         max_steps=t["max_steps"], seed=t["seed"],
                         eval_seed=t["eval_seed"], eval_batches=t["eval_batches"],
                         best_step=s["best_step"], best_val_loss=s["best_val_loss"],
                         last_val_loss=s["last_val_loss"],
                         elapsed_minutes=s["elapsed_seconds"] / 60,
                         tokens_seen=s["tokens_seen"], data_sha256=s["data_sha256"]))
    if not rows:
        raise ValueError("No completed run records found")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_validation_comparison(run_dirs, output_dir):
    """Overlay validation curves at actual completed optimizer steps."""
    run_dirs = list(map(Path, run_dirs))
    units = {json.loads((p / "summary.json").read_text()).get("loss_unit", "nats/character")
             for p in run_dirs}
    if len(units) != 1:
        raise ValueError("Do not overlay CE from different tokenizer units")
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_dir in map(Path, run_dirs):
        history = json.loads((run_dir / "history.json").read_text())
        line, = ax.plot([row["step"] for row in history],
                        [row["val_loss"] for row in history], label=run_dir.name)
        best = min(history, key=lambda row: row["val_loss"])
        ax.scatter(best["step"], best["val_loss"], color=line.get_color())
    ax.set(xlabel="Completed optimizer steps", ylabel=f"Validation CE ({next(iter(units))})")
    ax.legend()
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "validation_comparison.png", dpi=160)
    return fig


def attention_equivariance_demo():
    """Numerically illustrate the proof assumptions; this is NOT a proof.

    Fixed absolute positions break general permutation equivariance. A fixed
    causal mask also breaks arbitrary-permutation equivariance, even without PE.
    """
    rng = torch.Generator().manual_seed(7)
    x = torch.randn(5, 4, generator=rng, dtype=torch.float64)
    positions = torch.randn(5, 4, generator=rng, dtype=torch.float64)
    wq, wk, wv = [torch.randn(4, 4, generator=rng, dtype=torch.float64) for _ in range(3)]
    permutation = torch.tensor([2, 0, 4, 1, 3])

    def attention(z, causal=False):
        scores = (z @ wq) @ (z @ wk).T / math.sqrt(4)
        if causal:
            scores = scores.masked_fill(torch.ones(5, 5, dtype=torch.bool).triu(1), -torch.inf)
        return scores.softmax(-1) @ (z @ wv)

    return {
        "unmasked_no_positions_max_error":
            (attention(x[permutation]) - attention(x)[permutation]).abs().max().item(),
        "fixed_positions_max_error":
            (attention(x[permutation] + positions) -
             attention(x + positions)[permutation]).abs().max().item(),
        "positions_permuted_together_max_error":
            (attention((x + positions)[permutation]) -
             attention(x + positions)[permutation]).abs().max().item(),
        "fixed_causal_mask_no_positions_max_error":
            (attention(x[permutation], True) - attention(x, True)[permutation]).abs().max().item(),
    }
