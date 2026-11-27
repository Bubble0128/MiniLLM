"""Extra ablation runs filling in the gaps identified after the overnight batch.

Experiments (in order):
  E1  Continue D2 last.pt  — same arch, resume from D2's last checkpoint, 4000 more steps
  E2  block=512, weight_tying=False, dropout=0.1, batch=16  (from scratch)
  E3  block=256, weight_tying=False, dropout=0.2, lr=3e-4   (from scratch)
  E4  block=512, weight_tying=False, dropout=0.2, batch=8   (from scratch)

All use AdamW, weight_decay=0.1, grad_clip=1.0, eval every 100 steps.
E1 uses resume=True (preserves optimizer state/step counter).
E2-E4 use patience=5 to auto-stop.

Usage (on server):
    nohup /home/chenrui/.venvs/minillm/bin/python -u run_extra.py \
        > run_extra.log 2>&1 &
"""
import csv
import json
import math
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable
DATA = ROOT / "data" / "shakespeare.txt"

# ── import project modules ────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from model import GPTConfig
from train import TrainConfig, train, write_json

# ── paths ─────────────────────────────────────────────────────────────────────
TAG = datetime.now(timezone.utc).strftime("extra_%Y%m%dT%H%M%SZ")
BATCH = ROOT / "runs" / TAG
BATCH.mkdir(parents=True)

D2_DIR = ROOT / "runs" / "overnight_20261127T013704_336480Z" / "D2_champion_finetune1000_batch16"
C2_BEST = ROOT / "runs" / "overnight_20261127T013704_336480Z" / "C2_no_weight_tying" / "best.pt"

results = []
failures = []


def run(name, mc, tc, *, init_checkpoint=None, resume_dir=None):
    out = BATCH / name
    print(f"\n{'='*60}", flush=True)
    print(f"START {name}", flush=True)
    print(f"  model : {asdict(mc)}", flush=True)
    print(f"  train : {asdict(tc)}", flush=True)
    if init_checkpoint:
        print(f"  init  : {init_checkpoint}", flush=True)
    if resume_dir:
        print(f"  resume: {resume_dir}", flush=True)
    try:
        if resume_dir:
            # copy last.pt etc. into new output dir and resume
            shutil.copytree(resume_dir, out)
            result = train(str(DATA), out, mc, tc, device="cuda", resume=True)
        else:
            result = train(str(DATA), out, mc, tc, device="cuda",
                           init_checkpoint=str(init_checkpoint) if init_checkpoint else None)
        results.append(dict(name=name, best_val_loss=result["best_val_loss"],
                            best_step=result["best_step"], stop_reason=result.get("stop_reason"),
                            parameters=result["parameters"]))
        print(f"DONE {name}  best_val_CE={result['best_val_loss']:.6f}  "
              f"best_step={result['best_step']}  stop={result.get('stop_reason')}", flush=True)
    except Exception:
        failures.append(name)
        traceback.print_exc()
        print(f"FAILED {name}", flush=True)

    # running summary
    write_json(BATCH / "summary.json", dict(results=results, failures=failures))


# ── champion config (same as C2 / D1 / D2) ───────────────────────────────────
CHAMP_MC = GPTConfig(
    vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384,
    dropout=0.1, residual_scale=1.0, weight_tying=False,
)
BASE_TC = TrainConfig(
    max_lr=3e-4, min_lr=3e-5, warmup_steps=150,
    weight_decay=0.1, grad_clip=1.0,
    eval_interval=100, eval_batches=20,
    seed=42, eval_seed=12345,
    optimizer_name="adamw",
    eval_batch_size=64, eval_block_size=256,
)

# ── E1: resume D2 last.pt, 4000 more steps, lower lr ─────────────────────────
# Resume keeps D2's optimizer state; we just extend the step budget.
# train() with resume=True will add more steps on top of the existing step count.
# We copy D2's directory and set a step_limit or just extend max_steps.
# Because train() with resume checks that config matches, use same mc/tc as D2
# but increase max_steps from 1000 to 5000 (adds ~4000 steps from step 1000).
D2_MC = GPTConfig(
    vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384,
    dropout=0.1, residual_scale=1.0, weight_tying=False,
)
D2_TC = TrainConfig(
    max_steps=5000,          # was 1000 → extend to 5000 (4000 more steps)
    batch_size=16,
    max_lr=2e-4, min_lr=2e-5,
    warmup_steps=150,
    weight_decay=0.1, grad_clip=1.0,
    eval_interval=100, eval_batches=20,
    seed=42, eval_seed=12345,
    patience=5,              # stop if no improvement for 500 steps
    optimizer_name="adamw",
    eval_batch_size=64, eval_block_size=256,
)
run("E1_D2_resume_5k", D2_MC, D2_TC, resume_dir=D2_DIR)

# ── E2: block=512, weight_tying=False, dropout=0.1, batch=16 ─────────────────
E2_MC = replace(CHAMP_MC, block_size=512)
E2_TC = replace(BASE_TC,
                max_steps=5000, batch_size=16, patience=5,
                # keep eval at 256/64 for fair comparison with other runs
                eval_batch_size=64, eval_block_size=256)
run("E2_ctx512_notying_d01", E2_MC, E2_TC)

# ── E3: block=256, weight_tying=False, dropout=0.2 ───────────────────────────
E3_MC = replace(CHAMP_MC, dropout=0.2)
E3_TC = replace(BASE_TC, max_steps=5000, batch_size=64, patience=5)
run("E3_ctx256_notying_d02", E3_MC, E3_TC)

# ── E4: block=512, weight_tying=False, dropout=0.2, batch=8 ──────────────────
E4_MC = replace(CHAMP_MC, block_size=512, dropout=0.2)
E4_TC = replace(BASE_TC,
                max_steps=5000, batch_size=8, patience=5,
                eval_batch_size=64, eval_block_size=256)
run("E4_ctx512_notying_d02", E4_MC, E4_TC)

# ── final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print(f"FINISHED — {len(results)} ok, {len(failures)} failed", flush=True)
for r in sorted(results, key=lambda x: x["best_val_loss"]):
    print(f"  {r['name']:35s}  CE={r['best_val_loss']:.6f}  "
          f"best@{r['best_step']}  stop={r['stop_reason']}", flush=True)
if failures:
    print(f"FAILURES: {failures}", flush=True)
