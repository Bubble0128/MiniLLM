"""Repeat a named configuration with another training seed to measure run-to-run variation.

Usage: python experiments/seed_check.py EXPERIMENT SEED      (for example: final 2026)
Results are written to runs/original_data/ in a folder named after the configuration and seed.
"""
from dataclasses import replace
from pathlib import Path
import sys

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # core modules are in the repository root
from data import download_shakespeare
from train import experiment_config, run_folder, train

experiment, seed = sys.argv[1], int(sys.argv[2])
root = Path(__file__).resolve().parent.parent
model_config, train_config = experiment_config(experiment)
train_config = replace(train_config, seed=seed)
name = run_folder(model_config, train_config)
summary = train(download_shakespeare(root / "data/shakespeare.txt"), root / "runs" / name,
                model_config, train_config, device="cuda")
print(f"DONE {name} best_val_CE={summary['best_val_loss']:.6f} best_step={summary['best_step']} "
      f"step={summary['step']} stop={summary.get('stop_reason')}", flush=True)
