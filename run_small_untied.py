"""Small model with the two changes that helped Medium: untied output projection, weight decay 0.1.

Everything else matches the best earlier Small run (small_scale_d01_p5_10k_5080_v1): peak
learning rate 0.001, 100 warmup steps, 10,000-step cosine horizon, patience 5, dropout 0.1.

Usage: python run_small_untied.py [training_seed] [untied|tied] [weight_decay]
The optional arguments give the second-seed check and the single-change controls.
"""
from dataclasses import replace
from pathlib import Path
import sys

from data import download_shakespeare
from train import experiment_config, train

root = Path(__file__).resolve().parent
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
tying = sys.argv[2] if len(sys.argv) > 2 else "untied"
weight_decay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
if tying not in ("untied", "tied"):
    raise SystemExit("second argument must be untied or tied")
model_config, train_config = experiment_config("small")
# Improvement under test: separate output projection and stronger weight decay.
model_config = replace(model_config, weight_tying=(tying == "tied"))
train_config = replace(train_config, weight_decay=weight_decay, seed=seed)
name = (f"small_{tying}_wd{str(weight_decay).replace('.', '')}"
        + ("" if seed == 42 else f"_seed{seed}") + "_5080_v1")
summary = train(download_shakespeare(root / "data/shakespeare.txt"), root / "runs" / name,
                model_config, train_config, device="cuda")
print(f"DONE {name} best_val_CE={summary['best_val_loss']:.6f} best_step={summary['best_step']} "
      f"step={summary['step']} stop={summary.get('stop_reason')}", flush=True)
