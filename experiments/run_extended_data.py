"""More data: retrain with 28 extra Shakespeare plays added to the training split.

data/shakespeare_extra.txt comes from prepare_extended_data.py and is built on first use. The validation split is
unchanged, so validation CE is directly comparable with the runs on the original text.

Usage: python experiments/run_extended_data.py {final|medium_tuned|word|word_small} [training_seed]
  final         Small tuned, exactly as in the report; only the training text grows.
  medium_tuned  Medium tuned with the schedule doubled to 10,000 steps for the larger text.
  word          Word-level tokenizer on the Medium tuned architecture, 10,000-step ceiling.
  word_small    Word-level tokenizer on the Small tuned architecture and schedule.
"""
from dataclasses import replace
import os
from pathlib import Path
import sys

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # core modules are in the repository root
from data import download_shakespeare
from train import experiment_config, train

EXTRA = "data/shakespeare_extra.txt"
experiment = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
root = Path(__file__).resolve().parent.parent
if experiment == "final":
    model_config, train_config = experiment_config("final")
elif experiment == "medium_tuned":
    model_config, train_config = experiment_config("medium_tuned")
    train_config = replace(train_config, max_steps=10000)
elif experiment == "word":
    model_config, train_config = experiment_config("medium_tuned")
    # Gradient accumulation in micro-batches of 32 keeps the effective batch at 64.
    train_config = replace(train_config, max_steps=10000, patience=5,
                           tokenizer_type="word", micro_batch_size=32)
elif experiment == "word_small":
    model_config, train_config = experiment_config("final")
    train_config = replace(train_config, tokenizer_type="word", micro_batch_size=32)
else:
    raise SystemExit("choose final, medium_tuned, word or word_small")
os.chdir(root)                       # the extra-text path is stored relative to the repository root
if not Path(EXTRA).exists():
    import prepare_extended_data
    prepare_extended_data.main()
# Improvement under test: 3.4 million extra training characters.
train_config = replace(train_config, extra_train_path=EXTRA, seed=seed)
name = f"{experiment}_extra" + ("" if seed == 42 else f"_seed{seed}") + "_5080_v1"
summary = train(download_shakespeare(root / "data/shakespeare.txt"), root / "runs" / name,
                model_config, train_config, device="cuda")
print(f"DONE {name} best_val_CE={summary['best_val_loss']:.6f} best_step={summary['best_step']} "
      f"step={summary['step']} stop={summary.get('stop_reason')}", flush=True)
