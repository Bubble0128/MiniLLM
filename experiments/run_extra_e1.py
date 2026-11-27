"""E1 fix: fine-tune from D2's best.pt with a fresh optimizer, lower lr, 4000 steps.
Run after run_extra.py has started (E2/E3/E4 running in background).
Waits for E2/E3/E4 to finish first so GPU isn't double-booked.
"""
import sys, time
from pathlib import Path
from dataclasses import asdict, replace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import GPTConfig
from train import TrainConfig, train, write_json

DATA = ROOT / "data" / "shakespeare.txt"
D2_BEST = ROOT / "runs" / "overnight_20261127T013704_336480Z" / "D2_champion_finetune1000_batch16" / "best.pt"

# Find the extra batch directory (most recent extra_* dir)
runs = ROOT / "runs"
extra_dirs = sorted(runs.glob("extra_*"))
assert extra_dirs, "No extra_* run directory found yet"
BATCH = extra_dirs[-1]
print(f"Appending E1 to batch: {BATCH}", flush=True)

# Wait for E2/E3/E4 to finish (poll every 30s)
print("Waiting for E2/E3/E4 to finish...", flush=True)
while True:
    summary_path = BATCH / "summary.json"
    if summary_path.exists():
        import json
        s = json.loads(summary_path.read_text())
        done = {r["name"] for r in s.get("results", [])} | set(s.get("failures", []))
        expected = {"E2_ctx512_notying_d01", "E3_ctx256_notying_d02", "E4_ctx512_notying_d02"}
        if expected.issubset(done):
            print(f"E2/E3/E4 done. Results: {s['results']}", flush=True)
            break
    time.sleep(30)

# E1: init from D2 best.pt, fresh optimizer, lr=1e-4->1e-5, 4000 steps, patience=5
MC = GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384,
               dropout=0.1, residual_scale=1.0, weight_tying=False)
TC = TrainConfig(max_steps=4000, batch_size=16,
                 max_lr=1e-4, min_lr=1e-5,
                 warmup_steps=100, weight_decay=0.1, grad_clip=1.0,
                 eval_interval=100, eval_batches=20,
                 seed=42, eval_seed=12345, patience=5,
                 optimizer_name="adamw",
                 eval_batch_size=64, eval_block_size=256)

out = BATCH / "E1_D2_finetune2_4k"
print(f"\nSTART E1_D2_finetune2_4k", flush=True)
print(f"  init: {D2_BEST}", flush=True)
print(f"  model: {asdict(MC)}", flush=True)
print(f"  train: {asdict(TC)}", flush=True)

try:
    result = train(str(DATA), out, MC, TC, device="cuda", init_checkpoint=str(D2_BEST))
    print(f"DONE E1_D2_finetune2_4k  CE={result['best_val_loss']:.6f}  "
          f"best@{result['best_step']}  stop={result.get('stop_reason')}", flush=True)
    # append to summary
    import json
    summary_path = BATCH / "summary.json"
    s = json.loads(summary_path.read_text())
    s["results"].append(dict(name="E1_D2_finetune2_4k",
                             best_val_loss=result["best_val_loss"],
                             best_step=result["best_step"],
                             stop_reason=result.get("stop_reason"),
                             parameters=result["parameters"]))
    write_json(summary_path, s)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"FAILED: {e}", flush=True)

print("\nAll done.", flush=True)
