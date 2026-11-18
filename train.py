"""Train the tutorial GPT with reproducible evaluation and resumable runs."""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch

from data import CharacterData, SOURCE_REVISION, download_shakespeare
from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    max_steps: int = 5000
    batch_size: int = 64
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    seed: int = 42
    eval_seed: int = 12345

    def __post_init__(self):
        """Validate optimization and evaluation settings."""
        for name in ("max_steps", "batch_size", "eval_interval", "eval_batches"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("Require 0 <= warmup_steps < max_steps")
        if not 0 <= self.min_lr <= self.max_lr or self.max_lr <= 0:
            raise ValueError("Require 0 <= min_lr <= max_lr and max_lr > 0")
        if self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("grad_clip must be positive; weight_decay non-negative")


def experiment_config(name):
    """Return fresh configs; full experiments differ in one stated factor."""
    model, training = GPTConfig(), TrainConfig()
    if name == "smoke":
        return (replace(model, block_size=32, n_layer=2, n_head=2, n_embd=64),
                replace(training, max_steps=20, batch_size=4, warmup_steps=2,
                        eval_interval=10, eval_batches=2))
    if name == "baseline":
        return model, training
    if name == "dropout":
        return replace(model, dropout=0.1), training
    if name == "small":
        return replace(model, n_layer=4, n_head=4, n_embd=256), training
    raise ValueError("Choose smoke, baseline, dropout, or small")


def get_device():
    """Prefer Colab CUDA, then Apple MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(step, config):
    """Linear warmup and cosine decay, as in tutorial Part 3."""
    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    progress = min(1.0, (step - config.warmup_steps) /
                   (config.max_steps - config.warmup_steps))
    return config.min_lr + 0.5 * (config.max_lr - config.min_lr) * (
        1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, data, config, device):
    """Use the same held-out windows at each evaluation, independent of training.

    Train and validation losses are both measured in eval mode, without dropout.
    This is a sampled estimate, not a full-corpus score or an independent test set.
    """
    was_training = model.training
    model.eval()
    scores = {}
    try:
        for offset, split in enumerate(("train", "val")):
            rng = torch.Generator().manual_seed(config.eval_seed + offset)
            losses = []
            for _ in range(config.eval_batches):
                x, y = data.batch(split, model.config.block_size,
                                  config.batch_size, device, rng)
                _, loss = model(x, y)
                losses.append(loss.item())
            scores[split] = sum(losses) / len(losses)
            if not math.isfinite(scores[split]):
                raise FloatingPointError(f"Non-finite {split} loss")
        return scores
    finally:
        model.train(was_training)


def write_json(path, value):
    """Replace a JSON file atomically, preserving the previous file on failure."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, payload):
    """Atomically replace a tensor/primitive-only PyTorch checkpoint."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def source_metadata():
    """Record code hashes and a Git revision when available for the report."""
    root = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"git_revision": revision, "source_tutorial_revision": SOURCE_REVISION,
            "file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                            for name in ("model.py", "data.py", "train.py", "generate.py")}}


def train(data_path, output_dir, model_config, train_config, device=None,
          resume=False, step_limit=None):
    """Train or resume one experiment, saving best/last checkpoints and metrics.

    step_limit pauses at a completed-step count without changing the LR schedule.
    It is useful for a resumability check; normal runs leave it unset.
    """
    device = torch.device(device) if device is not None else get_device()
    config = train_config
    data = CharacterData(data_path)
    model_config = replace(model_config, vocab_size=len(data.chars))
    if any(len(tokens) <= model_config.block_size for tokens in data.splits.values()):
        raise ValueError("Both data splits must be longer than block_size")
    output_dir = Path(output_dir)
    last_path = output_dir / "last.pt"
    if resume and not last_path.exists():
        raise FileNotFoundError(f"No checkpoint to resume: {last_path}")
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run already exists: {output_dir}; use resume=True or a new name")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr,
                                 weight_decay=config.weight_decay)
    train_rng = torch.Generator().manual_seed(config.seed + 1)
    history, completed, best_loss, best_step, elapsed_before = [], 0, float("inf"), 0, 0.0
    manifest = dict(model_config=asdict(model_config), train_config=asdict(config),
                    data_sha256=data.sha256, data_split_index=data.split_index,
                    train_chars=len(data.splits["train"]), val_chars=len(data.splits["val"]),
                    chars=data.chars, parameters=sum(p.numel() for p in model.parameters()),
                    python=platform.python_version(), torch=str(torch.__version__),
                    device=str(device), gpu=torch.cuda.get_device_name(device)
                    if device.type == "cuda" else None, **source_metadata())
    if resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        for key in ("model_config", "train_config", "data_sha256"):
            if state[key] != manifest[key]:
                raise ValueError(f"Cannot resume: {key} changed; use a new output directory")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        train_rng.set_state(state["batch_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            if len(state["cuda_rng"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        completed, history = state["step"], state["history"]
        best_loss, best_step = state["best_val_loss"], state["best_step"]
        elapsed_before = state["elapsed_seconds"]
        # Record environment/code changes on resume; exact replay is only expected
        # on the same backend/software and unchanged implementation.
        old_manifest = json.loads((output_dir / "config.json").read_text())
        old_manifest.setdefault("resume_sessions", []).append(manifest)
        write_json(output_dir / "config.json", old_manifest)
    else:
        write_json(output_dir / "config.json", manifest)
    stop_at = config.max_steps if step_limit is None else min(step_limit, config.max_steps)
    if stop_at < completed:
        raise ValueError("step_limit is earlier than the resumed step")
    print(f"Device: {device} | parameters: {manifest['parameters']:,} | "
          f"steps: {completed} -> {stop_at}", flush=True)
    started = time.perf_counter()

    def export_metrics():
        """Rebuild sidecar files from checkpoint history after an interrupted save."""
        row = history[-1]
        write_json(output_dir / "history.json", history)
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        summary = dict(
            run=output_dir.name, step=row["step"], max_steps=config.max_steps,
            complete=row["step"] == config.max_steps, best_step=best_step,
            best_val_loss=best_loss, best_val_perplexity=math.exp(best_loss),
            last_val_loss=row["val_loss"], parameters=manifest["parameters"],
            elapsed_seconds=row["elapsed_seconds"], tokens_seen=row["tokens_seen"],
            model_config=asdict(model_config), train_config=asdict(config),
            data_sha256=data.sha256)
        write_json(output_dir / "summary.json", summary)
        return summary

    def record(step):
        """Evaluate, update best selection, and save a complete recovery point."""
        nonlocal best_loss, best_step
        scores = evaluate(model, data, config, device)
        elapsed = elapsed_before + time.perf_counter() - started
        row = dict(step=step, train_loss=scores["train"], val_loss=scores["val"],
                   lr=get_lr(max(0, step - 1), config), elapsed_seconds=elapsed,
                   tokens_seen=step * config.batch_size * model_config.block_size)
        history.append(row)
        is_best = scores["val"] < best_loss
        if is_best:
            best_loss, best_step = scores["val"], step
        payload = dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       model_config=asdict(model_config), train_config=asdict(config),
                       chars=data.chars, data_sha256=data.sha256, val_loss=scores["val"],
                       best_val_loss=best_loss, best_step=best_step, history=history,
                       elapsed_seconds=elapsed, batch_rng=train_rng.get_state(),
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()
                       if device.type == "cuda" else [])
        if is_best:
            save_checkpoint(output_dir / "best.pt", payload)
        save_checkpoint(last_path, payload)
        export_metrics()
        print(f"Step {step:5d} | train {scores['train']:.4f} | val {scores['val']:.4f} "
              f"| best {best_loss:.4f} at {best_step} | {elapsed/60:.1f} min", flush=True)

    if not history:
        record(0)
    model.train()
    for step in range(completed, stop_at):
        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, config)
        x, y = data.batch("train", model_config.block_size, config.batch_size,
                          device, train_rng)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; reduce learning rate")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                       error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        if completed % config.eval_interval == 0 or completed == stop_at:
            record(completed)
    return export_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["smoke", "baseline", "dropout", "small"],
                        default="smoke")
    parser.add_argument("--data", default="data/shakespeare.txt")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    model_config, train_config = experiment_config(args.experiment)
    train(download_shakespeare(args.data), args.output or f"runs/{args.experiment}",
          model_config, train_config, device=args.device, resume=args.resume)
