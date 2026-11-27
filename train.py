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
    max_steps: int = 10000
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
    # Improvement: stop after this many evaluations without a strictly lower CE.
    patience: int = 5
    # Course ablation: AdamW (default) or SGD with momentum.
    optimizer_name: str = "adamw"
    momentum: float = 0.9
    # Keep evaluation windows identical when training B or T changes.
    eval_batch_size: int | None = None
    eval_block_size: int | None = None
    tokenizer_type: str = "character"
    # Gradient accumulation preserves effective B when the word vocabulary grows.
    micro_batch_size: int | None = None
    # Improvement: extra training text placed before the original training split.
    extra_train_path: str | None = None

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
        if not isinstance(self.patience, int) or self.patience < 0:
            raise ValueError("patience must be a non-negative integer; 0 disables stopping")
        if self.optimizer_name not in ("adamw", "sgd"):
            raise ValueError("optimizer_name must be adamw or sgd")
        if self.tokenizer_type not in ("character", "word"):
            raise ValueError("tokenizer_type must be character or word")
        if self.micro_batch_size is not None and self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")


def experiment_config(name):
    """Named configurations: tutorial baseline, the size comparison, and the final model."""
    model, training = GPTConfig(), TrainConfig()
    if name == "baseline":
        # Historical configuration retained for replay/resumption of existing runs.
        return model, replace(training, max_steps=5000, patience=0)
    model = replace(model, dropout=0.1)
    if name == "final":
        # Improvement (report Section 3.4): Small with an untied output projection and
        # stronger weight decay; every other setting is the size-experiment default.
        return (replace(model, n_layer=4, n_head=4, n_embd=256, weight_tying=False),
                replace(training, weight_decay=0.1))
    if name == "medium_tuned":
        # Improvement (report Sections 3.2-3.3): untied output projection, lower peak
        # learning rate, stronger weight decay, longer warmup, fixed 5,000-step budget.
        return (replace(model, weight_tying=False),
                replace(training, max_steps=5000, max_lr=3e-4, min_lr=3e-4 / 10,
                        warmup_steps=150, weight_decay=0.1, patience=0,
                        eval_batch_size=64, eval_block_size=256))
    if name in ("medium", "dropout"):
        return model, training
    if name == "tiny":
        return replace(model, n_layer=2, n_head=2, n_embd=128), training
    if name == "small":
        return replace(model, n_layer=4, n_head=4, n_embd=256), training
    raise ValueError("Choose tiny, small, medium, baseline, dropout, medium_tuned, or final")


def describe_run(model_config, train_config, continued=False):
    """Folder name that shows tokenizer, architecture, main hyperparameters and seed."""
    m, t = model_config, train_config
    parts = ["word" if t.tokenizer_type == "word" else "char",
             f"L{m.n_layer}-H{m.n_head}-D{m.n_embd}", f"ctx{m.block_size}", f"drop{m.dropout:g}",
             f"lr{t.max_lr:g}", f"wd{t.weight_decay:g}", "tied" if m.weight_tying else "untied"]
    if m.residual_scale != 1.0:
        parts.append(f"resscale{m.residual_scale:g}")
    if t.optimizer_name != "adamw":
        parts.append(t.optimizer_name)
    if t.batch_size != 64:
        parts.append(f"batch{t.batch_size}")
    parts.append(f"steps{t.max_steps}" + (f"-earlystop{t.patience}" if t.patience else ""))
    parts.append(f"seed{t.seed}")
    if continued:
        parts.append("continued")
    return "_".join(parts)


def run_folder(model_config, train_config, continued=False):
    """Results path relative to runs/: the first level names the training data."""
    data = "extended_data" if train_config.extra_train_path else "original_data"
    return f"{data}/{describe_run(model_config, train_config, continued)}"


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
                x, y = data.batch(split, config.eval_block_size or model.config.block_size,
                                  config.eval_batch_size or config.batch_size, device, rng)
                micro = config.micro_batch_size or len(x)
                total = 0.0
                for start in range(0, len(x), micro):
                    xx, yy = x[start:start + micro], y[start:start + micro]
                    logits, loss = model(xx, yy)
                    total += loss.item() * len(xx) / len(x)
                    del logits, loss
                losses.append(total)
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
                            for name in ("model.py", "data.py", "train.py", "generate.py", "word_data.py")
                            if (root / name).exists()}}


def train(data_path, output_dir, model_config, train_config, device=None,
          resume=False, step_limit=None, init_checkpoint=None):
    """Train or resume one experiment, saving best/last checkpoints and metrics.

    step_limit pauses at a completed-step count without changing the LR schedule.
    It is useful for a resumability check; normal runs leave it unset.
    """
    device = torch.device(device) if device is not None else get_device()
    config = train_config
    if resume and init_checkpoint:
        raise ValueError("Choose exact resume OR a new fine-tuning run")
    if config.eval_block_size and config.eval_block_size > model_config.block_size:
        raise ValueError("Evaluation context exceeds model context")
    if config.tokenizer_type == "word":
        from word_data import WordData
        data = WordData(data_path, extra_train_path=config.extra_train_path)
    else:
        data = CharacterData(data_path)
        if config.extra_train_path:
            # Improvement: more training text; vocabulary and validation split unchanged.
            data.add_training_text(config.extra_train_path)
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
    parent = None
    if init_checkpoint:
        state = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        if (asdict(GPTConfig(**state["model_config"])) != asdict(model_config)
                or state["data_sha256"] != data.sha256 or state["chars"] != data.chars
                or state.get("tokenizer_type", "character") != config.tokenizer_type):
            raise ValueError("Fine-tuning parent model/data/tokenizer mismatch")
        model.load_state_dict(state["model"])
        parent = dict(checkpoint=str(Path(init_checkpoint).resolve()), step=state["step"],
                      best_val_loss=state["val_loss"], optimizer_reset=True,
                      tokens_seen=state["history"][-1]["tokens_seen"])
        del state
    if config.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.max_lr,
                                    momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr,
                                     weight_decay=config.weight_decay)
    train_rng = torch.Generator().manual_seed(config.seed + 1)
    history, completed, best_loss, best_step, elapsed_before = [], 0, float("inf"), 0, 0.0
    bad_evaluations = 0
    manifest = dict(model_config=asdict(model_config), train_config=asdict(config),
                    data_sha256=data.sha256, data_split_index=data.split_index,
                    train_chars=getattr(data, "train_char_count", len(data.splits["train"])),
                    val_chars=getattr(data, "val_char_count", len(data.splits["val"])),
                    train_tokens=len(data.splits["train"]), val_tokens=len(data.splits["val"]),
                    tokenizer_type=config.tokenizer_type,
                    loss_unit="nats/character" if config.tokenizer_type == "character" else "nats/word_or_layout_token",
                    chars=data.chars, parameters=sum(p.numel() for p in model.parameters()),
                    python=platform.python_version(), torch=str(torch.__version__),
                    device=str(device), gpu=torch.cuda.get_device_name(device)
                    if device.type == "cuda" else None, **source_metadata())
    manifest["parent"] = parent
    if config.tokenizer_type == "word":
        manifest["tokenizer_statistics"] = data.statistics
        write_json(output_dir / "tokenizer.json", data.tokenizer.metadata())
    if resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        # Older checkpoints predate early stopping and always ran to max_steps.
        state["train_config"].setdefault("patience", 0)
        state["model_config"] = asdict(GPTConfig(**state["model_config"]))
        state["train_config"] = asdict(TrainConfig(**state["train_config"]))
        for key in ("model_config", "train_config", "data_sha256"):
            if state[key] != manifest[key]:
                raise ValueError(f"Cannot resume: {key} changed; use a new output directory")
        if state["chars"] != data.chars:
            raise ValueError("Cannot resume: tokenizer vocabulary changed")
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
        bad_evaluations = state.get("bad_evaluations", 0)
        parent = state.get("parent")
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

    def early_stopped():
        return config.patience > 0 and bad_evaluations >= config.patience

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
            complete=row["step"] == config.max_steps or early_stopped(), best_step=best_step,
            stop_reason="early_stopping" if early_stopped() else (
                "max_steps" if row["step"] == config.max_steps else "paused"),
            bad_evaluations=bad_evaluations,
            best_val_loss=best_loss,
            last_val_loss=row["val_loss"], parameters=manifest["parameters"],
            elapsed_seconds=row["elapsed_seconds"], tokens_seen=row["tokens_seen"],
            model_config=asdict(model_config), train_config=asdict(config),
            data_sha256=data.sha256, parent=parent,
            tokenizer_type=config.tokenizer_type, loss_unit=manifest["loss_unit"])
        write_json(output_dir / "summary.json", summary)
        return summary

    def record(step):
        """Evaluate, update best selection, and save a complete recovery point."""
        nonlocal best_loss, best_step, bad_evaluations
        scores = evaluate(model, data, config, device)
        elapsed = elapsed_before + time.perf_counter() - started
        row = dict(step=step, train_loss=scores["train"], val_loss=scores["val"],
                   lr=get_lr(max(0, step - 1), config), elapsed_seconds=elapsed,
                   tokens_seen=step * config.batch_size * model_config.block_size)
        history.append(row)
        is_best = scores["val"] < best_loss
        if is_best:
            best_loss, best_step = scores["val"], step
            bad_evaluations = 0
        else:
            bad_evaluations += 1
        payload = dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       model_config=asdict(model_config), train_config=asdict(config),
                       chars=data.chars, data_sha256=data.sha256, val_loss=scores["val"],
                       tokenizer_type=config.tokenizer_type,
                       tokenizer=data.tokenizer.metadata() if config.tokenizer_type == "word" else None,
                       best_val_loss=best_loss, best_step=best_step, history=history,
                       parent=parent,
                       bad_evaluations=bad_evaluations,
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
    if early_stopped():
        return export_metrics()
    model.train()
    for step in range(completed, stop_at):
        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, config)
        x, y = data.batch("train", model_config.block_size, config.batch_size,
                          device, train_rng)
        optimizer.zero_grad(set_to_none=True)
        micro = config.micro_batch_size or len(x)
        for start in range(0, len(x), micro):
            xx, yy = x[start:start + micro], y[start:start + micro]
            logits, loss = model(xx, yy)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; reduce learning rate")
            (loss * (len(xx) / len(x))).backward()
            del logits, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                       error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        if completed % config.eval_interval == 0 or completed == stop_at:
            record(completed)
            if early_stopped():
                print(f"Early stopping at {completed}: {bad_evaluations} evaluations "
                      f"without improvement; using best.pt from step {best_step}.", flush=True)
                break
    return export_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["tiny", "small", "medium", "baseline", "dropout", "medium_tuned", "final"],
                        default="medium")
    parser.add_argument("--data", default="data/shakespeare.txt")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int, help="Optimizer-step ceiling and cosine schedule horizon")
    parser.add_argument("--max-lr", type=float, help="Peak learning rate in the schedule")
    parser.add_argument("--min-lr", type=float, help="Final learning rate in the schedule")
    args = parser.parse_args()
    model_config, train_config = experiment_config(args.experiment)
    if args.max_steps is not None:
        train_config = replace(train_config, max_steps=args.max_steps)
    if args.max_lr is not None or args.min_lr is not None:
        train_config = replace(
            train_config,
            max_lr=train_config.max_lr if args.max_lr is None else args.max_lr,
            min_lr=train_config.min_lr if args.min_lr is None else args.min_lr,
        )
    train(download_shakespeare(args.data), args.output or f"runs/{args.experiment}",
          model_config, train_config, device=args.device, resume=args.resume)
