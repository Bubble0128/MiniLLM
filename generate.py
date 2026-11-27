"""Seeded temperature/top-k sampling, adapted from tutorial Part 4."""
import argparse
import json
from pathlib import Path

import torch

from model import GPT, GPTConfig

DEFAULT_PROMPT = "To be, or not to be, that is the question:\n"


def load_model(checkpoint_path, device="cpu", expected_tokenizer="character"):
    """Load tensor/primitive-only checkpoints without pickled custom classes."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("tokenizer_type", "character") != expected_tokenizer:
        raise ValueError("Tokenizer mismatch: use the word-generation entry point for word checkpoints")
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def generate(model, prompt, chars, max_new_tokens=800, temperature=0.8,
             top_k=40, seed=42):
    """Continue the full prompt; crop only the model's context window.

    A local CPU RNG gives repeatable sampling without consuming training RNG.
    temperature=0 selects greedy decoding. top_k=0 disables top-k filtering.
    """
    if not prompt:
        raise ValueError("The prompt must not be empty")
    if temperature < 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("temperature must be finite and non-negative")
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(f"Prompt contains unseen characters: {unknown!r}")
    device = next(model.parameters()).device
    idx = torch.tensor([[stoi[c] for c in prompt]], device=device)
    rng = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -model.config.block_size:])
            logits = logits[:, -1, :].float()
            if temperature == 0:
                token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    cutoff = torch.topk(logits, min(top_k, len(chars))).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probs = logits.softmax(dim=-1).cpu()
                token = torch.multinomial(probs, 1, generator=rng).to(device)
            idx = torch.cat((idx, token), dim=1)
        return "".join(chars[i] for i in idx[0].tolist())
    finally:
        model.train(was_training)


def save_samples(checkpoint_path, output_dir, device="cpu", prompt=DEFAULT_PROMPT,
                 temperatures=(0.6, 0.8, 1.0), top_k=40, seeds=(42, 123, 2026),
                 max_new_tokens=800):
    """Save every predeclared sample with checkpoint identity and parameters."""
    model, checkpoint = load_model(checkpoint_path, device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for temperature in temperatures:
        for seed in seeds:
            output = generate(model, prompt, checkpoint["chars"], max_new_tokens,
                              temperature, top_k, seed)
            name = f"temperature_{temperature:g}_topk_{top_k}_seed_{seed}.txt"
            (output_dir / name).write_text(output, encoding="utf-8")
            records.append(dict(file=name, prompt=prompt, text=output,
                                temperature=temperature, top_k=top_k, seed=seed,
                                max_new_tokens=max_new_tokens,
                                checkpoint=str(checkpoint_path),
                                checkpoint_step=checkpoint["step"],
                                validation_loss=checkpoint["val_loss"],
                                model_config=checkpoint["model_config"],
                                data_sha256=checkpoint["data_sha256"]))
    (output_dir / "samples.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, checkpoint = load_model(args.checkpoint, args.device)
    print(generate(model, args.prompt, checkpoint["chars"], args.max_new_tokens,
                   args.temperature, args.top_k, args.seed))
