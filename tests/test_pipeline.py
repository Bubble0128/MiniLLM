"""Behavioral checks for causal prediction, actual learning and exact CPU resume.

Run: python -m unittest discover -s tests -v
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from analysis import attention_equivariance_demo
from data import CharacterData
from generate import generate, load_model, save_samples
from model import GPT, GPTConfig
from train import TrainConfig, evaluate, train


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Keep these CPU checks small and deterministic."""
        torch.set_num_threads(1)

    def test_causality_and_gradients(self):
        """Changing future tokens must not change prefix predictions."""
        torch.manual_seed(1)
        model = GPT(GPTConfig(vocab_size=5, block_size=8, n_layer=2, n_head=2, n_embd=16))
        a = torch.tensor([[0, 1, 2, 3, 4, 0]])
        b = torch.tensor([[0, 1, 2, 1, 1, 1]])
        logits, loss = model(a, b)
        other, _ = model(b)
        torch.testing.assert_close(logits[:, :3], other[:, :3])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_training_resume_generation_and_data(self):
        """A paused dropout run must match uninterrupted CPU training exactly."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            text = "abc abc\n" * 300
            path = root / "data.txt"
            path.write_text(text)
            data = CharacterData(path)
            self.assertEqual(data.decode(data.encode(text)), text)
            self.assertEqual(len(data.splits["train"]), int(len(text) * 0.9))
            x, y = data.batch("val", 8, 3, "cpu", torch.Generator().manual_seed(1))
            torch.testing.assert_close(x[:, 1:], y[:, :-1])
            mc = GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.1)
            tc = TrainConfig(max_steps=300, batch_size=8, warmup_steps=10, max_lr=0.003,
                             min_lr=0.0003, eval_interval=100, eval_batches=2)
            full = train(path, root / "full", mc, tc, device="cpu")
            train(path, root / "paused", mc, tc, device="cpu", step_limit=100)
            resumed = train(path, root / "paused", mc, tc, device="cpu", resume=True)
            left = torch.load(root / "full/last.pt", weights_only=True)
            right = torch.load(root / "paused/last.pt", weights_only=True)
            for name in left["model"]:
                torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
            self.assertEqual(full["best_val_loss"], resumed["best_val_loss"])
            self.assertLess(full["last_val_loss"], left["history"][0]["val_loss"] * 0.6)
            # Colab can disconnect after last.pt is saved but before sidecar files.
            for name in ("summary.json", "history.json", "metrics.csv"):
                (root / "paused" / name).unlink()
            recovered = train(path, root / "paused", mc, tc, device="cpu", resume=True)
            self.assertEqual(recovered, resumed)
            self.assertTrue((root / "paused/metrics.csv").exists())
            model, state = load_model(root / "full/best.pt")
            model.train()
            rng_before = torch.get_rng_state().clone()
            scores = evaluate(model, data, tc, "cpu")
            self.assertTrue(model.training)
            torch.testing.assert_close(rng_before, torch.get_rng_state())
            self.assertEqual(scores, evaluate(model, data, tc, "cpu"))
            prompt = "abc abc abc abc "  # deliberately longer than context
            first = generate(model, prompt, state["chars"], 20, top_k=999)
            self.assertTrue(model.training)
            self.assertTrue(first.startswith(prompt))
            self.assertEqual(len(first), len(prompt) + 20)
            self.assertEqual(first, generate(model, prompt, state["chars"], 20, top_k=999))
            torch.testing.assert_close(rng_before, torch.get_rng_state())
            self.assertEqual(generate(model, prompt, state["chars"], 10, temperature=0, seed=1),
                             generate(model, prompt, state["chars"], 10, temperature=0, seed=2))
            for bad_prompt in ("", "unknown☃"):
                with self.assertRaises(ValueError):
                    generate(model, bad_prompt, state["chars"])
            records = save_samples(root / "full/best.pt", root / "samples", prompt="abc ",
                                   temperatures=(0.8,), seeds=(42,), max_new_tokens=5)
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads((root / "samples/samples.json").read_text()), records)
            with self.assertRaises(FileExistsError):
                train(path, root / "full", mc, tc, device="cpu")
            with self.assertRaises(ValueError):
                train(path, root / "full", mc, replace(tc, max_lr=0.02), device="cpu", resume=True)

    def test_theory_assumptions(self):
        """Check the numerical examples distinguish fixed vs permuted positions."""
        result = attention_equivariance_demo()
        self.assertLess(result["unmasked_no_positions_max_error"], 1e-10)
        self.assertLess(result["positions_permuted_together_max_error"], 1e-10)
        self.assertGreater(result["fixed_positions_max_error"], 1e-4)
        self.assertGreater(result["fixed_causal_mask_no_positions_max_error"], 1e-4)


if __name__ == "__main__":
    unittest.main()
