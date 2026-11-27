"""Checks for the new ablations and weight-only fine-tuning transition."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from data import CharacterData
from generate import load_model
from model import Block, GPT, GPTConfig
from train import TrainConfig, evaluate, train


class OvernightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_residual_and_weight_tying(self):
        config = GPTConfig(vocab_size=5, block_size=4, n_layer=1, n_head=1, n_embd=8)
        block = Block(replace(config, residual_scale=0.0))
        x = torch.randn(2, 4, 8)
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)
        half = Block(replace(config, residual_scale=0.5))
        h = x + 0.5 * half.attn(half.ln_1(x))
        torch.testing.assert_close(half(x), h + 0.5 * half.mlp(half.ln_2(h)))
        tied, untied = GPT(config), GPT(replace(config, weight_tying=False))
        self.assertIs(tied.transformer.wte.weight, tied.lm_head.weight)
        self.assertIsNot(untied.transformer.wte.weight, untied.lm_head.weight)
        self.assertEqual(sum(p.numel() for p in untied.parameters()) -
                         sum(p.numel() for p in tied.parameters()), 5 * 8)

    def test_sgd_finetune_and_common_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_path = root / 'data.txt'
            data_path.write_text('abc ' * 100)
            mc = GPTConfig(vocab_size=4, block_size=8, n_layer=1, n_head=1, n_embd=8,
                           residual_scale=0.5, weight_tying=False)
            tc = TrainConfig(max_steps=2, warmup_steps=0, batch_size=2, eval_interval=1,
                             eval_batches=1, eval_batch_size=2, eval_block_size=4,
                             patience=0, optimizer_name='sgd', max_lr=0.1, min_lr=0.01)
            train(data_path, root / 'parent', mc, tc, device='cpu')
            parent = torch.load(root / 'parent/best.pt', weights_only=True)
            final = torch.load(root / 'parent/last.pt', weights_only=True)
            self.assertTrue(all('momentum_buffer' in value for value in final['optimizer']['state'].values()))
            model, _ = load_model(root / 'parent/best.pt')
            data = CharacterData(data_path)
            self.assertEqual(evaluate(model, data, tc, 'cpu'),
                             evaluate(model, data, replace(tc, batch_size=1), 'cpu'))
            ft = replace(tc, optimizer_name='adamw', batch_size=1, max_lr=2e-4, min_lr=2e-5)
            train(data_path, root / 'ft', mc, ft, device='cpu', step_limit=0,
                  init_checkpoint=root / 'parent/best.pt')
            initial = torch.load(root / 'ft/last.pt', weights_only=True)
            self.assertEqual(initial['step'], 0)
            self.assertFalse(initial['optimizer']['state'])
            self.assertTrue(initial['parent']['optimizer_reset'])
            for key in parent['model']:
                torch.testing.assert_close(parent['model'][key], initial['model'][key], rtol=0, atol=0)
            self.assertEqual(initial['val_loss'], parent['val_loss'])
            result = train(data_path, root / 'ft', mc, ft, device='cpu', resume=True)
            self.assertEqual(result['step'], 2)
            self.assertEqual(result['parent']['step'], parent['step'])


if __name__ == '__main__':
    unittest.main()
