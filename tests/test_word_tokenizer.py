"""Whole-word leakage/UNK checks and gradient-accumulation equivalence."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from generate import load_model
from model import GPT, GPTConfig
from train import TrainConfig, train
from word_data import WordData, WordTokenizer


class WordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_only_vocabulary_and_layout(self):
        tokenizer = WordTokenizer.fit("Known Known rare!\nDon't Don't ")
        self.assertEqual(tokenizer.encode('rare'), [0])
        self.assertEqual(tokenizer.encode('Unseen'), [0])
        text = "Known!\nDon't "
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.txt'
            path.write_text('hello ' * 150 + 'Novelword ' * 10)
            data = WordData(path)
            self.assertEqual(data.split_index, 900)
            self.assertNotIn('Novelword', data.tokenizer.stoi)
            self.assertEqual(data.statistics['val']['unknown_word_rate'], 1)
            x, y = data.batch('train', 4, 2, 'cpu', torch.Generator().manual_seed(42))
            torch.testing.assert_close(x[:, 1:], y[:, :-1])

    def test_accumulation_matches_full_batch_without_dropout(self):
        torch.manual_seed(42)
        mc = GPTConfig(vocab_size=5, block_size=4, n_layer=1, n_head=1, n_embd=8)
        whole, micro = GPT(mc), GPT(mc)
        micro.load_state_dict(whole.state_dict())
        x, y = torch.randint(5, (5, 4)), torch.randint(5, (5, 4))
        whole(x, y)[1].backward()
        for start in range(0, 5, 2):
            xx, yy = x[start:start+2], y[start:start+2]
            (micro(xx, yy)[1] * len(xx) / len(x)).backward()
        for a, b in zip(whole.parameters(), micro.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)

    def test_word_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'data.txt'
            path.write_text('To be, or not to be: question!\n' * 100)
            mc = GPTConfig(block_size=4, n_layer=1, n_head=1, n_embd=8, weight_tying=False)
            tc = TrainConfig(max_steps=4, warmup_steps=1, batch_size=4, eval_interval=2,
                             eval_batches=1, patience=0, micro_batch_size=2, tokenizer_type='word')
            train(path, root / 'run', mc, tc, device='cpu', step_limit=2)
            result = train(path, root / 'run', mc, tc, device='cpu', resume=True)
            self.assertEqual(result['step'], 4)
            self.assertEqual(result['loss_unit'], 'nats/word_or_layout_token')
            with self.assertRaisesRegex(ValueError, 'Tokenizer mismatch'):
                load_model(root / 'run/best.pt')
            model, state = load_model(root / 'run/best.pt', expected_tokenizer='word')
            tokenizer = WordTokenizer(state['tokenizer']['vocabulary'])
            self.assertEqual(model.config.vocab_size, len(tokenizer.vocabulary))
            self.assertTrue((root / 'run/tokenizer.json').exists())


if __name__ == '__main__':
    unittest.main()
