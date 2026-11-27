"""Train-only word vocabulary; punctuation and whitespace remain separate tokens.

This is whole-word tokenization, not BPE. Rare/unseen words map to <unk>.
The raw 90/10 character boundary matches the character experiments exactly.
"""
from collections import Counter
import hashlib
import re

import torch

from data import CharacterData

PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+|\s+|[^A-Za-z0-9\s]"
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+")


class WordTokenizer:
    """Case-sensitive, deterministic tokenizer fitted only to training text."""
    def __init__(self, vocabulary, min_frequency=2):
        self.vocabulary = list(vocabulary)
        if not self.vocabulary or self.vocabulary[0] != '<unk>':
            raise ValueError('Vocabulary must reserve index 0 for <unk>')
        self.stoi = {word: index for index, word in enumerate(self.vocabulary)}
        if len(self.stoi) != len(self.vocabulary):
            raise ValueError('Duplicate vocabulary items')
        self.min_frequency = min_frequency

    @staticmethod
    def split(text):
        tokens = re.findall(PATTERN, text)
        if ''.join(tokens) != text:
            raise ValueError('Tokenizer did not cover the entire input')
        return tokens

    @classmethod
    def fit(cls, text, min_frequency=2):
        counts = Counter(cls.split(text))
        # Keep all punctuation/layout; rare words create training targets for UNK.
        vocabulary = ['<unk>'] + sorted(t for t, n in counts.items()
                                        if n >= min_frequency or not WORD.fullmatch(t))
        return cls(vocabulary, min_frequency)

    def encode(self, text):
        return [self.stoi.get(token, 0) for token in self.split(text)]

    def decode(self, ids):
        return ''.join(self.vocabulary[int(index)] for index in ids)

    def metadata(self):
        return dict(type='word', pattern=PATTERN, case_sensitive=True,
                    min_frequency=self.min_frequency, unknown_token='<unk>',
                    vocabulary=self.vocabulary, fitted_on='training_split_only',
                    whitespace_and_punctuation='separate, preserved when in vocabulary')

    def statistics(self, text):
        pieces = self.split(text)
        words = [p for p in pieces if WORD.fullmatch(p)]
        unknown = [p for p in pieces if p not in self.stoi]
        unknown_words = [p for p in words if p not in self.stoi]
        return dict(characters=len(text), tokens=len(pieces), word_tokens=len(words),
                    unknown_tokens=len(unknown), unknown_token_rate=len(unknown) / max(1, len(pieces)),
                    unknown_word_rate=len(unknown_words) / max(1, len(words)))


class WordData(CharacterData):
    """Reuse only the split-local window sampler from CharacterData."""
    def __init__(self, path):
        from pathlib import Path
        raw = Path(path).read_bytes()
        text = raw.decode('utf-8')
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.split_index = int(0.9 * len(text))
        train_text, val_text = text[:self.split_index], text[self.split_index:]
        self.tokenizer = WordTokenizer.fit(train_text)
        # Compatibility field: this contains vocabulary tokens, NOT characters.
        self.chars = self.tokenizer.vocabulary
        self.splits = {name: torch.tensor(self.encode(value), dtype=torch.long)
                       for name, value in [('train', train_text), ('val', val_text)]}
        self.train_char_count, self.val_char_count = len(train_text), len(val_text)
        self.statistics = {name: self.tokenizer.statistics(value)
                           for name, value in [('train', train_text), ('val', val_text)]}

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
