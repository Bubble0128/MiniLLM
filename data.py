"""Pinned Shakespeare data, reversible character encoding, disjoint splits."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

import torch

SOURCE_REVISION = "63069f99af8ce9a6ab6e261000f4f9f1daaa1ee9"
DATA_URL = ("https://raw.githubusercontent.com/angelos-p/llm-from-scratch/"
            f"{SOURCE_REVISION}/data/shakespeare.txt")
DATA_GIT_BLOB = "7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9"


def download_shakespeare(path="data/shakespeare.txt"):
    """Download the exact source dataset; verify Git's published blob digest."""
    path = Path(path)
    if path.exists():
        raw = path.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=60) as response:
            raw = response.read()
    digest = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if digest != DATA_GIT_BLOB:
        raise ValueError(f"Dataset differs from the pinned source: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return path


class CharacterData:
    """A 90/10 contiguous train/validation split, matching tutorial Part 3."""

    def __init__(self, path):
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.chars = sorted(set(text))
        self.stoi = {char: i for i, char in enumerate(self.chars)}
        self.itos = {i: char for char, i in self.stoi.items()}
        tokens = torch.tensor(self.encode(text), dtype=torch.long)
        self.split_index = int(0.9 * len(tokens))
        self.splits = {"train": tokens[:self.split_index],
                       "val": tokens[self.split_index:]}

    def encode(self, text):
        """Reject unseen characters instead of silently deleting prompt text."""
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"Characters outside vocabulary: {unknown!r}")
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        """Map token IDs back to the exact original characters."""
        return "".join(self.itos[int(i)] for i in ids)

    def batch(self, split, block_size, batch_size, device, generator):
        """Sample windows wholly inside one split; targets are shifted by one."""
        tokens = self.splits[split]
        if len(tokens) <= block_size:
            raise ValueError(f"{split} needs more than {block_size} characters")
        starts = torch.randint(len(tokens) - block_size, (batch_size,),
                               generator=generator)
        indices = starts[:, None] + torch.arange(block_size)[None, :]
        return tokens[indices].to(device), tokens[indices + 1].to(device)
