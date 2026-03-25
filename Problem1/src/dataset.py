"""
dataset.py
----------
Word2Vec Dataset: reads the clean corpus, builds vocabulary,
and generates (context, target) training pairs for CBOW,
Skip-gram, and Skip-gram with Negative Sampling.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from collections import Counter


def build_vocab(corpus_path, min_freq=2):
    sentences = []
    word_freq = Counter()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) > 1:
                sentences.append(tokens)
                word_freq.update(tokens)

    # Filter by minimum frequency
    vocab = [w for w, c in word_freq.items() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    print(f"  Vocabulary built: {len(word2idx):,} words (min_freq={min_freq})")
    return sentences, word2idx, idx2word, word_freq


# ─────────────────────────────────────────────
# CBOW Dataset
# ─────────────────────────────────────────────

class CBOWDataset(Dataset):

    def __init__(self, sentences, word2idx, window_size=2):
        self.data = []
        for sent in sentences:
            indices = [word2idx[w] for w in sent if w in word2idx]
            for i in range(window_size, len(indices) - window_size):
                context = (
                    indices[i - window_size : i]
                    + indices[i + 1 : i + window_size + 1]
                )
                target = indices[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# ─────────────────────────────────────────────
# Skip-gram Dataset
# ─────────────────────────────────────────────

class SkipGramDataset(Dataset):
    def __init__(self, sentences, word2idx, window_size=2):
        self.data = []
        for sent in sentences:
            indices = [word2idx[w] for w in sent if w in word2idx]
            for i in range(len(indices)):
                for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                    if i != j:
                        self.data.append((indices[i], indices[j]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target, context = self.data[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)


# ─────────────────────────────────────────────
# Skip-gram + Negative Sampling Dataset
# ─────────────────────────────────────────────

class SkipGramNSDataset(Dataset):
    def __init__(self, sentences, word2idx, word_freq, window_size=2, num_neg=5):
        self.data = []
        self.num_neg = num_neg
        self.vocab_size = len(word2idx)

        # Build noise distribution: freq^(3/4)
        inv_map = {v: k for k, v in word2idx.items()}
        freqs = np.array([word_freq.get(inv_map[i], 1)
                          for i in range(len(word2idx))], dtype=np.float64)
        freqs = np.power(freqs, 0.75)
        self.noise_dist = freqs / freqs.sum()

        for sent in sentences:
            indices = [word2idx[w] for w in sent if w in word2idx]
            for i in range(len(indices)):
                for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                    if i != j:
                        self.data.append((indices[i], indices[j]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target, context = self.data[idx]
        neg_samples = np.random.choice(self.vocab_size, size=self.num_neg,
                                       replace=False, p=self.noise_dist)
        return (
            torch.tensor(target, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),
        )
