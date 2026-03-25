"""
cbow_model.py
-------------
Continuous Bag of Words (CBOW) model in PyTorch.
Predicts the center word from its surrounding context words.
"""

import torch
import torch.nn as nn


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, context):
        embeds = self.embeddings(context)          # (B, 2*W, D)
        mean_embed = embeds.mean(dim=1)            # (B, D)
        out = self.linear(mean_embed)              # (B, V)
        return out

    def get_embeddings(self):
        """Return the embedding weight matrix as a numpy array."""
        return self.embeddings.weight.data.cpu().numpy()
