"""
skipgram_model.py
-----------------
Skip-gram models in PyTorch:
  1. SkipGramModel      — standard softmax-based skip-gram
  2. SkipGramNSModel    — skip-gram with negative sampling
"""

import torch
import torch.nn as nn


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, target):
        embeds = self.embeddings(target)       # (B, D)
        out = self.linear(embeds)              # (B, V)
        return out

    def get_embeddings(self):
        return self.embeddings.weight.data.cpu().numpy()


class SkipGramNSModel(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(SkipGramNSModel, self).__init__()
        # Match the saved state_dict keys exactly
        self.input_embed = nn.Embedding(vocab_size, embed_dim)
        self.output_embed = nn.Embedding(vocab_size, embed_dim)
        
        # The user's saved model also had an unused 'embeddings' layer.
        # We include it so load_state_dict() succeeds without strict=False.
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.input_embed.weight)
        nn.init.xavier_uniform_(self.output_embed.weight)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, target, context, neg_samples):
        target_emb = self.input_embed(target)         # (B, D)
        context_emb = self.output_embed(context)       # (B, D)
        neg_emb = self.output_embed(neg_samples)       # (B, K, D)

        # Positive score
        pos_score = torch.sum(target_emb * context_emb, dim=1)       # (B,)
        pos_score = torch.clamp(pos_score, max=10, min=-10)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)        # (B,)

        # Negative scores
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)   # (B,)

        return (pos_loss + neg_loss).mean()

    def get_embeddings(self):
        # We use input_embed as the primary representation
        return self.input_embed.weight.data.cpu().numpy()
