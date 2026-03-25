"""
train.py
--------
Generic training loop for CBOW, Skip-gram, and Skip-gram + Negative Sampling.
"""

import torch
import torch.nn as nn
from utils.save_load import save_model


def train_model(model, dataloader, epochs=20, lr=0.001, save_path="models/model.pth",
                model_type="cbow", word2idx=None, idx2word=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if model_type == "skipgram_ns":
        # SkipGramNSModel computes its own loss inside forward()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*50}")
    print(f"  Training {model_type.upper()} | epochs={epochs} | lr={lr}")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            if model_type == "cbow":
                context, target = batch
                context, target = context.to(device), target.to(device)
                output = model(context)
                loss = criterion(output, target)

            elif model_type == "skipgram":
                target, context = batch
                target, context = target.to(device), context.to(device)
                output = model(target)
                loss = criterion(output, context)

            elif model_type == "skipgram_ns":
                target, context, neg_samples = batch
                target = target.to(device)
                context = context.to(device)
                neg_samples = neg_samples.to(device)
                loss = model(target, context, neg_samples)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch:>3}/{epochs}  |  Loss: {avg_loss:.4f}")

    # Save model and vocab
    save_model(model, word2idx, idx2word, save_path)
    print(f"\n Model saved at {save_path}")
