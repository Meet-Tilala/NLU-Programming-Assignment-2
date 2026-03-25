"""
PA-2 | Problem 2 | Training Script
=====================================
Trains all three character-level name generation models:
    1. VanillaRNN
    2. BidirectionalLSTM
    3. RNNWithAttention

For each model this script:
  - Builds the character vocabulary from TrainingNames.txt
  - Creates a DataLoader with a 90/10 train/val split
  - Runs the training loop with cross-entropy loss
  - Saves the best model checkpoint (lowest validation loss)
  - Plots training / validation loss curves

Usage:
    python train.py

Outputs (one per model):
    vanilla_rnn.pt         — best VanillaRNN checkpoint
    blstm.pt               — best BLSTM checkpoint
    rnn_attention.pt       — best RNNWithAttention checkpoint
    *_loss_curve.png       — loss plots
    training_summary.txt   — parameter counts and hyperparameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.dataset import load_names, CharVocab, get_dataloader
from src.models import VanillaRNN, BidirectionalLSTM, RNNWithAttention

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility — fix all random seeds for consistent results
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Shared hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
NAMES_FILE   = "data/TrainingNames.txt"
EMBED_DIM    = 64       # dimensionality of character embeddings
HIDDEN_SIZE  = 256      # RNN / LSTM hidden state size
NUM_LAYERS   = 2        # number of stacked RNN / LSTM layers
DROPOUT      = 0.3      # dropout probability (regularisation)
BATCH_SIZE   = 64       # mini-batch size
LEARNING_RATE = 1e-3    # Adam optimizer learning rate
NUM_EPOCHS   = 50       # training epochs per model
VAL_SPLIT    = 0.1      # fraction of names held out for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on device: {DEVICE}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one full pass over the training DataLoader.

    At each batch:
      1. Forward pass: compute logits from input sequences
      2. Reshape logits to (B*T, V) and targets to (B*T,) for cross-entropy
      3. Ignore PAD positions by masking them out (ignore_index=0 in criterion)
      4. Backpropagate and clip gradients to prevent exploding gradients
      5. Update parameters with Adam

    Returns average per-token training loss.
    """
    model.train()
    total_loss, n_batches = 0.0, 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)    # (B, T)
        targets = targets.to(device)   # (B, T)

        optimizer.zero_grad()
        logits, _ = model(inputs)      # (B, T, vocab_size)

        # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))

        loss.backward()
        # Gradient clipping: prevents exploding gradients common in RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on the validation set (no gradient computation).
    Returns average per-token validation loss.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        logits, _ = model(inputs)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def plot_losses(train_losses, val_losses, model_name):
    """Save a training / validation loss curve as a PNG file."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{model_name} — Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"analysis/{model_name.lower().replace(' ', '_')}_loss_curve.png", dpi=150)
    plt.close()
    print(f"  Saved loss curve: analysis/{model_name.lower().replace(' ', '_')}_loss_curve.png")


def train_model(model, model_name, checkpoint_path, train_loader, val_loader):
    """
    Full training loop for a single model.

    Uses:
      - Adam optimizer (adaptive learning rate, robust default for NLP)
      - Cross-entropy loss with PAD token ignored (index 0)
      - ReduceLROnPlateau scheduler: halves the LR when val loss stagnates
      - Early stopping patience of 10 epochs
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"{'='*60}")

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Reduce LR by factor 0.5 if val_loss does not improve for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ignore_index=0 means PAD tokens do not contribute to the loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss  = float("inf")
    patience_count = 0
    PATIENCE       = 10       # stop if val loss does not improve for 10 epochs
    train_losses, val_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss   = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"best={best_val_loss:.4f}")

        # Early stopping
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    plot_losses(train_losses, val_losses, model_name)
    print(f"  Best val loss: {best_val_loss:.4f}  |  checkpoint saved: {checkpoint_path}")
    return best_val_loss


def main():
    # ── Load and split data ────────────────────────────────────────────────────
    names = load_names(NAMES_FILE)
    print(f"Loaded {len(names)} names from '{NAMES_FILE}'")

    random.shuffle(names)
    split = int(len(names) * (1 - VAL_SPLIT))
    train_names = names[:split]
    val_names   = names[split:]
    print(f"Train: {len(train_names)} names  |  Val: {len(val_names)} names")

    # ── Build character vocabulary from TRAINING names only ────────────────────
    vocab = CharVocab(train_names)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size} characters")

    # Persist vocabulary for use in evaluation / generation scripts
    torch.save(vocab, "models/vocab.pt")
    print("Vocabulary saved to models/vocab.pt")

    # ── Create DataLoaders ─────────────────────────────────────────────────────
    train_loader = get_dataloader(train_names, vocab, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = get_dataloader(val_names,   vocab, batch_size=BATCH_SIZE, shuffle=False)

    # ── Hyperparameter summary ─────────────────────────────────────────────────
    hparams = {
        "embed_dim":    EMBED_DIM,
        "hidden_size":  HIDDEN_SIZE,
        "num_layers":   NUM_LAYERS,
        "dropout":      DROPOUT,
        "batch_size":   BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs":   NUM_EPOCHS,
        "val_split":    VAL_SPLIT,
        "device":       str(DEVICE),
    }

    # ── Define models ─────────────────────────────────────────────────────────
    models_cfg = [
        (
            VanillaRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT),
            "Vanilla RNN",
            "models/vanilla_rnn.pt"
        ),
        (
            BidirectionalLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT),
            "Bidirectional LSTM",
            "models/blstm.pt"
        ),
        (
            RNNWithAttention(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT),
            "RNN with Attention",
            "models/rnn_attention.pt"
        ),
    ]

    # ── Train each model and record results ───────────────────────────────────
    summary_lines = ["Model Training Summary", "=" * 50]
    summary_lines.append(f"Hyperparameters: {hparams}\n")

    for model, name, ckpt in models_cfg:
        best_val = train_model(model, name, ckpt, train_loader, val_loader)
        summary_lines.append(
            f"{name}:\n"
            f"  Parameters : {model.count_parameters():,}\n"
            f"  Best Val Loss : {best_val:.4f}\n"
            f"  Checkpoint : {ckpt}\n"
        )

    # Write summary to disk
    with open("analysis/training_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("\nTraining complete. Summary saved to analysis/training_summary.txt")


if __name__ == "__main__":
    main()
