import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from utils.dataset import load_names, CharVocab
from src.models import VanillaRNN, BidirectionalLSTM, RNNWithAttention

import utils.dataset as dataset_shim
sys.modules['dataset'] = dataset_shim

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
NAMES_FILE      = "data/TrainingNames.txt"
N_GENERATE      = 500           # number of names to generate per model for evaluation
MAX_LEN         = 15            # maximum characters per generated name
TEMPERATURE     = 0.8           # sampling temperature (< 1 → sharper, > 1 → flatter)
EMBED_DIM       = 64
HIDDEN_SIZE     = 256
NUM_LAYERS      = 2
DROPOUT         = 0.0           # set to 0.0 at inference time (no stochastic dropout)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_class, checkpoint_path, vocab_size):
    """
    Instantiate a model class, load its saved weights, and put it in eval mode.
    dropout=0.0 at inference — we only want stochasticity from sampling, not dropout.
    """
    model = model_class(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, dropout=0.0)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def generate_names(model, vocab, n: int, max_len: int, temperature: float) -> list[str]:
    """
    Generate `n` names from a model using temperature-scaled sampling.

    Empty strings (model immediately emitted <EOS>) are discarded and
    regenerated so we always return exactly `n` names.
    """
    names = []
    while len(names) < n:
        name = model.generate(vocab, max_len=max_len,
                              temperature=temperature, device=DEVICE)
        if name.strip():          # skip empty outputs
            names.append(name)
    return names


def compute_novelty(generated: list[str], training_set: set[str]) -> float:
    """
    Novelty Rate: fraction of generated names NOT in the training set.
    Comparison is case-insensitive.
    """
    novel = sum(1 for n in generated if n.lower() not in training_set)
    return novel / len(generated) * 100


def compute_diversity(generated: list[str]) -> float:
    """
    Diversity: fraction of unique names among all generated names.
    """
    return len(set(n.lower() for n in generated)) / len(generated)


def print_table(results: list[dict]):
    """Pretty-print evaluation results as an ASCII table."""
    print(f"\n{'Model':<25} {'Novelty Rate':>14} {'Diversity':>12} {'Unique / Total':>16}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<25} {r['novelty']:>13.1f}%  {r['diversity']:>11.3f}  "
              f"{r['unique']:>6} / {r['total']:<6}")
    print("-" * 70)


def plot_metrics(results: list[dict]):
    """Bar chart comparing Novelty Rate and Diversity across models."""
    model_names = [r["model"] for r in results]
    novelties   = [r["novelty"]   for r in results]
    diversities = [r["diversity"] * 100 for r in results]  # scale to % for same axis

    x = range(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar([i - width/2 for i in x], novelties,   width, label="Novelty Rate (%)")
    bars2 = ax.bar([i + width/2 for i in x], diversities, width, label="Diversity (%)")

    # Annotate bars with their numeric value
    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Model Comparison — Novelty Rate & Diversity")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.savefig("analysis/evaluation_metrics.png", dpi=150)
    plt.close()
    print("\nSaved bar chart: analysis/evaluation_metrics.png")


def main():
    # ── Load vocabulary and training names ────────────────────────────────────
    vocab: CharVocab = torch.load("models/vocab.pt", map_location="cpu", weights_only=False)
    vocab_size = len(vocab)

    training_names = load_names(NAMES_FILE)
    # Lower-case set for fast O(1) membership testing
    training_set = {n.lower() for n in training_names}
    print(f"Training set size: {len(training_set)} names  |  Vocab size: {vocab_size}")

    # ── Define model registry ─────────────────────────────────────────────────
    model_registry = [
        ("Vanilla RNN",        VanillaRNN,         "models/vanilla_rnn.pt"),
        ("Bidirectional LSTM", BidirectionalLSTM,  "models/blstm.pt"),
        ("RNN + Attention",    RNNWithAttention,   "models/rnn_attention.pt"),
    ]

    results = []

    for model_name, model_class, ckpt in model_registry:
        print(f"\n[{model_name}] Loading checkpoint: {ckpt}")
        model = load_model(model_class, ckpt, vocab_size)

        print(f"  Generating {N_GENERATE} names (temperature={TEMPERATURE})...")
        generated = generate_names(model, vocab, N_GENERATE, MAX_LEN, TEMPERATURE)

        novelty   = compute_novelty(generated, training_set)
        diversity = compute_diversity(generated)

        results.append({
            "model":     model_name,
            "novelty":   novelty,
            "diversity": diversity,
            "unique":    len(set(n.lower() for n in generated)),
            "total":     len(generated),
        })

        print(f"  Novelty Rate : {novelty:.1f}%")
        print(f"  Diversity    : {diversity:.3f}")

        # Save generated names for qualitative analysis
        out_file = f"analysis/generated_{model_name.lower().replace(' ', '_')}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(generated))
        print(f"  Saved generated names: {out_file}")

    # ── Print comparison table and plot ───────────────────────────────────────
    print_table(results)
    plot_metrics(results)


if __name__ == "__main__":
    main()
