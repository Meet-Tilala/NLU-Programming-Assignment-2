import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from src.dataset import build_vocab, CBOWDataset, SkipGramDataset, SkipGramNSDataset
from src.cbow_model import CBOWModel
from src.skipgram_model import SkipGramModel, SkipGramNSModel
from src.train import train_model
from src.visualise import visualize_embeddings
from utils.similarity import print_similar_words
from utils.analogies import analogy, interactive_analogy
from utils.save_load import save_model, load_model


# ═════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════

CORPUS_PATH = "data/clean_corpus.txt"
MODEL_DIR   = "models"
OUTPUT_DIR  = "outputs"

# Hyperparameters — experiment with these for your report
EMBED_DIM   = 300        # Embedding dimension (saved models use 300)
WINDOW_SIZE = 5          # Context window size
NUM_NEG     = 5          # Number of negative samples (for skipgram_ns)
BATCH_SIZE  = 512
EPOCHS      = 20
LR          = 0.001
MIN_FREQ    = 2          # Minimum word frequency for vocabulary

# Model paths
CBOW_PATH       = os.path.join(MODEL_DIR, "cbow.pth")
SKIPGRAM_PATH   = os.path.join(MODEL_DIR, "skipgram.pth")
SKIPGRAM_NS_PATH = os.path.join(MODEL_DIR, "skipgram_ns.pth")

# Words for similarity analysis (Task 3)
QUERY_WORDS = ["research", "student", "phd", "exam"]


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

if __name__ == "__main__":

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Build vocabulary (only needed for training) ──
    # print("\n" + "=" * 50)
    # print("  BUILDING VOCABULARY")
    # print("=" * 50)
    # sentences, word2idx, idx2word, word_freq = build_vocab(CORPUS_PATH, min_freq=MIN_FREQ)
    # VOCAB_SIZE = len(word2idx)
    # print(f"  Total sentences: {len(sentences):,}")
    # print(f"  Vocab size     : {VOCAB_SIZE:,}")



    # │  OPTION A: TRAIN A NEW MODEL             │


    # ---------- CBOW ----------
    # MODEL_PATH = CBOW_PATH
    # print("\n  Creating CBOW dataset...")
    # dataset = CBOWDataset(sentences, word2idx, window_size=WINDOW_SIZE)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # model = CBOWModel(VOCAB_SIZE, EMBED_DIM)
    # print("Training new model...")
    # train_model(model, dataloader, epochs=EPOCHS, lr=LR,
    #             save_path=MODEL_PATH, model_type="cbow",
    #             word2idx=word2idx, idx2word=idx2word)

    # ---------- SKIP-GRAM ----------
    # MODEL_PATH = SKIPGRAM_PATH
    # print("\n  Creating Skip-gram dataset...")
    # dataset = SkipGramDataset(sentences, word2idx, window_size=WINDOW_SIZE)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # model = SkipGramModel(VOCAB_SIZE, EMBED_DIM)
    # print("Training new model...")
    # train_model(model, dataloader, epochs=EPOCHS, lr=LR,
    #             save_path=MODEL_PATH, model_type="skipgram",
    #             word2idx=word2idx, idx2word=idx2word)

    # ---------- SKIP-GRAM + NEGATIVE SAMPLING ----------
    # MODEL_PATH = SKIPGRAM_NS_PATH
    # print("\n  Creating Skip-gram NS dataset...")
    # dataset = SkipGramNSDataset(sentences, word2idx, word_freq,
    #                             window_size=WINDOW_SIZE, num_neg=NUM_NEG)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # model = SkipGramNSModel(VOCAB_SIZE, EMBED_DIM)
    # print("Training new model...")
    # train_model(model, dataloader, epochs=15, lr=LR,
    #             save_path=MODEL_PATH, model_type="skipgram_ns",
    #             word2idx=word2idx, idx2word=idx2word)


    # │  OPTION B: LOAD A SAVED MODEL            │


    print("\n" + "=" * 50)
    print("  LOADING SAVED MODEL")
    print("=" * 50)

    # For CBOW:
    model, word2idx, idx2word = load_model(CBOWModel, CBOW_PATH, EMBED_DIM)

    # For Skip-gram:
    # model, word2idx, idx2word = load_model(SkipGramModel, SKIPGRAM_PATH, EMBED_DIM)

    # For Skip-gram + NS:
    #model, word2idx, idx2word = load_model(SkipGramNSModel, SKIPGRAM_NS_PATH, EMBED_DIM)

    embeddings = model.get_embeddings()

    # ── Step 3: Similarity Analysis (Task 3.1) ──

    print("\n" + "=" * 50)
    print("  TOP-5 NEAREST NEIGHBORS")
    print("=" * 50)

    for word in QUERY_WORDS:
        print_similar_words(word, embeddings, word2idx, idx2word, k=5)

    # ── Step 4: Analogy Task (Task 3.2) ──────────

    print("\n" + "=" * 50)
    print("  ANALOGY TASK")
    print("=" * 50)

    # Pre-defined analogies
    analogies_list = [
        ("ug", "btech", "pg"),
        ("student", "exam", "professor"),
        ("mtech", "engineering", "msc"),
    ]

    for w1, w2, w3 in analogies_list:
        result = analogy(w1, w2, w3, embeddings, word2idx, idx2word)
        if result:
            print(f"  {w1} : {w2} :: {w3} : {result}")

    # Interactive analogy
    # interactive_analogy(embeddings, word2idx, idx2word)

    # ── Step 5: Embedding Visualization (Task 4) ─

    print("\n" + "=" * 50)
    print("  GENERATING EMBEDDING PLOT")
    print("=" * 50)

    plot_path = os.path.join(OUTPUT_DIR, "embedding_plot.png")
    visualize_embeddings(embeddings, word2idx, idx2word,
                         save_path=plot_path, method="pca")

    print("\n" + "=" * 50)
    print("  ALL DONE!")
    print("=" * 50 + "\n")
