import os
import pickle
import torch


def save_model(model, word2idx, idx2word, model_path):

    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else "models", exist_ok=True)
    torch.save(model.state_dict(), model_path)

    vocab_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "models"
    with open(os.path.join(vocab_dir, "word2idx.pkl"), "wb") as f:
        pickle.dump(word2idx, f)
    with open(os.path.join(vocab_dir, "idx2word.pkl"), "wb") as f:
        pickle.dump(idx2word, f)


def load_model(model_class, model_path, embed_dim, vocab_size=None, **kwargs):
    vocab_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "models"
    with open(os.path.join(vocab_dir, "word2idx.pkl"), "rb") as f:
        word2idx = pickle.load(f)
    with open(os.path.join(vocab_dir, "idx2word.pkl"), "rb") as f:
        idx2word = pickle.load(f)

    if vocab_size is None:
        vocab_size = len(word2idx)

    model = model_class(vocab_size, embed_dim, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    print(f"   Model loaded from {model_path}")
    print(f"     Vocab size: {len(word2idx):,}  |  Embed dim: {embed_dim}")
    return model, word2idx, idx2word

