"""
similarity.py
-------------
Cosine similarity-based nearest neighbor search over word embeddings.
"""

import numpy as np


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return dot / norm if norm > 0 else 0.0


def get_top_k_similar(word, embeddings, word2idx, idx2word, k=5):
    if word not in word2idx:
        print(f"   '{word}' not in vocabulary.")
        return []

    word_vec = embeddings[word2idx[word]]
    similarities = []

    for idx in range(len(embeddings)):
        if idx2word[idx] == word:
            continue
        sim = cosine_similarity(word_vec, embeddings[idx])
        similarities.append((idx2word[idx], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def print_similar_words(word, embeddings, word2idx, idx2word, k=5):
    """Find and print top-k similar words in formatted output."""
    results = get_top_k_similar(word, embeddings, word2idx, idx2word, k)
    print(f"\nWord: {word}")
    for w, sim in results:
        print(f"{w} ({sim:.4f})")
    return results
