"""
analogies.py
------------
Word analogy task using vector arithmetic:
    w1 : w2 :: w3 : ?   →   ? = w2 - w1 + w3
"""

import numpy as np
from utils.similarity import cosine_similarity


def analogy(w1, w2, w3, embeddings, word2idx, idx2word):
    for w in [w1, w2, w3]:
        if w not in word2idx:
            print(f"  ⚠ '{w}' not in vocabulary.")
            return None

    vec = embeddings[word2idx[w2]] - embeddings[word2idx[w1]] + embeddings[word2idx[w3]]

    best_word = None
    best_sim = -1
    exclude = {w1, w2, w3}

    for idx in range(len(embeddings)):
        word = idx2word[idx]
        if word in exclude:
            continue
        sim = cosine_similarity(vec, embeddings[idx])
        if sim > best_sim:
            best_sim = sim
            best_word = word

    return best_word


def interactive_analogy(embeddings, word2idx, idx2word):
    print("\n" + "=" * 40)
    print("  ANALOGY TASK")
    print("=" * 40)
    print("  Format:  w1 : w2 :: w3 : ?")
    print("  Type 'quit' to exit.\n")

    while True:
        w1 = input("Enter word1 (e.g., ug): ").strip().lower()
        if w1 == "quit":
            break
        w2 = input("Enter word2 (e.g., btech): ").strip().lower()
        if w2 == "quit":
            break
        w3 = input("Enter word3 (e.g., pg): ").strip().lower()
        if w3 == "quit":
            break

        result = analogy(w1, w2, w3, embeddings, word2idx, idx2word)
        if result:
            print(f"\nResult: {w1} : {w2} :: {w3} : {result}\n")
        else:
            print("\n  Could not compute analogy.\n")
