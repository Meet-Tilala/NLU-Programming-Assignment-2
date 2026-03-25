"""
visualise.py
------------
Visualize word embeddings using PCA or t-SNE.
Projects selected word groups into 2D and saves cluster plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Word groups for clustering visualization
WORD_GROUPS = {
    "Academic Programs": ["btech", "mtech", "phd", "msc", "dual", "degree",
                          "undergraduate", "postgraduate", "programme"],
    "Departments":       ["engineering", "science", "humanities", "mathematics",
                          "physics", "chemistry", "computer", "electrical",
                          "mechanical", "civil"],
    "Research":          ["research", "thesis", "project", "publication",
                          "supervisor", "seminar", "journal", "paper"],
    "Evaluation":        ["exam", "grade", "cgpa", "sgpa", "evaluation",
                          "semester", "credits", "marks"],
}

COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]


def visualize_embeddings(embeddings, word2idx, idx2word, save_path="outputs/embedding_plot.png",
                         method="pca"):
    words = []
    vecs = []
    labels = []
    colors = []

    for group_idx, (group_name, word_list) in enumerate(WORD_GROUPS.items()):
        for word in word_list:
            if word in word2idx:
                words.append(word)
                vecs.append(embeddings[word2idx[word]])
                labels.append(group_name)
                colors.append(COLORS[group_idx % len(COLORS)])

    if len(vecs) == 0:
        print("   No matching words found in vocabulary for visualization.")
        return

    vecs = np.array(vecs)

    # Dimensionality reduction
    if method == "tsne":
        perplexity = min(30, len(vecs) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        title_method = "t-SNE"
    else:
        reducer = PCA(n_components=2)
        title_method = "PCA"

    coords = reducer.fit_transform(vecs)

    # Plot
    plt.figure(figsize=(14, 10))
    seen_labels = set()
    for i, (x, y) in enumerate(coords):
        label = labels[i] if labels[i] not in seen_labels else None
        if label:
            seen_labels.add(labels[i])
        plt.scatter(x, y, c=colors[i], s=100, label=label, edgecolors="white", linewidth=0.5)
        plt.annotate(words[i], (x, y), fontsize=9, ha="center", va="bottom",
                     textcoords="offset points", xytext=(0, 6))

    plt.title(f"Word Embedding Clusters ({title_method})", fontsize=16)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(fontsize=11, loc="best", framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Plot saved at {save_path}")
