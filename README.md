<h1 align="center">
  NLU Programming Assignment 2
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NLTK-Language_Processing-154F5B?style=for-the-badge" alt="NLTK">
</p>

<p align="center">
  <b>A comprehensive Natural Language Understanding (NLU) exploration covering Word Embeddings (CBOW/Skip-gram) and Generative Character-Level Language Models (RNN/LSTM/Attention).</b>
</p>

---

##  Table of Contents
- [About the Project](#-about-the-project)
- [Project Architecture](#-project-architecture)
- [Problem 1: Word Embeddings (Word2Vec)](#-problem-1-word-embeddings-word2vec)
- [Problem 2: Character-Level Name Generation](#-problem-2-character-level-name-generation)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Results & Artifacts](#-results--artifacts)

---

##  About the Project
This repository contains solutions for the **Natural Language Understanding (NLU) Programming Assignment 2**. The assignment is divided into two distinct machine learning challenges:
1. **Domain-Specific Word Embeddings:** Scraping content from the IIT Jodhpur domain, tokenizing a corpus, and training continuous bag-of-words (CBOW) and Skip-gram architectures from scratch in PyTorch.
2. **Generative Language Modeling:** Synthesizing novel, culturally accurate Indian names utilizing character-level Vanilla RNNs, Bidirectional LSTMs, and Attention-augmented RNNs.

---

##  Project Architecture
The codebase is structured modularly for maintainability and clean execution.

```text
PA2/
├── Problem1/                     # Word2Vec and Embeddings
│   ├── main.py                   # Master script to orchestrate execution
│   ├── Preprocessing/            # Text tokenization and cleaning heuristics
│   ├── Scraping/                 # BeautifulSoup scripts extracting data from IITJ
│   ├── src/                      # Core neural models (CBOW, Skipgram, Trainer)
│   ├── utils/                    # Similarity and Analogy evaluation scripts
│   └── Analysis/                 # Output directories for embeddings and WordClouds
│
├── Problem2/                     # Character-level Language Models
│   ├── src/
│   │   ├── train.py              # PyTorch training loop for 3 RNN variants
│   │   ├── evaluate.py           # Quantitative validation (Diversity & Novelty)
│   │   └── models.py             # VanillaRNN, BLSTM, and RNNWithAttention classes
│   ├── utils/
│   │   ├── generate_names.py     # Anthropic API dataset generator
│   │   ├── dataset.py            # DataLoader and Character Vocabulary build
│   │   └── qualitative.py        # Temperature sampling and failure mode analysis
│   ├── data/                     # Holds the 1000-sample name dataset
│   ├── models/                   # Saved .pt PyTorch checkpoints
│   └── analysis/                 # Loss curves, summaries, and generated text
│
├── requirements.txt              # Standardized dependency list
└── .gitignore                    # Prevents datasets and heavy .pt models from polluting Git
```

---

##  Problem 1: Word Embeddings (Word2Vec)
The first portion of this assignment builds a domain-specific Word2Vec variant.

### Key Features:
- **Web Scraping:** `scrape_iitj.py` and `extract_pdfs.py` scrape HTML text and PDF documents recursively from the IIT Jodhpur domain.
- **Corpus Preprocessing:** Cleans whitespace, standardizes casing, purges stop-words, and manages out-of-vocabulary artifacts.
- **Model Training:** Implements two PyTorch models:
  1. **CBOW (Continuous Bag of Words)**
  2. **Skip-gram with Negative Sampling**
- **Evaluation:** Evaluates semantic validity using analogy completion (`V(king) - V(man) + V(woman) = V(queen)`) and Cosine Similarity computations.
- **Visualisation:** Uses t-SNE/PCA to project 64D embeddings into 2D scatter plots and utilizes `WordCloud` visualizations of the dominant corpus themes.

---

##  Problem 2: Character-Level Name Generation
The second portion pivots from word-level distributed representation to character-level generative modeling to synthesize Indian Names.

### Key Features:
- **Curated Dataset:** Sourced 1000 unique, culturally diverse Indian Names to form a baseline training set.
- **Architectures Built from Scratch:**
  - `VanillaRNN`: Baseline character generation tracking context sequentially.
  - `BidirectionalLSTM`: Tracks forward and backward sequences to circumvent vanishing gradients.
  - `RNN + Attention`: Implements an attention mechanism, allowing dynamic character focus.
- **Quantitative Metrics:** Measures structural generalisation via **Novelty Rate** (producing outputs absent in training data) and **Diversity Rate** (unique tokens per run).
- **Qualitative Sampler:** Samples the models at various **Temperatures (T=0.5, T=0.8, T=1.2)** to analyze hallucination properties, realism, and failure modes (e.g., repeating loops, truncation).

---

##  Setup & Installation
Ensure you possess Python 3.10+ and a machine with a CUDA-capable GPU (optional, but highly recommended).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Meet-Tilala/NLU-Programming-Assignment-2.git
   cd NLU-Programming-Assignment-2
   ```

2. **Establish a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

##  How to Run

### Executing Problem 1
Orchestrated entirely through a central `main.py` entrypoint.
```bash
cd Problem1
python main.py
```
> Outputs, charts, and visualizations are dispatched into `Problem1/Analysis/`.

### Executing Problem 2
Problem 2 utilizes decoupled scripts due to its extensive evaluation logic. The scripts automatically resolve relative paths.

**1. Train the Models:**
```bash
cd Problem2
python src/train.py
```

**2. Evaluate Quantitatively (Novelty & Diversity Metrics):**
```bash
cd Problem2
python src/evaluate.py
```

**3. Evaluate Qualitatively (Temperature Mapping):**
```bash
cd Problem2
python utils/qualitative.py
```
> Evaluated reports and loss curves are dispatched into `Problem2/analysis/`.

---

##  Results & Artifacts
The repositories automatically produce a diverse range of artifacts for the evaluation phase. Due to GitHub file constraints, actual `*.pt` weights are ignored natively utilizing the `.gitignore` setup. Instead, analytical documents are produced:

- `Problem1/Analysis/corpus_stats.txt`
- `Problem1/Analysis/Embedding plots/*.png`
- `Problem2/analysis/evaluation_metrics.png`
- `Problem2/analysis/training_summary.txt`
- `Problem2/analysis/qualitative_report.txt`

---

*Authored for the NLU curriculum.*
