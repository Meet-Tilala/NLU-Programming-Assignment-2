import re
import os
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize

# Download required NLTK data (safe to re-run)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

INPUT_FILES = [
    "pdf_extracted.txt",
    "iitj_corpus.txt",
]

OUTPUT_CLEANED   = "cleaned_corpus.txt"
OUTPUT_TOKENIZED = "tokenized_corpus.txt"
OUTPUT_STATS     = "corpus_stats.txt"
OUTPUT_WORDCLOUD = "wordcloud.png"

STOP_WORDS = set(stopwords.words("english"))


# ─────────────────────────────────────────────
# STEP 1 — LOAD AND MERGE
# ─────────────────────────────────────────────

def load_and_merge(file_paths):
    """
    Read all input files and concatenate into one raw string.
    Missing files are warned about and skipped gracefully.
    """
    merged = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"  Warning: File not found, skipping: {path}")
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        merged.append(content)
        print(f"  Loaded: {path}  ({len(content):,} characters)")

    return "\n\n".join(merged)


# ─────────────────────────────────────────────
# STEP 2 — SPLIT INTO DOCUMENTS
# ─────────────────────────────────────────────

def split_into_documents(raw_text):
    docs = re.split(r"### SOURCE:.*?###", raw_text)
    docs = [d.strip() for d in docs if len(d.strip()) > 50]
    print(f"  Split into {len(docs)} raw documents")
    return docs


# ─────────────────────────────────────────────
# STEP 3 — CLEAN EACH DOCUMENT
# ─────────────────────────────────────────────

def clean_document(text):

    # ── (i) Boilerplate removal ────────────────────────────

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove any leftover HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove page number patterns like "Page 3 of 12" or "- 4 -"
    text = re.sub(r"page\s*\d+\s*(of\s*\d+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"-\s*\d+\s*-", "", text)

    # Remove separator lines (e.g., "-------", "=======", "* * *")
    text = re.sub(r"^[\s\-_=*#|~]{3,}$", "", text, flags=re.MULTILINE)

    # Remove LaTeX commands like \textbf{} \section{}
    text = re.sub(r"\\[a-zA-Z]+\{?[^}]*\}?", "", text)

    # Remove non-ASCII characters (Hindi, emojis, symbols, etc.)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # ── (iii) Lowercase ────────────────────────────────────
    text = text.lower()

    # ── (iv) Remove excessive punctuation / non-textual content ──

    # Remove lines that are only numbers or single characters
    text = re.sub(r"^\s*[\d]+[\.\)]*\s*$", "", text, flags=re.MULTILINE)

    # Replace 2+ consecutive punctuation characters with a space
    text = re.sub(r"[^\w\s]{2,}", " ", text)

    # Keep only letters, digits, whitespace, and basic sentence punctuation
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\;\:\'\-]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Drop very short lines (stray characters left after cleaning)
    lines = [ln for ln in text.splitlines() if len(ln.strip()) >= 4]
    text = "\n".join(lines)

    return text.strip()


# ─────────────────────────────────────────────
# STEP 4 — TOKENIZE
# ─────────────────────────────────────────────

def tokenize_document(clean_text):
    tokens = word_tokenize(clean_text)
    tokens = [t for t in tokens if t.isalpha() and len(t) >= 2]
    return tokens


# ─────────────────────────────────────────────
# STEP 5 — STATS
# ─────────────────────────────────────────────

def compute_and_save_stats(cleaned_docs, all_tokens, path):
    vocab_size   = len(set(all_tokens))
    total_tokens = len(all_tokens)

    # Frequency counter excluding stopwords (for meaningful ranking)
    content_tokens = [t for t in all_tokens if t not in STOP_WORDS]
    freq    = Counter(content_tokens)
    top_20  = freq.most_common(20)

    # Print to console
    print(f"\n  Total documents  : {len(cleaned_docs):,}")
    print(f"  Total tokens     : {total_tokens:,}")
    print(f"  Vocabulary size  : {vocab_size:,}")
    print(f"\n  Top 10 words (stopwords excluded):")
    for word, count in top_20[:10]:
        print(f"    {word:<20} {count}")

    # Save to file
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("  CORPUS STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"  Total documents  : {len(cleaned_docs):,}\n")
        f.write(f"  Total tokens     : {total_tokens:,}\n")
        f.write(f"  Vocabulary size  : {vocab_size:,}\n\n")
        f.write("-" * 50 + "\n")
        f.write("  Top 20 Most Frequent Words (stopwords excluded)\n")
        f.write("-" * 50 + "\n")
        for rank, (word, count) in enumerate(top_20, start=1):
            f.write(f"  {rank:>2}. {word:<20} {count:>6}\n")
        f.write("\n" + "=" * 50 + "\n")

    print(f"\n  Stats saved to: {path}")
    return freq


# ─────────────────────────────────────────────
# STEP 6 — WORD CLOUD
# ─────────────────────────────────────────────

def generate_wordcloud(freq, path):
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=STOP_WORDS,
        max_words=150,
        colormap="viridis",
        collocations=False,
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("IIT Jodhpur Corpus — Most Frequent Words", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"  Word cloud saved to: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*50)
    print("  CORPUS PREPROCESSING PIPELINE")
    print("="*50 + "\n")

    # Step 1: Load and merge
    print("── Step 1: Loading and merging files")
    raw_text = load_and_merge(INPUT_FILES)

    # Step 2: Split into documents
    print("\n── Step 2: Splitting into documents")
    documents = split_into_documents(raw_text)

    # Step 3 + 4: Clean and tokenize
    print("\n── Step 3 & 4: Cleaning and tokenizing")
    cleaned_docs   = []
    tokenized_docs = []
    all_tokens     = []

    for doc in documents:
        cleaned = clean_document(doc)
        tokens  = tokenize_document(cleaned)

        # Skip documents that are too short after cleaning
        if len(tokens) < 10:
            continue

        cleaned_docs.append(cleaned)
        tokenized_docs.append(tokens)
        all_tokens.extend(tokens)

    print(f"  Documents after cleaning : {len(cleaned_docs):,}")
    print(f"  Total tokens             : {len(all_tokens):,}")

    # Step 4: Save cleaned corpus (human-readable)
    print("\n── Step 5: Saving outputs")
    with open(OUTPUT_CLEANED, "w", encoding="utf-8") as f:
        f.write("\n\n".join(cleaned_docs))
    print(f"  Saved: {OUTPUT_CLEANED}")

    # Save tokenized corpus — one document per line, space-separated tokens
    # This is the direct input format for Word2Vec training (Task 2)
    with open(OUTPUT_TOKENIZED, "w", encoding="utf-8") as f:
        for tokens in tokenized_docs:
            f.write(" ".join(tokens) + "\n")
    print(f"  Saved: {OUTPUT_TOKENIZED}")

    # Step 5: Compute and save stats
    print("\n── Step 6: Computing corpus statistics")
    freq = compute_and_save_stats(cleaned_docs, all_tokens, OUTPUT_STATS)

    # Step 6: Word cloud
    print("\n── Step 7: Generating word cloud")
    generate_wordcloud(freq, OUTPUT_WORDCLOUD)

    print("\n" + "="*50)
    print("  PREPROCESSING COMPLETE")
    print("="*50)
    print(f"  cleaned_corpus.txt   → cleaned text")
    print(f"  tokenized_corpus.txt → Word2Vec input (Task 2)")
    print(f"  corpus_stats.txt     → paste into report")
    print(f"  wordcloud.png        → paste into report")
    print("="*50 + "\n")
