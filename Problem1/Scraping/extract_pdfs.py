

import os
import re
import pdfplumber
from langdetect import detect, LangDetectException


# ─────────────────────────────────────────────
# CONFIGURATION — edit these paths before running
# ─────────────────────────────────────────────

# Folder where all your downloaded PDFs are kept
PDF_FOLDER = "./pdfs"

# Output file where all extracted text will be saved (one doc per line block)
OUTPUT_FILE = "pdf_extracted.txt"

# Minimum number of characters a page must have to be considered non-empty
MIN_PAGE_CHARS = 50


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def is_english(text):

    try:
        return detect(text) == "en"
    except LangDetectException:
        # langdetect fails on very short or symbol-only strings — skip them
        return False


def clean_page_text(text):
    if not text:
        return ""

    # Step 1: Replace common unicode ligatures with ASCII equivalents
    # e.g., 'ﬁ' → 'fi', 'ﬂ' → 'fl' (common in PDFs generated from LaTeX)
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")

    # Remove soft hyphens (used for word-wrap in PDFs, not real hyphens)
    text = text.replace("\u00ad", "")

    # Step 2: Remove isolated page numbers — lines that contain ONLY digits
    # e.g., a line that just says "23" or " 4 " is a page number artifact
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Step 3: Remove header/footer boilerplate patterns common in IIT docs
    # These are lines like "IIT Jodhpur | Academic Regulations 2023"
    text = re.sub(r"(?i)iit\s*jodhpur[^\n]*", "", text)

    # Step 4: Collapse 3+ consecutive newlines into just two (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 5: Remove lines that are purely special characters or symbols
    # e.g., lines like "─────────" or "* * * * *"
    text = re.sub(r"^[\s\W]+$", "", text, flags=re.MULTILINE)

    # Step 6: Strip leading/trailing whitespace from entire text block
    return text.strip()


def extract_text_from_pdf(pdf_path):
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"  → Processing: {os.path.basename(pdf_path)} ({len(pdf.pages)} pages)")

        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract raw text from the page
            raw_text = page.extract_text()

            # Skip pages with no extractable text (image-only pages)
            if not raw_text or len(raw_text) < MIN_PAGE_CHARS:
                print(f"     Skipping page {page_num} (too short or empty)")
                continue

            # Clean the raw text
            cleaned = clean_page_text(raw_text)

            # Filter: only keep this page if it's predominantly English
            if not is_english(cleaned):
                print(f"     Skipping page {page_num} (non-English content detected)")
                continue

            all_text.append(cleaned)

    # Join all pages with a double newline (paragraph separator)
    return "\n\n".join(all_text)


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Collect all PDF files from the PDF_FOLDER
    pdf_files = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in '{PDF_FOLDER}'. Please check your folder path.")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s). Starting extraction...\n")

    all_documents = []  # Will hold extracted text from each PDF separately

    for pdf_path in sorted(pdf_files):
        extracted = extract_text_from_pdf(pdf_path)

        if extracted:
            # Tag each document with its source filename (useful for tracing)
            tagged = f"### SOURCE: {os.path.basename(pdf_path)} ###\n{extracted}"
            all_documents.append(tagged)
        else:
            print(f"   No usable text extracted from {os.path.basename(pdf_path)}")

    # Write all extracted text to the output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_documents))

    print(f"\n Extraction complete!")
    print(f"   Total PDFs processed : {len(pdf_files)}")
    print(f"   Total documents saved : {len(all_documents)}")
    print(f"   Output saved to       : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
