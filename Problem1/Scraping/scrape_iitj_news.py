"""
scrape_iitj_express.py
-----------------------
Scrapes ONLY IIT Jodhpur relevant articles from Indian Express.
- Saves each article immediately after fetching (Ctrl+C safe)
- Skips any article that doesn't mention IIT Jodhpur

Install: pip install requests beautifulsoup4
"""

import time
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LISTING_URL   = "https://indianexpress.com/about/iit-jodhpur/"
OUTPUT_FILE   = "iitj_corpus.txt"
REQUEST_DELAY = 1.0
TIMEOUT       = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; IITJCorpusBot/1.0)"
}

# Article must contain at least one of these to be saved
RELEVANCE_KEYWORDS = ["iit jodhpur", "iitj", "indian institute of technology jodhpur"]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_article_links(listing_url):
    """Fetch the listing page and return all /article/ links found on it."""
    print(f"Fetching listing page: {listing_url}\n")
    try:
        resp = requests.get(listing_url, headers=HEADERS, timeout=TIMEOUT)
    except Exception as e:
        print(f"✗ Could not fetch listing page: {e}")
        return []

    soup  = BeautifulSoup(resp.text, "html.parser")
    links, seen = [], set()

    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(listing_url, anchor["href"].strip())
        if "/article/" in absolute and absolute not in seen:
            seen.add(absolute)
            links.append(absolute)

    print(f"Found {len(links)} article links on listing page.\n")
    return links


def is_relevant(text):
    """Return True only if the article text mentions IIT Jodhpur."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in RELEVANCE_KEYWORDS)


def extract_article_text(url):
    """Fetch one article page and return its body text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return ""

    if resp.status_code != 200:
        print(f"  ✗ HTTP {resp.status_code}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove all non-content tags
    for tag in soup(["script", "style", "nav", "footer",
                     "header", "form", "button", "aside", "figure"]):
        tag.decompose()

    # Try known Indian Express article body containers
    content = (
        soup.find("div", class_="story-details")   or
        soup.find("div", class_="full-details")    or
        soup.find("div", class_="ie-network-desc") or
        soup.find("article")                        or
        soup.find("main")
    )

    if not content:
        print(f"  ✗ Article body not found")
        return ""

    return content.get_text(separator="\n").strip()


def save_article(url, text):
    """
    Append a single article to the output file immediately.
    Called right after extracting — so Ctrl+C never loses saved data.
    """
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"### SOURCE: {url} ###\n")
        f.write(text)
        f.write("\n\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    article_links = get_article_links(LISTING_URL)
    if not article_links:
        print("No articles found. Exiting.")
        exit()

    saved_count   = 0
    skipped_count = 0

    for i, url in enumerate(article_links, start=1):
        print(f"[{i}/{len(article_links)}] {url}")

        text = extract_article_text(url)

        if not text or len(text) < 100:
            print(f"  ✗ Skipped (too short or empty)")
            skipped_count += 1

        elif not is_relevant(text):
            # Article exists but doesn't mention IIT Jodhpur at all — skip it
            print(f"  ✗ Skipped (not IIT Jodhpur related)")
            skipped_count += 1

        else:
            # ✅ Relevant — save immediately to file
            save_article(url, text)
            saved_count += 1
            print(f"  ✓ Saved ({len(text.split())} words)")

        time.sleep(REQUEST_DELAY)

    print(f"\n{'='*50}")
    print(f"  Done!")
    print(f"  Saved   : {saved_count} articles")
    print(f"  Skipped : {skipped_count} articles")
    print(f"  Output  : {OUTPUT_FILE}")
    print(f"{'='*50}")
