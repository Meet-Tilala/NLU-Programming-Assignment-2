"""
scrape_iitj.py
--------------
Recursive BFS web scraper for IIT Jodhpur corpus collection.
- Follows all internal links deeply from each seed
- Extracts only English text
- Skips binary/media files
- Saves raw text (you clean it separately)

Install: pip install requests beautifulsoup4 langdetect
"""

import time
import requests
from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException


# ─────────────────────────────────────────────────────────────────
# SEED DICTIONARY — paste your starting links here
# Key   = a label for your own reference (not used in code)
# Value = the starting URL for that crawl
# ─────────────────────────────────────────────────────────────────

SEEDS = {
    "iitj_main"        : "https://www.iitj.ac.in/",
    "iitj_cse"         : "https://www.iitj.ac.in/computer-science-engineering/en/",
    "iitj_cse_research": "https://www.iitj.ac.in/computer-science-engineering/en/Research-Archive",
    "iitj_academics"   : "https://www.iitj.ac.in/academic/",
    "iitj_research"    : "https://www.iitj.ac.in/research/",
    "iitj_people"      : "https://www.iitj.ac.in/people/",
    "iitj_techscape"   : "https://iitj.ac.in/techscape",
    "iitj_senate"      : "https://iitjsenateportal.vercel.app/",
    "iitj_express"     : "https://indianexpress.com/about/iit-jodhpur/",
    # ↓ Add more seeds below as needed
    # "label"          : "https://...",
}

# ─────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────

OUTPUT_FILE   = "iitj_corpus.txt"   # All scraped text appended here
MAX_PAGES     = 50                  # Per seed — increase for more data
REQUEST_DELAY = 1.0                 # Seconds between requests (be polite)
TIMEOUT       = 10                  # Seconds before giving up on a request

# File extensions to skip — no binary/media files
SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".tar", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".css", ".js", ".xml", ".json", ".rss", ".atom",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; IITJCorpusBot/1.0)"
}


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def get_base_domain(url):
    """Extract base domain from a URL — used to restrict crawl to same site."""
    parsed = urlparse(url)
    return parsed.netloc  # e.g., 'www.iitj.ac.in'


def is_valid_url(url, allowed_domain):
    """
    Return True only if the URL:
    - Is http/https
    - Belongs to the same domain as the seed (no external links)
    - Does not point to a binary/media file
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    # Stay within the seed's domain
    if parsed.netloc != allowed_domain:
        return False

    # Skip binary and media file extensions
    path = parsed.path.lower().split("?")[0]  # ignore query params for ext check
    if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False

    return True


def normalize(url):
    """Strip fragment (#section) and trailing slash for consistent deduplication."""
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl().rstrip("/")


def is_english(text):
    """Return True if langdetect thinks the text is English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def extract_text(soup):
    """
    Pull raw visible text from a parsed page.
    Removes script/style/nav/footer tags first.
    Returns a single string — no cleaning done here (you handle that separately).
    """
    # Remove tags that never contain useful content
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "button"]):
        tag.decompose()

    # Get all visible text, separated by newlines
    return soup.get_text(separator="\n")


def get_links(soup, current_url, allowed_domain):
    """
    Find all valid internal links on a page.
    Converts relative URLs to absolute and filters with is_valid_url().
    """
    links = set()
    for anchor in soup.find_all("a", href=True):
        absolute = normalize(urljoin(current_url, anchor["href"].strip()))
        if is_valid_url(absolute, allowed_domain):
            links.add(absolute)
    return links


# ─────────────────────────────────────────────────────────────────
# CORE SCRAPER
# ─────────────────────────────────────────────────────────────────

def scrape(seed_url, max_pages=50):
    """
    BFS crawler starting from seed_url.
    Visits up to max_pages pages within the same domain.
    Returns all extracted text as a single string.
    """

    allowed_domain = get_base_domain(seed_url)
    queue          = deque([normalize(seed_url)])   # BFS queue
    visited        = {normalize(seed_url)}          # Already seen URLs
    collected_text = []                             # Text from each page

    print(f"\n── Crawling: {seed_url}")
    print(f"   Domain : {allowed_domain} | Limit : {max_pages} pages\n")

    session = requests.Session()  # Reuse TCP connection across requests

    while queue and len(collected_text) < max_pages:
        url = queue.popleft()
        print(f"  [{len(collected_text)+1}/{max_pages}] {url}")

        # ── Fetch ──────────────────────────────────────────
        try:
            resp = session.get(url, headers=HEADERS, timeout=TIMEOUT)
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

        if resp.status_code != 200:
            print(f"    ✗ HTTP {resp.status_code}")
            continue

        # Only parse HTML responses
        if "text/html" not in resp.headers.get("Content-Type", ""):
            print(f"    ✗ Not HTML, skipping")
            continue

        # ── Parse ──────────────────────────────────────────
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Extract text ───────────────────────────────────
        text = extract_text(soup)

        if len(text.strip()) < 100:
            print(f"    ✗ Too little text")
        elif not is_english(text):
            print(f"    ✗ Non-English, skipping")
        else:
            # Tag text with source URL so you know where it came from
            collected_text.append(f"### SOURCE: {url} ###\n{text.strip()}")
            print(f"    ✓ Saved ({len(text.split())} words)")

        # ── Queue new links ────────────────────────────────
        new_links = get_links(soup, url, allowed_domain)
        added = 0
        for link in new_links:
            if link not in visited:
                visited.add(link)
                queue.append(link)
                added += 1
        print(f"    → {len(new_links)} links found, {added} newly queued")

        time.sleep(REQUEST_DELAY)

    session.close()

    total_words = sum(len(t.split()) for t in collected_text)
    print(f"\n Done: {len(collected_text)} pages | {total_words:,} words\n")

    return "\n\n".join(collected_text)


# ─────────────────────────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    total_words = 0

    # Iterate over every seed in the dictionary
    for label, seed_url in SEEDS.items():
        print(f"\n{'='*60}")
        print(f"  SEED: {label}")
        print(f"{'='*60}")

        text = scrape(seed_url, max_pages=MAX_PAGES)

        if text.strip():
            # Append to output file (each seed's text added on top of previous)
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n{'#'*60}\n")
                f.write(f"# SEED: {label} | URL: {seed_url}\n")
                f.write(f"{'#'*60}\n\n")
                f.write(text)

            words = len(text.split())
            total_words += words
            print(f"  Appended {words:,} words to {OUTPUT_FILE}")

    print(f"\n{'='*60}")
    print(f"  ALL SEEDS DONE")
    print(f"  Total words collected : {total_words:,}")
    print(f"  Output file           : {OUTPUT_FILE}")
    print(f"{'='*60}\n")
