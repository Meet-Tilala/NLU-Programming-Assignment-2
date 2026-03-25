import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import anthropic

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
OUTPUT_FILE  = "data/TrainingNames.txt"
TOTAL_NAMES  = 1000          # target unique names
BATCH_SIZE   = 100           # names requested per API call
MODEL        = "claude-sonnet-4-20250514"
MAX_TOKENS   = 1024          # enough for a JSON list of 100 names


def generate_batch(client: anthropic.Anthropic, n: int) -> list[str]:

    prompt = (
        f"Generate exactly {n} unique Indian first names. "
        "Include a diverse mix of male and female names from different regions: "
        "North India (Hindi/Sanskrit), South India (Tamil, Telugu, Kannada, Malayalam), "
        "Bengal, Maharashtra, Gujarat, and Punjab/Sikh traditions. "
        "Return ONLY a raw JSON array of strings — no explanation, no markdown, "
        "no numbering. Example format: [\"Arjun\", \"Priya\", \"Venkatesh\"]"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract text from the first content block
    raw_text = response.content[0].text.strip()

    # Defensively strip any accidental markdown fences the model may add
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    # Parse JSON list; raises json.JSONDecodeError if model misbehaved
    names: list[str] = json.loads(raw_text)
    return [name.strip() for name in names if isinstance(name, str) and name.strip()]


def main():
    # Initialise client — reads ANTHROPIC_API_KEY from environment automatically
    client = anthropic.Anthropic()

    all_names: list[str] = []

    batch_num = 0
    # Keep requesting batches until we have at least TOTAL_NAMES unique names
    while len(set(n.lower() for n in all_names)) < TOTAL_NAMES:
        n_unique_so_far = len(set(n.lower() for n in all_names))
        remaining = TOTAL_NAMES - n_unique_so_far
        # Request a bit more than needed to compensate for cross-batch duplicates
        request_n = min(BATCH_SIZE, remaining + 10)
        batch_num += 1

        print(f"[Batch {batch_num}] Requesting {request_n} names "
              f"(have {n_unique_so_far} unique so far)...")

        try:
            batch = generate_batch(client, request_n)
            all_names.extend(batch)
            print(f"  -> Received {len(batch)} names from API")
        except json.JSONDecodeError as exc:
            print(f"  [WARNING] JSON parse error in batch {batch_num}: {exc}. Retrying...")
        except anthropic.APIError as exc:
            print(f"  [ERROR] API error in batch {batch_num}: {exc}. Stopping.")
            break

    # ── Deduplicate: preserve first-seen ordering, case-insensitive comparison ──
    seen:   set[str]  = set()
    unique: list[str] = []
    for name in all_names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)

    unique = unique[:TOTAL_NAMES]   # trim to exactly TOTAL_NAMES

    # ── Write to disk (one name per line) ──────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(unique))

    print(f"\n[OK] Saved {len(unique)} unique names to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
