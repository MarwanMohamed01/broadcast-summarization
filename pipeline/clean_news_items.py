"""
LLM post-processing step: clean up OCR-extracted news items.

Reads the raw v6 output (news_items.json) and uses an LLM (Gemini 2.5 Flash)
to:
  1. Split any items that contain multiple headlines merged together
  2. Complete incomplete headlines using cross-item context
  3. Fix OCR typos (e.g., "HORMU" -> "HORMUZ", "KilLed" -> "killed")
  4. Remove fragments and garbage items
  5. Deduplicate across the list

Writes the cleaned result to news_items_cleaned.json (separate file — does
NOT overwrite the raw v6 output, so both are available for evaluation).

Usage:
    python clean_news_items.py
"""

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import requests
from rapidfuzz import fuzz

# Load API keys from llm_summarization/.env
PROJECT_DIR = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_DIR / "llm_summarization" / ".env")

RAW_INPUT_PATH = (
    PROJECT_DIR / "ticker_extraction_v6" / "output" / "final" / "news_items.json"
)
CLEANED_OUTPUT_PATH = (
    PROJECT_DIR / "ticker_extraction_v6" / "output" / "final" / "news_items_cleaned.json"
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


SYSTEM_PROMPT = (
    "You are a news headline cleaner. You receive a list of raw news "
    "headlines extracted from a TV news ticker via OCR. The OCR has "
    "introduced errors: some headlines are merged together, some are "
    "truncated, and words contain typos.\n\n"
    "Your task:\n"
    "1. SPLIT any item that contains two or more distinct headlines "
    "into separate items. Distinct headlines have different subjects "
    "(different countries, people, or topics).\n"
    "2. COMPLETE any truncated headlines. If one item in the list is "
    "a partial version of a headline you can see in another item, "
    "output the complete version. Use cross-item context.\n"
    "3. FIX obvious OCR typos in proper nouns and common words "
    "(e.g., 'HORMU' -> 'HORMUZ', 'KilLed' -> 'KILLED'). Keep the "
    "original wording otherwise — do NOT paraphrase or add "
    "information that isn't in the text.\n"
    "4. DROP fragments shorter than 30 characters that don't contain "
    "a complete news statement.\n"
    "5. DROP exact duplicates and near-duplicates.\n\n"
    "Output ONLY a JSON array of strings. Each string is one cleaned "
    "headline in ALL CAPS. No explanation, no markdown fences."
)


def call_gemini(user_prompt: str) -> str:
    """Call Gemini 2.5 Flash."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in llm_summarization/.env")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    return next(p["text"] for p in reversed(parts) if not p.get("thought"))


def call_groq(user_prompt: str) -> str:
    """Call Groq Llama 3.3 70B."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in llm_summarization/.env")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def extract_json_array(text: str) -> list[str]:
    """Extract a JSON array of strings from an LLM response."""
    # Remove markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    # Find the array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if isinstance(x, (str,)) and str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Fallback: line-based parsing
    lines = []
    for line in text.split("\n"):
        line = line.strip().lstrip("-").strip().strip(",").strip('"').strip("'")
        if len(line) > 20:
            lines.append(line)
    return lines


def deduplicate(items: list[str]) -> list[str]:
    """Fuzzy dedup on the cleaned output."""
    unique = []
    for item in items:
        dup = False
        for existing in unique:
            if fuzz.ratio(item.lower(), existing.lower()) > 70:
                dup = True
                break
            if fuzz.partial_ratio(item.lower(), existing.lower()) > 85:
                ratio = min(len(item), len(existing)) / max(len(item), len(existing))
                if ratio > 0.5:
                    dup = True
                    break
        if not dup:
            unique.append(item)
    return unique


def clean_batch(items: list[str], batch_idx: int, total_batches: int) -> list[str]:
    """Send one batch of raw items to the LLM and return cleaned items."""
    numbered = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
    user_prompt = (
        f"Below are {len(items)} raw OCR-extracted news headlines from a "
        f"TV news ticker (batch {batch_idx+1}/{total_batches}). Clean them "
        f"up following the instructions. Output a JSON array of strings.\n\n"
        f"RAW HEADLINES:\n{numbered}"
    )

    # Try Gemini first, fall back to Groq
    for attempt_name, fn in [("Gemini 2.5 Flash", call_gemini), ("Groq Llama 3.3 70B", call_groq)]:
        try:
            print(f"  Batch {batch_idx+1}/{total_batches}: calling {attempt_name}...")
            cleaned_text = fn(user_prompt)
            parsed = extract_json_array(cleaned_text)
            print(f"    Got {len(parsed)} cleaned items")
            return parsed
        except Exception as e:
            print(f"    {attempt_name} failed: {e}")
    return []


def main():
    print("=" * 60)
    print("  LLM-powered news item cleaner")
    print("=" * 60)

    if not RAW_INPUT_PATH.exists():
        print(f"ERROR: raw input not found: {RAW_INPUT_PATH}")
        return

    raw_items = json.loads(RAW_INPUT_PATH.read_text(encoding="utf-8"))
    print(f"  Raw items loaded: {len(raw_items)}\n")

    # Process in batches of 15 items so the LLM has full context within each
    # batch but can't run out of output tokens.
    BATCH_SIZE = 15
    raw_texts = [it["text"] for it in raw_items]

    all_cleaned = []
    batches = [raw_texts[i:i + BATCH_SIZE] for i in range(0, len(raw_texts), BATCH_SIZE)]

    for bi, batch in enumerate(batches):
        cleaned = clean_batch(batch, bi, len(batches))
        all_cleaned.extend(cleaned)

    print(f"\n  Total cleaned items from all batches: {len(all_cleaned)}")
    cleaned_items = all_cleaned
    provider_used = "Gemini 2.5 Flash / Groq fallback"

    # Deduplicate (in case LLM left near-duplicates)
    cleaned_items = deduplicate(cleaned_items)
    print(f"  After final dedup: {len(cleaned_items)}")

    # Filter out items that are too short
    cleaned_items = [c for c in cleaned_items if len(c) >= 30]
    print(f"  After length filter: {len(cleaned_items)}")

    # Build output
    results = [{"id": i, "text": t} for i, t in enumerate(cleaned_items, 1)]

    CLEANED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLEANED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(results)} cleaned items -> {CLEANED_OUTPUT_PATH}")
    print(f"  Provider used: {provider_used}")
    print()
    print("=" * 60)
    print(f"  RAW: {len(raw_items)} items  |  CLEANED: {len(results)} items")
    print("=" * 60)
    for item in results:
        print(f"  [{item['id']:3d}] {item['text']}")


if __name__ == "__main__":
    main()
