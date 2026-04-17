"""
Improve v6's raw news_items.json by re-processing the per-chunk OCR text.

This script:
  1. Reads every chunk's full_ocr_text.txt (from v6's output/chunks/)
  2. Splits each chunk into ticker cycles at "FOR MORE GO TO ALJAZEERA"
  3. For each cycle, splits on " - " delimiter to get headline candidates
  4. Collects items from ALL cycles of ALL chunks (not just the "best" one)
  5. Re-splits items that still contain " - " internally
  6. Quality-filters and deduplicates, keeping the cleanest version of each
     headline across the entire 14-hour video
  7. Writes the result to v6's output/final/news_items.json

This improves coverage (more unique headlines captured) and cleanliness
(cleaner versions picked) before the optional LLM correction step.
"""

import json
import re
from pathlib import Path
from rapidfuzz import fuzz

# ── Paths ────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent.resolve()
CHUNKS_DIR = PROJECT_DIR / "ticker_extraction_v6" / "output" / "chunks"
OUTPUT_PATH = PROJECT_DIR / "ticker_extraction_v6" / "output" / "final" / "news_items.json"

MIN_ITEM_LEN = 35
MAX_ITEM_LEN = 200


# ── Text cleaning ────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize OCR noise."""
    for ch in "|[]{}\\~`^":
        text = text.replace(ch, "")
    text = text.replace("\ufffd", "'").replace("�", "'")
    text = re.sub(r"[—–]", "-", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_prefixes_suffixes(item: str) -> str:
    """Remove FOR MORE GO TO..., BREAKING NEWS, and similar noise."""
    item = re.sub(r"(?i)\s*FOR\s+MORE\s+GO\s+TO\s+ALJAZEERA[\.\w]*\s*", " ", item)
    item = re.sub(r"(?i)^BREAKING\s+NEWS\s*-?\s*", "", item)
    item = re.sub(r"\s+", " ", item).strip()
    # Strip leading/trailing dashes
    item = item.strip("- ").strip()
    return item


# ── Quality scoring ──────────────────────────────────────

def quality_score(text: str) -> float:
    """
    Score text quality 0-1.
    Strongly prefers complete single headlines (50-130 chars) over
    merged items (>150 chars = likely two headlines glued together)
    or fragments (<40 chars).
    """
    if len(text) < MIN_ITEM_LEN:
        return 0.0

    words = text.split()
    if len(words) < 3:
        return 0.0

    good_words = 0
    total_scored = 0
    for w in words:
        letters = re.sub(r"[^a-zA-Z]", "", w)
        if len(letters) == 0:
            continue
        total_scored += 1
        is_good = True
        # glued words too long
        if len(letters) > 16:
            is_good = False
        # consonant soup
        if is_good and len(letters) >= 5:
            vowels = sum(1 for c in letters if c.lower() in "aeiou")
            if vowels == 0:
                is_good = False
        # case switching
        if is_good and len(letters) >= 4:
            switches = sum(
                1 for a, b in zip(letters, letters[1:])
                if a.isupper() != b.isupper()
            )
            if switches > 2 and switches >= len(letters) * 0.4:
                is_good = False
        # Repeated char runs (like "CONGCONGO")
        if is_good and len(letters) >= 6:
            for j in range(len(letters) - 4):
                chunk = letters[j:j + 5].lower()
                if len(set(chunk)) <= 2:
                    is_good = False
                    break
        if is_good:
            good_words += 1

    word_q = good_words / max(total_scored, 1)

    # Penalize ending with tiny fragment word
    last_letters = re.sub(r"[^a-zA-Z]", "", words[-1])
    ending_penalty = 0.1 if len(last_letters) <= 2 else 0

    # Length score: strongly prefer 50-130 char range (single complete headline)
    length = len(text)
    if length < 40:
        len_q = 0.3
    elif length < 50:
        len_q = 0.6
    elif length <= 130:
        len_q = 1.0  # ideal single-headline range
    elif length <= 150:
        len_q = 0.7
    elif length <= 180:
        len_q = 0.4  # likely merged
    else:
        len_q = 0.2  # definitely merged

    return max(0, word_q * 0.5 + len_q * 0.5 - ending_penalty)


def is_garbage(text: str) -> bool:
    if len(text) < MIN_ITEM_LEN:
        return True
    if len(text) > MAX_ITEM_LEN:
        return True
    if quality_score(text) < 0.50:
        return True
    return False


def looks_like_merged(text: str) -> bool:
    """Length-based check: items > 140 chars are likely merged headlines."""
    return len(text) > 140


# ── Splitting ────────────────────────────────────────────

def split_on_internal_delimiters(items: list[str]) -> list[str]:
    """
    Re-split any item that contains ' - ' internally.
    Catches items where the first-pass split missed a delimiter.
    """
    result = []
    for item in items:
        # Split on any dash with spaces around it
        parts = re.split(r"\s+-\s+", item)
        for p in parts:
            p = p.strip()
            if len(p) >= MIN_ITEM_LEN:
                result.append(p)
    return result


def extract_items_from_chunk(ocr_text: str) -> list[str]:
    """
    Extract all headline candidates from a chunk's OCR text.
    Splits into cycles, then each cycle on " - ", collecting all items.
    """
    cleaned = clean_text(ocr_text)

    # Split into ticker cycles
    cycles = re.split(
        r"FOR\s+MORE\s+GO\s+TO\s+ALJAZEERA[\.\w]*",
        cleaned,
        flags=re.IGNORECASE,
    )
    cycles = [c.strip() for c in cycles if len(c.strip()) > 100]

    all_items = []
    for cycle in cycles:
        # Split cycle on " - "
        parts = re.split(r"\s+-\s+", cycle)
        for part in parts:
            part = strip_prefixes_suffixes(part)
            if len(part) >= MIN_ITEM_LEN:
                all_items.append(part)

    # Post-split: catch any items still containing " - "
    all_items = split_on_internal_delimiters(all_items)

    return all_items


# ── Dedup ────────────────────────────────────────────────

def extract_numbers(text: str) -> set:
    """Significant numbers (excluding years)."""
    result = set()
    for n in re.findall(r"\d[\d,]+", text):
        clean = n.replace(",", "")
        if len(clean) >= 3:
            val = int(clean)
            if not (1900 <= val <= 2099):
                result.add(val)
    return result


def is_duplicate(a: str, b: str) -> bool:
    """Aggressive fuzzy duplicate check.

    Uses multiple strategies:
      1. Direct similarity on full text
      2. Digit-stripped comparison (OCR corrupts numbers)
      3. Prefix comparison (first 5 words) — catches variants of same headline
    """
    al, bl = a.lower(), b.lower()

    # Strategy 1: direct
    if fuzz.ratio(al, bl) > 50:
        return True
    if fuzz.token_sort_ratio(al, bl) > 55:
        return True
    partial = fuzz.partial_ratio(al, bl)
    len_ratio = min(len(a), len(b)) / max(len(a), len(b))
    if partial > 70 and len_ratio > 0.35:
        return True

    # Strategy 2: digit-stripped (handles OCR number corruption)
    a_nd = re.sub(r"[\d,.]", " ", al)
    a_nd = re.sub(r"\s+", " ", a_nd).strip()
    b_nd = re.sub(r"[\d,.]", " ", bl)
    b_nd = re.sub(r"\s+", " ", b_nd).strip()
    if len(a_nd) > 15 and len(b_nd) > 15:
        if fuzz.ratio(a_nd, b_nd) > 55:
            return True
        if fuzz.token_sort_ratio(a_nd, b_nd) > 60:
            return True
        if fuzz.partial_ratio(a_nd, b_nd) > 75:
            nd_ratio = min(len(a_nd), len(b_nd)) / max(len(a_nd), len(b_nd))
            if nd_ratio > 0.4:
                return True

    # Strategy 3: prefix match (first 5-6 words)
    a_words = al.split()[:6]
    b_words = bl.split()[:6]
    if len(a_words) >= 4 and len(b_words) >= 4:
        a_prefix = " ".join(a_words)
        b_prefix = " ".join(b_words)
        if fuzz.ratio(a_prefix, b_prefix) > 65:
            return True

    return False


def deduplicate_keeping_best(items: list[str]) -> list[str]:
    """
    Deduplicate, keeping the version with highest quality score.
    Tiebreaker: prefer items closest to ideal length (~90 chars).
    """
    def ideal_length_dist(text):
        # 0 at 90 chars, 1 at 0 or 180 chars
        return abs(len(text) - 90) / 90

    scored = [(quality_score(it), it) for it in items]
    # Sort: high quality first, then closest to ideal length
    scored.sort(key=lambda x: (-x[0], ideal_length_dist(x[1])))

    unique = []
    for q, item in scored:
        dup_idx = -1
        for j, (eq, et) in enumerate(unique):
            if is_duplicate(item, et):
                dup_idx = j
                break
        if dup_idx >= 0:
            eq, et = unique[dup_idx]
            # Replace only if clearly better quality OR clearly closer to ideal
            if q > eq + 0.05:
                unique[dup_idx] = (q, item)
            elif abs(q - eq) < 0.05:
                if ideal_length_dist(item) < ideal_length_dist(et) - 0.1:
                    unique[dup_idx] = (q, item)
        else:
            unique.append((q, item))

    return [text for q, text in unique]


# ── Main ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Improving raw v6 news_items.json")
    print("=" * 60)

    if not CHUNKS_DIR.exists():
        print(f"ERROR: chunks directory not found: {CHUNKS_DIR}")
        return

    # Step 1: extract items from every chunk's OCR text
    all_good_items = []  # items 40-140 chars (single complete headlines)
    all_merged_items = []  # items >140 chars (likely merged — kept as fallback)
    total_chunks = 0

    for chunk_dir in sorted(CHUNKS_DIR.glob("chunk_*")):
        ocr_file = chunk_dir / "ocr" / "full_ocr_text.txt"
        if not ocr_file.exists():
            continue
        try:
            text = ocr_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  {chunk_dir.name}: READ ERROR {e}")
            continue
        if not text.strip():
            continue

        chunk_idx = int(chunk_dir.name.split("_")[1])
        items = extract_items_from_chunk(text)
        items = [it for it in items if not is_garbage(it)]

        # Separate into "good" (complete single headlines) and "merged"
        good = [it for it in items if 40 <= len(it) <= 140 and not looks_like_merged(it)]
        merged = [it for it in items if it not in good]

        all_good_items.extend(good)
        all_merged_items.extend(merged)
        total_chunks += 1
        print(f"  chunk_{chunk_idx:03d}: {len(good)} good + {len(merged)} merged items")

    if total_chunks == 0:
        print("ERROR: no chunks found")
        return

    print(f"\n  Total good items: {len(all_good_items)}")
    print(f"  Total merged items: {len(all_merged_items)}")

    # Step 2: dedup the GOOD items first (these are the clean single headlines)
    good_unique = deduplicate_keeping_best(all_good_items)
    print(f"  Good items after dedup: {len(good_unique)}")

    # Step 3: for each merged item, check if its content is already covered
    # by the good items. If not, keep it (it might contain unique headlines
    # that only appeared in merged form).
    uncovered_merged = []
    for m in all_merged_items:
        # Check if any 50+ char substring of m is already in a good item
        is_covered = False
        for good in good_unique:
            if fuzz.partial_ratio(good.lower(), m.lower()) > 80:
                is_covered = True
                break
        if not is_covered:
            uncovered_merged.append(m)

    # Dedup uncovered merged items among themselves
    uncovered_merged = deduplicate_keeping_best(uncovered_merged)
    print(f"  Merged items not covered by good items: {len(uncovered_merged)}")

    # Combine: good items first, then uncovered merged items (LLM will split them)
    merged = good_unique + uncovered_merged
    print(f"  Combined total: {len(merged)}")

    # Sort by quality score descending for readability
    merged.sort(key=lambda x: quality_score(x), reverse=True)

    # Build output
    results = [{"id": i, "text": t} for i, t in enumerate(merged, 1)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(results)} items -> {OUTPUT_PATH}")
    print()
    print("=" * 60)
    print(f"  FINAL: {len(results)} unique news items")
    print("=" * 60)
    for item in results:
        print(f"  [{item['id']:3d}] {item['text']}")


if __name__ == "__main__":
    main()
