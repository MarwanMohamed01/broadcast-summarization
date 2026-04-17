"""
Step 5: Segment the continuous OCR text into individual news items.

Strategy:
  1. Auto-detect delimiter (" - ", " | ", etc.) in the OCR text
  2. Split text at delimiters
  3. Quality-filter each item (reject OCR garbage)
  4. Deduplicate with fuzzy matching (keeping cleanest version)
  5. Remove merge artifacts (items glued from two different headlines)
"""

import json
import re
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
import config


def _clean_text(text: str) -> str:
    """Remove common OCR artifacts and normalize whitespace."""
    noise_chars = "|[]{}\\~`^"
    for ch in noise_chars:
        text = text.replace(ch, "")

    text = text.replace("\ufffd", "'")
    text = text.replace("�", "'")

    text = re.sub(r"[—–]", "-", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _find_delimiter(text: str) -> tuple[str, int]:
    """Auto-detect the most common delimiter in the text.
    Returns (delimiter, count)."""
    counts = {}
    for delim in config.NEWS_DELIMITERS:
        counts[delim] = text.count(delim)

    if counts:
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            print(f"  Detected delimiter: '{best}' (found {counts[best]} times)")
            return best, counts[best]

    print("  No delimiter detected")
    return " - ", 0


def _quality_score(text: str) -> float:
    """Score text quality from 0-1."""
    if len(text) < 15:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    alpha = sum(1 for c in text if c.isalpha())
    upper = sum(1 for c in text if c.isupper())
    if alpha == 0:
        return 0.0

    upper_ratio = upper / alpha
    real_words = sum(1 for w in words if len(w) >= 3 and
                     sum(c.isalpha() for c in w) / len(w) > 0.7)
    real_word_ratio = real_words / len(words) if words else 0

    length_score = 1.0
    if len(text) < 30:
        length_score = 0.5
    elif len(text) > 200:
        length_score = 0.6

    return (upper_ratio * 0.4 + real_word_ratio * 0.4 + length_score * 0.2)


def _is_garbage(text: str) -> bool:
    """Check if text is mostly OCR garbage."""
    if _quality_score(text) < config.QUALITY_THRESHOLD:
        return True
    if len(text) > config.MAX_NEWS_LENGTH:
        return True
    return False


def _extract_significant_numbers(text: str) -> set:
    raw_numbers = re.findall(r'\d[\d,]+', text)
    result = set()
    for n in raw_numbers:
        clean = n.replace(',', '')
        if len(clean) >= 3:
            val = int(clean)
            if 1900 <= val <= 2099:
                continue
            result.add(val)
    return result


def _is_duplicate(item: str, existing: str) -> bool:
    """Number-aware fuzzy duplicate check."""
    item_lower = item.lower()
    existing_lower = existing.lower()

    full_sim = fuzz.ratio(item_lower, existing_lower)
    token_sim = fuzz.token_sort_ratio(item_lower, existing_lower)
    partial_sim = fuzz.partial_ratio(item_lower, existing_lower)

    is_match = False
    if full_sim > config.DEDUP_SIMILARITY_THRESHOLD:
        is_match = True
    elif token_sim > config.DEDUP_SIMILARITY_THRESHOLD:
        is_match = True
    else:
        len_ratio = min(len(item), len(existing)) / max(len(item), len(existing))
        if (partial_sim > config.DEDUP_PARTIAL_THRESHOLD and len_ratio > 0.5):
            is_match = True

    if not is_match:
        return False

    nums_item = _extract_significant_numbers(item)
    nums_existing = _extract_significant_numbers(existing)
    if nums_item and nums_existing:
        if nums_item.isdisjoint(nums_existing):
            return False

    return True


def _remove_merge_artifacts(unique_items: list[str]) -> list[str]:
    """Remove items that are two headlines glued together."""
    to_remove = set()
    for i, item in enumerate(unique_items):
        if len(item) < 30:
            continue
        words = item.split()
        if len(words) < 4:
            continue
        mid = len(words) // 2
        first_half = " ".join(words[:mid])
        second_half = " ".join(words[mid:])
        first_matches = []
        second_matches = []
        for j, other in enumerate(unique_items):
            if i == j:
                continue
            other_start = " ".join(other.split()[:mid])
            if fuzz.ratio(first_half.lower(), other_start.lower()) > 65:
                first_matches.append(j)
            other_end = " ".join(other.split()[-mid:])
            if fuzz.ratio(second_half.lower(), other_end.lower()) > 65:
                second_matches.append(j)
        if first_matches and second_matches:
            first_set = set(first_matches)
            second_set = set(second_matches)
            if first_set != second_set and not first_set.issubset(second_set):
                to_remove.add(i)
                print(f"    MERGE ARTIFACT: '{item[:70]}...'")
    if to_remove:
        return [item for i, item in enumerate(unique_items) if i not in to_remove]
    return unique_items


def segment_news(full_text: str) -> tuple[list[dict], dict]:
    """
    Split continuous text into individual news items.
    Returns (list of dicts with id and text, stats dict).
    """
    config.FINAL_DIR.mkdir(parents=True, exist_ok=True)

    if not full_text.strip():
        print("No text to segment.")
        return [], {}

    cleaned = _clean_text(full_text)
    delimiter, delim_count = _find_delimiter(cleaned)

    if delim_count >= 3:
        # Delimiter found — split text into ticker cycles at "FOR MORE GO TO
        # ALJAZEERA" markers, then pick the BEST cycle (cleanest, most complete)
        # and split that cycle on the delimiter.
        cycles = re.split(
            r"FOR\s+MORE\s+GO\s+TO\s+ALJAZEERA[\.\w]*",
            cleaned,
            flags=re.IGNORECASE,
        )
        cycles = [c.strip() for c in cycles if len(c.strip()) > 200]

        if len(cycles) >= 2:
            # Pick the best cycle: prefer middle cycles (not first/last which
            # are usually partial), and prefer cycles with 15-35 delimiter splits
            # (roughly one per headline — too many means false splits from noise)
            scored_cycles = []
            for ci, cycle in enumerate(cycles):
                n_splits = cycle.count(delimiter)
                # Ideal: 15-30 splits per cycle (one per headline)
                # Too many splits = false dashes from OCR noise
                split_score = 1.0
                if n_splits < 10:
                    split_score = n_splits / 10
                elif n_splits > 35:
                    split_score = 35 / n_splits  # heavily penalize noisy cycles
                # Penalize first and last cycles (usually partial)
                position_score = 1.0
                if ci == 0 or ci == len(cycles) - 1:
                    position_score = 0.3
                # Prefer cycles with moderate length (1500-3000 chars = one full cycle)
                len_score = 1.0
                if len(cycle) < 500:
                    len_score = 0.2
                elif len(cycle) > 4000:
                    len_score = 3000 / len(cycle)  # penalize too-long (noisy)
                score = split_score * position_score * len_score
                scored_cycles.append((score, ci, cycle))

            scored_cycles.sort(reverse=True)
            best_score, best_ci, best_cycle = scored_cycles[0]
            print(f"  Best cycle: #{best_ci} ({len(best_cycle)} chars, "
                  f"{best_cycle.count(delimiter)} delimiters)")

            # Split the best cycle on delimiter, then also process
            # ALL cycles to find any headlines missing from the best one
            raw_items = best_cycle.split(delimiter)

            # Add items from other good cycles that aren't in the best
            for score, ci, cycle in scored_cycles[1:3]:  # top 3 cycles
                other_items = cycle.split(delimiter)
                raw_items.extend(other_items)
        else:
            # Not enough cycle markers — just split the whole text
            raw_items = cleaned.split(delimiter)
    else:
        # No delimiters — treat entire text as one item
        print("  WARNING: Too few delimiters found. The OCR engine may not be")
        print("  detecting the ' - ' separator. Try --engine tesseract")
        raw_items = [cleaned]

    # Clean each item and filter
    items = []
    filtered_out = []
    for item in raw_items:
        item = item.strip()
        # Remove "FOR MORE GO TO..." trailer
        item = re.sub(r"(?i)\s*FOR\s+MORE\s+GO\s+TO\s+ALJAZEERA[\.\w]*\s*", "", item).strip()
        # Remove "BREAKING NEWS" prefix
        item = re.sub(r"(?i)^BREAKING\s+NEWS\s*", "", item).strip()
        if len(item) < config.MIN_NEWS_LENGTH:
            filtered_out.append(("too_short", item))
        elif _is_garbage(item):
            filtered_out.append(("garbage", item))
        else:
            items.append(item)

    print(f"  Raw items after split: {len(raw_items)}")
    print(f"  Items after filtering: {len(items)}")
    print(f"  Filtered out: {len(filtered_out)}")
    for reason, txt in filtered_out:
        print(f"    [{reason}] {txt[:80]}...")

    # Sort by quality score descending
    items.sort(key=lambda x: _quality_score(x), reverse=True)

    # Deduplicate with smart version selection
    unique_items = []
    dedup_log = []
    for item in items:
        is_dup = False
        for j, existing in enumerate(unique_items):
            if _is_duplicate(item, existing):
                if (len(item) > len(existing)
                        and len(item) < 130
                        and _quality_score(item) > 0.85):
                    dedup_log.append(
                        f"    REPLACE: '{existing[:50]}...' WITH '{item[:50]}...' (longer, good quality)")
                    unique_items[j] = item
                else:
                    dedup_log.append(f"    DROP: '{item[:60]}...' (dup of '{existing[:60]}...')")
                is_dup = True
                break
        if not is_dup:
            unique_items.append(item)

    print(f"  Unique items after deduplication: {len(unique_items)}")
    if dedup_log:
        print(f"  Dedup decisions ({len(dedup_log)}):")
        for log in dedup_log:
            print(log)

    dedup_drops = sum(1 for log in dedup_log if "DROP:" in log)
    dedup_replaces = sum(1 for log in dedup_log if "REPLACE:" in log)
    items_before_merge_check = len(unique_items)

    print("\n  Checking for merge artifacts...")
    unique_items = _remove_merge_artifacts(unique_items)
    merge_artifacts_removed = items_before_merge_check - len(unique_items)
    print(f"  Items after merge artifact removal: {len(unique_items)}")

    final_quality_scores = [_quality_score(item) for item in unique_items]
    final_lengths = [len(item) for item in unique_items]

    results = []
    for i, text in enumerate(unique_items, 1):
        results.append({"id": i, "text": text})

    output_path = config.FINAL_DIR / "news_items.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(results)} news items to {output_path}")

    filter_reasons = {}
    for reason, _ in filtered_out:
        filter_reasons[reason] = filter_reasons.get(reason, 0) + 1

    estimated_cycles = round(len(raw_items) / max(len(unique_items), 1), 1)

    stats = {
        "segmentation": {
            "delimiter_detected": delimiter,
            "delimiter_count": delim_count,
            "raw_items_after_split": len(raw_items),
            "items_after_quality_filter": len(items),
            "filtered_out_total": len(filtered_out),
            "filtered_out_by_reason": filter_reasons,
            "items_passed_filter": len(items),
            "duplicates_dropped": dedup_drops,
            "duplicates_replaced_with_better": dedup_replaces,
            "merge_artifacts_removed": merge_artifacts_removed,
            "final_unique_news_items": len(results),
            "estimated_ticker_cycles": estimated_cycles,
            "final_items_quality": {
                "avg_quality_score": round(float(np.mean(final_quality_scores)), 3) if final_quality_scores else 0,
                "min_quality_score": round(float(np.min(final_quality_scores)), 3) if final_quality_scores else 0,
                "max_quality_score": round(float(np.max(final_quality_scores)), 3) if final_quality_scores else 0,
            },
            "final_items_length": {
                "avg_chars": round(float(np.mean(final_lengths)), 1) if final_lengths else 0,
                "min_chars": min(final_lengths) if final_lengths else 0,
                "max_chars": max(final_lengths) if final_lengths else 0,
            },
        },
    }

    return results, stats


if __name__ == "__main__":
    ocr_file = config.OCR_DIR / "full_ocr_text.txt"
    if ocr_file.exists():
        text = ocr_file.read_text(encoding="utf-8")
        items, stats = segment_news(text)
        print(json.dumps(stats, indent=2))
        for item in items[:5]:
            print(f"  [{item['id']}] {item['text'][:80]}...")
    else:
        print("No OCR text found. Run step4 first.")
