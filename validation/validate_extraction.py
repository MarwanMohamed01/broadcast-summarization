"""
Compute validation metrics for the ticker extraction pipeline.

Compares the extracted news items against manually-annotated ground
truth to produce the thesis-grade evaluation:

  - Precision / Recall / F1  (at fuzzy-match thresholds 60 / 70 / 80%)
  - Character Error Rate (CER) and Word Error Rate (WER) on matched pairs
  - Exact match count (100% fuzzy ratio)
  - Coverage (% of GT with any ≥50% partial match)
  - Missed headlines list (GT headlines not found)
  - False positives list (extracted items with no GT match)
  - Per-stage comparison: raw v6 output vs LLM-cleaned output

Ground truth files:
  validation/ground_truth/slice_A_headlines.txt
  validation/ground_truth/slice_B_headlines.txt

Stages evaluated:
  1. Raw v6 output — ticker_extraction_v6/output/final/news_items.json
  2. LLM-cleaned   — ticker_extraction_v6/output/final/news_items_cleaned.json

Output:
  results/validation_report.json  — full machine-readable report
  printed summary table            — for copy-paste into the thesis
"""

import json
import re
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

PROJECT_DIR = Path(__file__).parent.parent.resolve()
GT_DIR = PROJECT_DIR / "validation" / "ground_truth"
RAW_PATH = PROJECT_DIR / "ticker_extraction_v6" / "output" / "final" / "news_items.json"
CLEANED_PATH = PROJECT_DIR / "ticker_extraction_v6" / "output" / "final" / "news_items_cleaned.json"
RESULTS_DIR = PROJECT_DIR / "results"
REPORT_PATH = RESULTS_DIR / "validation_report.json"

THRESHOLDS = [60, 70, 80]


# ── Loading ─────────────────────────────────────────────

def load_ground_truth(path: Path) -> list[str]:
    """Load GT headlines, skipping comments and empty lines."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def load_extracted(path: Path) -> list[str]:
    """Load extracted news items from a JSON file."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item["text"].strip() for item in data if item.get("text", "").strip()]


# ── Metrics ─────────────────────────────────────────────

def normalize(text: str) -> str:
    """Normalize text for fuzzy comparison: uppercase, collapse whitespace."""
    text = text.upper()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    """Symmetric fuzzy similarity 0-100, tolerant to OCR noise and punctuation."""
    a_n = normalize(a)
    b_n = normalize(b)
    # Use the maximum of standard ratio and token-sort-ratio
    # (token-sort tolerates word reordering; ratio is a stricter edit distance)
    return max(fuzz.ratio(a_n, b_n), fuzz.token_sort_ratio(a_n, b_n))


def best_match(target: str, candidates: list[str]) -> tuple[int, float]:
    """Return (index, score) of the best-matching candidate."""
    best_idx = -1
    best_score = 0.0
    for i, c in enumerate(candidates):
        s = similarity(target, c)
        if s > best_score:
            best_score = s
            best_idx = i
    return best_idx, best_score


def character_error_rate(ref: str, hyp: str) -> float:
    """CER = edit distance / len(ref), character level."""
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return Levenshtein.distance(ref_n, hyp_n) / len(ref_n)


def word_error_rate(ref: str, hyp: str) -> float:
    """WER = edit distance / len(ref words), word level."""
    ref_words = normalize(ref).split()
    hyp_words = normalize(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)


# ── Core comparison ────────────────────────────────────

def evaluate_stage(extracted: list[str], gt: list[str], stage_name: str,
                   slice_name: str) -> dict:
    """
    Compute all metrics for one (stage, slice) pair.

    Uses the fuzzy-similarity scores to determine matches. For each GT
    headline, finds its best matching extracted item. For each extracted
    item, finds its best matching GT headline.

    A GT headline is "recalled" if there exists an extracted item with
    similarity >= threshold. An extracted item is "precise" if it
    matches any GT headline with similarity >= threshold.
    """
    # For each GT, find best matching extracted item
    gt_best_scores = []
    gt_best_matches = []  # (score, extracted_idx)
    for g in gt:
        idx, score = best_match(g, extracted)
        gt_best_scores.append(score)
        gt_best_matches.append((score, idx))

    # For each extracted item, find best matching GT
    ext_best_scores = []
    for e in extracted:
        _, score = best_match(e, gt)
        ext_best_scores.append(score)

    # Compute metrics at each threshold
    threshold_results = {}
    for thr in THRESHOLDS:
        tp_gt = sum(1 for s in gt_best_scores if s >= thr)
        tp_ext = sum(1 for s in ext_best_scores if s >= thr)
        fn = len(gt) - tp_gt
        fp = len(extracted) - tp_ext

        precision = tp_ext / len(extracted) if extracted else 0
        recall = tp_gt / len(gt) if gt else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        threshold_results[str(thr)] = {
            "tp_gt": tp_gt,
            "tp_ext": tp_ext,
            "fn": fn,
            "fp": fp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Exact matches (>= 95% similarity = effectively exact, tolerates minor punctuation)
    exact_match_count = sum(1 for s in gt_best_scores if s >= 95)
    coverage_50 = sum(1 for s in gt_best_scores if s >= 50)

    # CER/WER on pairs matched at 70% threshold (the standard one)
    cers = []
    wers = []
    matched_ext_indices = set()
    for (score, idx), g in zip(gt_best_matches, gt):
        if score >= 70 and idx >= 0:
            e = extracted[idx]
            cers.append(character_error_rate(g, e))
            wers.append(word_error_rate(g, e))
            matched_ext_indices.add(idx)

    avg_cer = sum(cers) / len(cers) if cers else None
    avg_wer = sum(wers) / len(wers) if wers else None

    # Missed GT headlines (no match at 70%)
    missed = [g for g, s in zip(gt, gt_best_scores) if s < 70]

    # False positives (extracted items with no match at 70%)
    false_positives = [e for e, s in zip(extracted, ext_best_scores) if s < 70]

    return {
        "stage": stage_name,
        "slice": slice_name,
        "gt_count": len(gt),
        "extracted_count": len(extracted),
        "thresholds": threshold_results,
        "exact_matches": exact_match_count,
        "coverage_at_50_pct": coverage_50,
        "avg_cer_at_70": round(avg_cer, 4) if avg_cer is not None else None,
        "avg_wer_at_70": round(avg_wer, 4) if avg_wer is not None else None,
        "missed_headlines": missed,
        "false_positive_count": len(false_positives),
        "false_positives_sample": false_positives[:10],  # first 10 for review
    }


# ── Report formatting ──────────────────────────────────

def format_table(results: list[dict]) -> str:
    """Format a compact results table for printing to terminal."""
    lines = []
    lines.append("=" * 100)
    lines.append("  VALIDATION RESULTS")
    lines.append("=" * 100)

    header = f"  {'Stage':<15} {'Slice':<12} {'GT':>4} {'Ext':>4} "
    for thr in THRESHOLDS:
        header += f"{'P@'+str(thr):>6} {'R@'+str(thr):>6} {'F1@'+str(thr):>6} "
    header += f"{'Exact':>6} {'CER':>7} {'WER':>7}"
    lines.append(header)
    lines.append("-" * 100)

    for r in results:
        row = f"  {r['stage']:<15} {r['slice']:<12} {r['gt_count']:>4} {r['extracted_count']:>4} "
        for thr in THRESHOLDS:
            t = r["thresholds"][str(thr)]
            row += f"{t['precision']*100:>5.1f}% {t['recall']*100:>5.1f}% {t['f1']*100:>5.1f}% "
        row += f"{r['exact_matches']:>6} "
        row += f"{r['avg_cer_at_70']*100:>6.1f}% " if r['avg_cer_at_70'] is not None else f"{'--':>7} "
        row += f"{r['avg_wer_at_70']*100:>6.1f}%" if r['avg_wer_at_70'] is not None else f"{'--':>7}"
        lines.append(row)

    lines.append("=" * 100)
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────

def main():
    print("Loading ground truth...")
    slice_a_gt = load_ground_truth(GT_DIR / "slice_A_headlines.txt")
    slice_b_gt = load_ground_truth(GT_DIR / "slice_B_headlines.txt")
    combined_gt = list(dict.fromkeys(slice_a_gt + slice_b_gt))  # dedup keep order

    print(f"  Slice A: {len(slice_a_gt)} headlines")
    print(f"  Slice B: {len(slice_b_gt)} headlines")
    print(f"  Combined (union): {len(combined_gt)} unique headlines")

    print("\nLoading extracted items...")
    raw_items = load_extracted(RAW_PATH)
    cleaned_items = load_extracted(CLEANED_PATH)
    print(f"  Raw v6:       {len(raw_items)} items")
    print(f"  LLM-cleaned:  {len(cleaned_items)} items")

    if not raw_items or not cleaned_items:
        print("ERROR: extracted items missing, cannot evaluate")
        return

    results = []
    stages = [
        ("raw_v6", raw_items),
        ("llm_cleaned", cleaned_items),
    ]
    slices = [
        ("slice_A", slice_a_gt),
        ("slice_B", slice_b_gt),
        ("combined", combined_gt),
    ]

    print("\nComputing metrics...")
    for stage_name, extracted in stages:
        for slice_name, gt in slices:
            r = evaluate_stage(extracted, gt, stage_name, slice_name)
            results.append(r)

    # Print summary table
    print()
    print(format_table(results))

    # Detailed output per result
    print("\nDetailed per-stage / per-slice report:")
    for r in results:
        print(f"\n  [{r['stage']} | {r['slice']}]")
        print(f"    Missed headlines ({len(r['missed_headlines'])}):")
        for m in r["missed_headlines"][:5]:
            print(f"      - {m[:90]}")
        if len(r["missed_headlines"]) > 5:
            print(f"      ... and {len(r['missed_headlines'])-5} more")
        print(f"    False positives ({r['false_positive_count']}):")
        for fp in r["false_positives_sample"][:5]:
            print(f"      - {fp[:90]}")

    # Write JSON report
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "ground_truth": {
            "slice_A": {
                "file": str(GT_DIR / "slice_A_headlines.txt"),
                "count": len(slice_a_gt),
            },
            "slice_B": {
                "file": str(GT_DIR / "slice_B_headlines.txt"),
                "count": len(slice_b_gt),
            },
            "combined_unique": len(combined_gt),
        },
        "stages": {
            "raw_v6": {
                "file": str(RAW_PATH),
                "item_count": len(raw_items),
            },
            "llm_cleaned": {
                "file": str(CLEANED_PATH),
                "item_count": len(cleaned_items),
            },
        },
        "thresholds": THRESHOLDS,
        "results": results,
        "notes": (
            "Precision is computed against the combined GT (union of both slices). "
            "Some 'false positives' may be legitimate headlines that appeared in "
            "parts of the 14-hour video outside the two sampled 30-minute slices; "
            "the FP count is therefore an upper bound. Recall is the primary metric "
            "for each slice independently. CER and WER are computed on pairs that "
            "matched at the 70% fuzzy-similarity threshold."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nFull report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
