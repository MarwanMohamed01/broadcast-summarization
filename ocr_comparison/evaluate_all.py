"""
Aggregate OCR-comparison results across engines and slices, in BOTH
configurations:

  1. Pure engines       — recognition only, no dash augmentation
                          (reads <engine>_headlines.json)
  2. Engine + dashes    — Tesseract PSM 6 dash augmentation applied,
                          matches v6 production architecture
                          (reads <engine>_headlines_augmented.json)

Both tables are printed and saved.

Usage:
    python -m ocr_comparison.evaluate_all
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR / "validation"))

from .config import SLICES, OUTPUT_DIR, ENGINES  # noqa: E402

from validate_extraction import (  # noqa: E402
    evaluate_stage,
    load_ground_truth,
)


HYBRID_ENGINES = {"craft_trocr"}
END_TO_END_ENGINES = [e for e in ENGINES if e not in HYBRID_ENGINES]


def _load_timing(slice_name: str, engine: str) -> Optional[dict]:
    p = OUTPUT_DIR / slice_name / f"{engine}_timing.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_headlines(path: Path) -> Optional[list]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item["text"].strip() for item in data if item.get("text", "").strip()]


def evaluate_one(headlines_filename_template: str, label: str) -> list[dict]:
    """Evaluate every (engine, slice) pair with the given headlines filename
    template. Template must have a {engine} placeholder."""
    results = []
    for engine in ENGINES:
        for slice_name, cfg in SLICES.items():
            slice_dir = OUTPUT_DIR / slice_name
            headlines_path = slice_dir / headlines_filename_template.format(
                engine=engine)
            headlines = _load_headlines(headlines_path)
            timing = _load_timing(slice_name, engine)
            if headlines is None:
                continue
            gt = load_ground_truth(cfg["ground_truth"])
            metrics = evaluate_stage(
                extracted=headlines,
                gt=gt,
                stage_name=engine,
                slice_name=slice_name,
            )
            metrics["is_hybrid"] = engine in HYBRID_ENGINES
            metrics["timing"] = timing
            metrics["config"] = label
            results.append(metrics)
    return results


def format_one_table(title: str, results: list[dict]) -> str:
    lines = []
    lines.append("=" * 118)
    lines.append(f"  {title} — END-TO-END ENGINES")
    lines.append("=" * 118)
    header = (f"  {'Engine':<14} {'Slice':<10} {'GT':>3} {'Ext':>4} "
              f"{'P@70':>7} {'R@70':>7} {'F1@70':>7} "
              f"{'CER':>6} {'WER':>6} {'Time(s)':>8} {'Mem(MB)':>8}")
    lines.append(header)
    lines.append("-" * 118)

    def row(r):
        t = r["thresholds"]["70"]
        cer = r["avg_cer_at_70"]
        wer = r["avg_wer_at_70"]
        tm = r.get("timing") or {}
        sec = tm.get("elapsed_seconds", "--")
        mem = tm.get("rss_mb_delta", "--")
        return (
            f"  {r['stage']:<14} {r['slice']:<10} {r['gt_count']:>3} {r['extracted_count']:>4} "
            f"{t['precision']*100:>6.1f}% {t['recall']*100:>6.1f}% {t['f1']*100:>6.1f}% "
            f"{(cer*100 if cer is not None else 0):>5.1f}% "
            f"{(wer*100 if wer is not None else 0):>5.1f}% "
            f"{sec:>8} {mem:>8}"
        )

    for r in results:
        if not r["is_hybrid"]:
            lines.append(row(r))

    hybrid_rows = [r for r in results if r["is_hybrid"]]
    if hybrid_rows:
        lines.append("")
        lines.append("=" * 118)
        lines.append(f"  {title} — RECOGNIZER-ONLY EXPERIMENT (CRAFT detector + TrOCR recognizer)")
        lines.append("  NOTE: Not directly comparable — uses EasyOCR's detector, only the recognizer differs.")
        lines.append("=" * 118)
        lines.append(header)
        lines.append("-" * 118)
        for r in hybrid_rows:
            lines.append(row(r))
    lines.append("=" * 118)
    return "\n".join(lines)


def main():
    print("Evaluating all engines across all slices in both configurations...")

    pure_results = evaluate_one("{engine}_headlines.json", "pure")
    aug_results = evaluate_one("{engine}_headlines_augmented.json", "augmented")

    full_report = {
        "timestamp": datetime.now().isoformat(),
        "engines_compared": ENGINES,
        "end_to_end_engines": END_TO_END_ENGINES,
        "hybrid_engines": sorted(HYBRID_ENGINES),
        "configurations": {
            "pure": "Engine alone — no Tesseract dash augmentation",
            "augmented": "Engine + Tesseract PSM 6 dash augmentation (matches v6 production)",
        },
        "pure_results": pure_results,
        "augmented_results": aug_results,
    }

    out_json = OUTPUT_DIR / "comparison_report.json"
    out_json.write_text(json.dumps(full_report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    pure_table = format_one_table("PURE ENGINES (no dash augmentation)", pure_results)
    aug_table = format_one_table("ENGINES + TESSERACT DASH AUGMENTATION", aug_results)

    full_table = pure_table + "\n\n" + aug_table
    print()
    print(full_table)

    out_table = OUTPUT_DIR / "comparison_report.txt"
    out_table.write_text(full_table, encoding="utf-8")

    print(f"\nFull report: {out_json}")
    print(f"Tables:      {out_table}")


if __name__ == "__main__":
    main()
