"""
Rebuild the *pure-engine* (no Tesseract dash augmentation) raw text and
segmented headlines from each engine's cached words.json.

Used after augment_dashes.py so we can report BOTH tables in the thesis:
  - Pure engines (recognition + native delimiters only)
  - Engines + Tesseract dash augmentation

No OCR is re-run; this only re-runs the v6 segmentation step.

Outputs (overwrites pure-engine files only — augmented files untouched):
    ocr_comparison/output/<slice>/<engine>_raw_text.txt
    ocr_comparison/output/<slice>/<engine>_headlines.json
    ocr_comparison/output/<slice>/<engine>_segmentation_stats.json
"""
import argparse
import json
from pathlib import Path

from .config import SLICES, OUTPUT_DIR, ENGINES
from .segment_ocr_output import segment_text_for_engine


def words_to_pure_text(words: list[dict]) -> str:
    """Sort words by left-x position and concatenate — no dash injection."""
    sorted_words = sorted(words, key=lambda w: int(w["x"]))
    return " ".join(w["text"] for w in sorted_words if w.get("text"))


def regenerate_engine_slice(engine: str, slice_name: str) -> dict:
    slice_dir = OUTPUT_DIR / slice_name
    words_path = slice_dir / f"{engine}_words.json"
    if not words_path.exists():
        return {"status": "skipped", "reason": "no_words_file"}

    words = json.loads(words_path.read_text(encoding="utf-8"))
    pure_text = words_to_pure_text(words)
    (slice_dir / f"{engine}_raw_text.txt").write_text(pure_text, encoding="utf-8")

    items = segment_text_for_engine(pure_text, slice_name, engine, OUTPUT_DIR)
    return {
        "status": "ok",
        "words": len(words),
        "pure_chars": len(pure_text),
        "final_headlines": len(items),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="all",
                        help=f"Engine name or 'all' (choices: {ENGINES})")
    parser.add_argument("--slice", default="all",
                        help="slice_A | slice_B | all")
    args = parser.parse_args()

    engines = ENGINES if args.engine == "all" else [args.engine]
    slices = list(SLICES) if args.slice == "all" else [args.slice]

    for slice_name in slices:
        print(f"\n=== {slice_name} (PURE) ===")
        for engine in engines:
            print(f"[{engine}] ...")
            r = regenerate_engine_slice(engine, slice_name)
            print(f"  -> {r}")


if __name__ == "__main__":
    main()
