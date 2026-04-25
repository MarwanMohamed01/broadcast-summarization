"""
Run one OCR engine on one (or both) slice panoramas, measuring time +
memory, and optionally run v6 segmentation on the OCR output.

Usage:
    python -m ocr_comparison.run_ocr_engine --engine tesseract --slice slice_A
    python -m ocr_comparison.run_ocr_engine --engine paddle --slice all
    python -m ocr_comparison.run_ocr_engine --engine easyocr --slice all --skip-segment

Outputs (per slice, per engine):
    ocr_comparison/output/<slice>/<engine>_raw_text.txt
    ocr_comparison/output/<slice>/<engine>_words.json
    ocr_comparison/output/<slice>/<engine>_timing.json
    ocr_comparison/output/<slice>/<engine>_headlines.json       (after segmentation)
    ocr_comparison/output/<slice>/<engine>_segmentation_stats.json

Already-processed (engine, slice) pairs are skipped unless --force.
"""
import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import cv2

from .config import SLICES, OUTPUT_DIR, LOGS_DIR
from .engines import load_engine
from .resource_tracking import current_rss_mb
from .segment_ocr_output import segment_text_for_engine


def _setup_logger(engine: str, slice_name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"ocr_comparison_{engine}_{slice_name}.log"
    logger = logging.getLogger(f"{engine}.{slice_name}")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers on re-run in same process
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)
    return logger


def _already_done(slice_dir: Path, engine: str) -> bool:
    return (
        (slice_dir / f"{engine}_timing.json").exists()
        and (slice_dir / f"{engine}_raw_text.txt").exists()
    )


def run_engine_on_slice(engine_name: str, slice_name: str, skip_segment: bool,
                        force: bool) -> dict:
    slice_cfg = SLICES[slice_name]
    pano_path: Path = slice_cfg["panorama"]
    slice_dir = OUTPUT_DIR / slice_name
    slice_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logger(engine_name, slice_name)

    if not pano_path.exists():
        log.error(f"Panorama missing: {pano_path}")
        return {"status": "error", "reason": "panorama_missing"}

    if _already_done(slice_dir, engine_name) and not force:
        log.info(f"[SKIP] {engine_name} on {slice_name} already done (use --force to redo)")
        return {"status": "skipped"}

    log.info(f"Loading panorama: {pano_path.name}")
    img = cv2.imread(str(pano_path))
    if img is None:
        log.error(f"cv2 failed to read {pano_path}")
        return {"status": "error", "reason": "cv2_read_failed"}
    log.info(f"  size: {img.shape[1]}x{img.shape[0]}")

    engine_mod = load_engine(engine_name)
    log.info(f"Running engine: {engine_name}")

    t0 = time.time()
    peak_rss_mb_before = current_rss_mb()
    try:
        result = engine_mod.run(img)
    except Exception as e:
        log.error(f"Engine crashed: {e}")
        log.error(traceback.format_exc())
        return {"status": "error", "reason": str(e)}
    elapsed = time.time() - t0
    peak_rss_mb_after = current_rss_mb()

    # Save raw text
    (slice_dir / f"{engine_name}_raw_text.txt").write_text(result.full_text, encoding="utf-8")

    # Save words (if any)
    words_out = [asdict(w) for w in result.words]
    with open(slice_dir / f"{engine_name}_words.json", "w", encoding="utf-8") as f:
        json.dump(words_out, f, indent=2, ensure_ascii=False)

    # Save timing
    timing = {
        "engine": engine_name,
        "slice": slice_name,
        "panorama_size_px": [img.shape[1], img.shape[0]],
        "elapsed_seconds": round(elapsed, 2),
        "pixels_per_second": round((img.shape[0] * img.shape[1]) / max(elapsed, 1e-6), 1),
        "rss_mb_before": round(peak_rss_mb_before, 1),
        "rss_mb_after": round(peak_rss_mb_after, 1),
        "rss_mb_delta": round(peak_rss_mb_after - peak_rss_mb_before, 1),
        "text_length_chars": len(result.full_text),
        "word_count": len(result.words),
        "meta": result.meta,
    }
    with open(slice_dir / f"{engine_name}_timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    log.info(f"  OCR done in {elapsed:.1f}s, {len(result.full_text)} chars, "
             f"{len(result.words)} words")

    if not skip_segment:
        log.info("Running v6 segmentation on OCR output...")
        items = segment_text_for_engine(
            result.full_text, slice_name, engine_name, OUTPUT_DIR
        )
        log.info(f"  Segmented into {len(items)} headlines")
    else:
        log.info("Skipping segmentation (--skip-segment)")

    return {"status": "ok", "timing": timing}


def main():
    parser = argparse.ArgumentParser(description="Run one OCR engine on slice(s)")
    parser.add_argument("--engine", required=True, help="Engine name (see ocr_comparison/engines/)")
    parser.add_argument("--slice", default="all", help="slice_A | slice_B | all")
    parser.add_argument("--skip-segment", action="store_true",
                        help="Only run OCR, skip v6 segmentation")
    parser.add_argument("--force", action="store_true",
                        help="Redo even if output already exists")
    args = parser.parse_args()

    slices = [args.slice] if args.slice != "all" else list(SLICES.keys())
    for s in slices:
        if s not in SLICES:
            print(f"Unknown slice: {s}. Valid: {list(SLICES)}")
            sys.exit(2)

    for s in slices:
        print(f"\n=== {args.engine} / {s} ===")
        run_engine_on_slice(args.engine, s, args.skip_segment, args.force)


if __name__ == "__main__":
    main()
