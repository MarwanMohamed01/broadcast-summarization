"""
Augment every engine's OCR output with Tesseract-detected dash positions,
then re-run v6 segmentation. This isolates the dash-detection variable
so the comparison measures recognition quality, not delimiter coverage.

Matches v6's production architecture: v6 uses EasyOCR for recognition
and Tesseract (PSM 6) for ' - ' delimiter detection. Without the dash
pass, scene-text recognizers skip the thin '-' glyph entirely.

Inputs (already produced by run_ocr_engine.py):
    ocr_comparison/output/<slice>/<engine>_words.json

Outputs (do NOT overwrite the pure-engine outputs — separate files):
    ocr_comparison/output/<slice>/<engine>_raw_text_augmented.txt
    ocr_comparison/output/<slice>/<engine>_headlines_augmented.json
    ocr_comparison/output/<slice>/<engine>_segmentation_stats_augmented.json

Plus a per-slice dash cache:
    ocr_comparison/output/<slice>/_tesseract_dashes.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .config import SLICES, OUTPUT_DIR, SLICE_W, STRIDE, TESSERACT_CMD, ENGINES
from .segment_ocr_output import segment_text_for_engine

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def detect_dashes_in_slice(slice_img: np.ndarray) -> list[int]:
    """Tesseract PSM 6 dash finder, one panorama-slice at a time."""
    gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        data = pytesseract.image_to_data(
            binary, config="--psm 6", output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []
    xs = []
    for i, txt in enumerate(data["text"]):
        if txt.strip() in ("-", "—", "–", "~"):
            xs.append(int(data["left"][i]))
    return xs


def detect_panorama_dashes(panorama_path: Path, cache_path: Path) -> list[int]:
    """Run Tesseract PSM 6 across the whole panorama and return dash
    x-positions in panorama coords. Cached because it is slow and
    identical for every engine sharing the same slice."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    img = cv2.imread(str(panorama_path))
    if img is None:
        raise RuntimeError(f"Cannot read panorama: {panorama_path}")

    h, w = img.shape[:2]
    margin = STRIDE // 2
    all_dashes: list[int] = []

    print(f"  Running Tesseract PSM 6 dash detection on {panorama_path.name} ({w}x{h})...")
    for x in range(0, w, STRIDE):
        x_end = min(x + SLICE_W, w)
        if x_end - x < 500:
            break
        slice_img = img[:, x:x_end]
        dashes = detect_dashes_in_slice(slice_img)

        slice_w = x_end - x
        if x == 0:
            keep = [d for d in dashes if d < slice_w - margin]
        elif x + SLICE_W >= w:
            keep = [d for d in dashes if d >= margin]
        else:
            keep = [d for d in dashes if margin <= d < slice_w - margin]

        for d in keep:
            all_dashes.append(d + x)

    all_dashes.sort()
    cache_path.write_text(json.dumps(all_dashes), encoding="utf-8")
    print(f"    {len(all_dashes)} dashes found, cached to {cache_path.name}")
    return all_dashes


def merge_words_and_dashes(words: list[dict], dash_xs: list[int]) -> str:
    """Insert a synthetic ' - ' token at each dash x-position, sort by x,
    and return the concatenated text — same strategy v6 step4 uses."""
    tokens = [(int(w["x"]), w["text"]) for w in words]
    tokens.extend([(x, "-") for x in dash_xs])
    tokens.sort(key=lambda t: t[0])
    return " ".join(tok for _, tok in tokens)


def augment_engine_slice(engine: str, slice_name: str,
                         panorama_path: Path, dash_cache: Path) -> dict:
    slice_dir = OUTPUT_DIR / slice_name
    words_path = slice_dir / f"{engine}_words.json"
    if not words_path.exists():
        return {"status": "skipped", "reason": "no_words_file"}

    words = json.loads(words_path.read_text(encoding="utf-8"))
    if not words:
        return {"status": "skipped", "reason": "empty_words"}

    dashes = detect_panorama_dashes(panorama_path, dash_cache)
    augmented_text = merge_words_and_dashes(words, dashes)

    (slice_dir / f"{engine}_raw_text_augmented.txt").write_text(
        augmented_text, encoding="utf-8")

    # Use a synthetic "engine name" so the segmentation wrapper writes
    # to <engine>_augmented_headlines.json without clobbering the pure
    # <engine>_headlines.json. We rename the result back to a tidy
    # <engine>_headlines_augmented.json afterwards.
    items = segment_text_for_engine(
        augmented_text, slice_name, f"{engine}_augmented", OUTPUT_DIR)

    # Rename to the canonical naming used by evaluate_all
    raw = slice_dir / f"{engine}_augmented_headlines.json"
    raw_stats = slice_dir / f"{engine}_augmented_segmentation_stats.json"
    if raw.exists():
        raw.replace(slice_dir / f"{engine}_headlines_augmented.json")
    if raw_stats.exists():
        raw_stats.replace(
            slice_dir / f"{engine}_segmentation_stats_augmented.json")

    return {
        "status": "ok",
        "dashes": len(dashes),
        "words": len(words),
        "augmented_chars": len(augmented_text),
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
        cfg = SLICES[slice_name]
        slice_dir = OUTPUT_DIR / slice_name
        slice_dir.mkdir(parents=True, exist_ok=True)
        dash_cache = slice_dir / "_tesseract_dashes.json"

        print(f"\n=== {slice_name} ===")
        for engine in engines:
            print(f"[{engine}] ...")
            r = augment_engine_slice(engine, slice_name, cfg["panorama"], dash_cache)
            print(f"  -> {r}")


if __name__ == "__main__":
    main()
