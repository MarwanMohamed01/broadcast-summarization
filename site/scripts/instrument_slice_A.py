"""
Re-run v6's pipeline on the same frame range as ground-truth Slice A
(8:30–9:00 of the 14h video) and capture *real* per-word EasyOCR
bounding boxes + per-frame scroll offsets.

v6 is frozen and does NOT persist these artefacts. We import its modules
read-only and intercept what we need by monkey-patching only at the
function-boundary level (no v6 source modifications).

Outputs (all under site/public/data/instrumented/slice_A/):
  - ocr_bboxes.json        per-word {x, y, w, h, text, confidence} on the
                           first 3000-px-wide center-only OCR slice. The
                           OCRStage.jsx component renders these on top of
                           the real panorama segment image.
  - scroll_deltas.json     per-frame {frame_idx, delta_px}. The
                           ScrollDetectionStage.jsx component picks the
                           median to display as a representative number.
  - frame_stats.json       per-chunk frame counts (kept vs total) — feeds
                           the FrameExtractionStage counts.

Idempotent: skips if all three outputs are newer than the source video
file. Pass --force to redo.

Time: ~5–10 min on CPU (does the heavy v6 steps 1+2+4 for one 30-min slice).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()
SITE_PUBLIC = PROJECT_DIR / "site" / "public" / "data" / "instrumented" / "slice_A"
WORK_ROOT = PROJECT_DIR / "logs" / "instrument_slice_A_work"
LOGS_DIR = PROJECT_DIR / "logs"
V6_DIR = PROJECT_DIR / "ticker_extraction_v6"
GT_DIR = PROJECT_DIR / "validation" / "ground_truth"
VIDEOS_DIR = PROJECT_DIR / "videos"

# v6 chunked mode = 30 minutes. Slice A == chunk 17 (8:30 → 9:00)
SLICE_A_CHUNK = 17
CHUNK_MINUTES = 30


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("instrument.slice_A")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)
        fh = logging.FileHandler(LOGS_DIR / "instrument_slice_A.log",
                                 encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger


log = setup_logger()


def find_source_video() -> Path:
    for c in VIDEOS_DIR.glob("*.mp4"):
        if "14hrs" in c.name or "14hr" in c.name:
            return c
    raise FileNotFoundError("no 14h video in videos/")


def all_outputs_fresh(src: Path) -> bool:
    expected = ["ocr_bboxes.json", "scroll_deltas.json", "frame_stats.json"]
    if not all((SITE_PUBLIC / f).exists() for f in expected):
        return False
    src_mt = src.stat().st_mtime
    return all((SITE_PUBLIC / f).stat().st_mtime >= src_mt for f in expected)


def main(force: bool = False) -> int:
    src = find_source_video()
    if all_outputs_fresh(src) and not force:
        log.info("[skip] all instrumented outputs already up-to-date "
                 "(use --force to redo)")
        return 0

    SITE_PUBLIC.mkdir(parents=True, exist_ok=True)

    # Import v6 modules (frozen — no edits) by adding its dir to sys.path
    sys.path.insert(0, str(V6_DIR))
    import config as v6_config           # noqa: E402
    from step1_extract_ticker import extract_ticker_frames  # noqa: E402
    from step2_scroll_detection import detect_all_scrolls   # noqa: E402

    # ── 1. Frame range (chunk 17 = 8:30–9:00) ──
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    chunk_frames = int(CHUNK_MINUTES * 60 * fps)
    start_frame = SLICE_A_CHUNK * chunk_frames
    end_frame = min(start_frame + chunk_frames, total_frames)
    log.info(f"Slice A frame range: {start_frame}–{end_frame} "
             f"({(end_frame-start_frame)/fps/60:.1f} min)")

    # Heavy intermediate output (extracted ticker frames) goes OUTSIDE
    # site/public/ so it doesn't blow the site asset budget and so a
    # crashed run doesn't leak into the deploy bundle.
    work = WORK_ROOT
    if work.exists():
        import shutil
        shutil.rmtree(work)
    work.mkdir(parents=True)
    frames_dir = work / "ticker_frames"

    # ── 2. v6 step 1: extract ticker frames (real numbers) ──
    log.info("\n[Step 1] Extracting ticker frames...")
    t0 = time.time()
    frame_paths, step1_stats = extract_ticker_frames(
        src,
        start_frame=start_frame,
        end_frame=end_frame,
        output_dir=frames_dir,
    )
    log.info(f"  Extracted {len(frame_paths)} frames in {time.time()-t0:.0f}s")

    # frame_stats.json — real counts for FrameExtractionStage
    frame_stats = {
        "source_frames":   end_frame - start_frame,
        "kept_frames":     len(frame_paths),
        "frame_sample_rate": v6_config.FRAME_SAMPLE_RATE,
        "fps":             fps,
        "duration_seconds":(end_frame - start_frame) / fps,
        "step1_extra":     step1_stats.get("frame_extraction") if isinstance(step1_stats, dict) else {},
    }
    (SITE_PUBLIC / "frame_stats.json").write_text(
        json.dumps(frame_stats, indent=2), encoding="utf-8")
    log.info(f"  -> frame_stats.json (kept {frame_stats['kept_frames']} / {frame_stats['source_frames']})")

    # ── 3. v6 step 2: scroll detection (real per-frame deltas) ──
    log.info("\n[Step 2] Scroll detection...")
    t0 = time.time()
    scroll_data, step2_stats = detect_all_scrolls(frame_paths)
    log.info(f"  Done in {time.time()-t0:.0f}s")

    # Per v6's step2_scroll_detection.detect_all_scrolls(), each entry is a
    # dict with keys: path, scroll_offset, cumulative_offset, confidence.
    # We want the per-frame `scroll_offset` (already in pixels), excluding
    # the first frame which is always 0 by construction.
    deltas = []
    for entry in (scroll_data or [])[1:]:
        if isinstance(entry, dict):
            v = entry.get("scroll_offset")
            if v is not None and v > 0:   # ignore zero-shift filler frames
                deltas.append(float(v))

    deltas_payload = {
        "count": len(deltas),
        "median_px": statistics.median(deltas) if deltas else None,
        "mean_px":   statistics.mean(deltas) if deltas else None,
        "min_px":    min(deltas) if deltas else None,
        "max_px":    max(deltas) if deltas else None,
        "samples":   deltas[:200],   # first 200 for the chart, rest implied
    }
    (SITE_PUBLIC / "scroll_deltas.json").write_text(
        json.dumps(deltas_payload, indent=2), encoding="utf-8")
    log.info(f"  -> scroll_deltas.json (median Δx = "
             f"{deltas_payload['median_px']} px over {len(deltas)} frames)")

    # ── 4. EasyOCR bboxes on the FIRST panorama segment ──
    log.info("\n[Step 4] EasyOCR bboxes on first 3000-px panorama segment...")
    pano_path = GT_DIR / "slice_A_panorama.png"
    if not pano_path.exists():
        log.warning(f"  missing {pano_path}; skipping bbox capture")
        return 0
    pano = cv2.imread(str(pano_path))
    h, w = pano.shape[:2]

    # Take the FIRST 3000-px slice (matches segment_000.webp on the site)
    SLICE_W = 3000
    slice0 = pano[:, :min(SLICE_W, w)]
    # Match v6's preprocessing: 3x upscale before EasyOCR
    slice_up = cv2.resize(slice0, (slice0.shape[1]*3, slice0.shape[0]*3),
                          interpolation=cv2.INTER_CUBIC)

    import easyocr  # noqa: E402
    log.info("  Loading EasyOCR (CPU)...")
    reader = easyocr.Reader(["en"], gpu=False)

    # Force WORD-level boxes by tightening EasyOCR's line-merging knobs.
    # Defaults are tuned for paragraph reading and merge most of a ticker
    # row into 4–6 boxes; for the demo we want one box per visible word.
    log.info("  Running readtext (word-level)...")
    t0 = time.time()
    results = reader.readtext(
        slice_up,
        paragraph=False,
        width_ths=0.0,    # never merge horizontally adjacent boxes
        ycenter_ths=0.5,
        height_ths=0.5,
        decoder="greedy",
    )
    log.info(f"  Done in {time.time()-t0:.0f}s ({len(results)} boxes)")

    bboxes = []
    for (bbox, txt, conf) in results:
        if conf <= 0.30 or not (txt or "").strip():
            continue
        # bbox = 4 corners in upscaled coords; divide by 3 to get original-slice px
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min = int(min(xs) / 3)
        y_min = int(min(ys) / 3)
        x_max = int(max(xs) / 3)
        y_max = int(max(ys) / 3)
        bboxes.append({
            "x": x_min,
            "y": y_min,
            "w": max(1, x_max - x_min),
            "h": max(1, y_max - y_min),
            "text": txt.strip(),
            "confidence": round(float(conf), 3),
        })

    bboxes_payload = {
        "panorama_segment": "slice_A/segment_000.webp",
        "segment_size_px": [slice0.shape[1], slice0.shape[0]],
        "engine": "EasyOCR (CRAFT + CRNN, en, CPU)",
        "preprocessing": "3x cubic upscale before recognition",
        "confidence_threshold": 0.30,
        "word_count": len(bboxes),
        "words": bboxes,
    }
    (SITE_PUBLIC / "ocr_bboxes.json").write_text(
        json.dumps(bboxes_payload, indent=2), encoding="utf-8")
    log.info(f"  -> ocr_bboxes.json ({len(bboxes)} words)")

    # ── 5. Cleanup heavy frame folder ──
    import shutil
    shutil.rmtree(work, ignore_errors=True)

    log.info("\nDone.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Redo even if outputs are up-to-date")
    args = parser.parse_args()
    sys.exit(main(force=args.force))
