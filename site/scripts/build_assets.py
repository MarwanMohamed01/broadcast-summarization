"""
Asset pipeline for site/. Copies, downsamples, and pre-renders everything
the defense site needs into site/public/data/.

Idempotent: every output is mtime-compared to its source(s); up-to-date
files are skipped. Anything that would exceed 2 MB is downsampled or
skipped with a loud warning.

Run after any change in upstream output folders, or after a fresh clone:
    python site/scripts/build_assets.py

Logs to logs/site_build.log.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()
SITE_DIR = PROJECT_DIR / "site"
DATA_DIR = SITE_DIR / "public" / "data"
VIDEO_DIR = SITE_DIR / "public" / "video"
LOGS_DIR = PROJECT_DIR / "logs"

V6_DIR  = PROJECT_DIR / "ticker_extraction_v6"
LLM_DIR = PROJECT_DIR / "llm_summarization"
GT_DIR  = PROJECT_DIR / "validation" / "ground_truth"
ASR_DIR = PROJECT_DIR / "asr" / "output"
RESULTS = PROJECT_DIR / "results"
VIDEOS  = PROJECT_DIR / "videos"

MAX_FILE_BYTES = 2 * 1024 * 1024   # 2 MB hard limit per file
MAX_TOTAL_BYTES = 20 * 1024 * 1024  # 20 MB hard limit total

PANO_THUMB_WIDTH = 1500   # full-panorama overview thumbnail width
PANO_SEGMENT_WIDTH = 3000 # OCR-stage animation strip width
PANO_SEGMENT_COUNT = 10
WAVEFORM_PEAKS = 2000


# ── logging ────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "site_build.log"
    logger = logging.getLogger("site.build")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


log = setup_logger()


# ── helpers ────────────────────────────────────────────────

def is_up_to_date(src: Path, out: Path) -> bool:
    return out.exists() and out.stat().st_mtime >= src.stat().st_mtime


def is_up_to_date_multi(srcs: Iterable[Path], out: Path) -> bool:
    if not out.exists():
        return False
    out_mt = out.stat().st_mtime
    return all(s.exists() and out_mt >= s.stat().st_mtime for s in srcs)


def enforce_size(path: Path) -> None:
    if path.stat().st_size > MAX_FILE_BYTES:
        log.warning("OVERSIZE %s = %.1f MB (limit 2 MB) — leaving but flag",
                    path.name, path.stat().st_size / 1024 / 1024)


def copy_file(src: Path, out: Path) -> None:
    if not src.exists():
        log.warning("missing source: %s", src)
        return
    if is_up_to_date(src, out):
        log.info("[skip] %s", out.relative_to(SITE_DIR))
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    enforce_size(out)
    log.info("[copy] %s (%d KB)", out.relative_to(SITE_DIR), out.stat().st_size // 1024)


# ── stage 1: structured JSON copies ────────────────────────

def aggregate_eval_scores() -> None:
    """
    Read llm_summarization/output*/evaluation_latest.json (or the latest
    timestamped evaluation_*.json) for each pipeline variant and write a
    single eval_scores.json the EvalStage chart can fetch directly.

    Output schema:
        {
          "visual_27min":      [{ "model": str, "rougeL": float, "bertscore": float, "latency_seconds": float }, ...],
          "visual_14h_cleaned":[ ... ],
          "asr_final":         [ ... ]
        }
    """
    log.info("\n== Aggregate per-LLM ROUGE/BERTScore ==")
    sources = {
        "visual_27min":       LLM_DIR / "output",
        "visual_14h_cleaned": LLM_DIR / "output_cleaned",
        "asr_final":          LLM_DIR / "output_asr",
    }
    out_path = DATA_DIR / "stats/eval_scores.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {}
    for variant, folder in sources.items():
        if not folder.exists():
            log.warning("  missing %s", folder)
            continue
        # Prefer evaluation_latest.json, else newest evaluation_*.json
        candidates = list(folder.glob("evaluation_latest.json"))
        if not candidates:
            candidates = sorted(folder.glob("evaluation_*.json"))
        if not candidates:
            log.warning("  no evaluation_*.json in %s", folder.name)
            payload[variant] = []
            continue
        src = candidates[-1]
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("  failed to parse %s: %s", src.name, e)
            payload[variant] = []
            continue

        rows = []
        for r in data.get("results", []):
            rows.append({
                "model":            r.get("display_name", "unknown"),
                "provider":         r.get("provider", ""),
                "rougeL":           r.get("rougeL", 0.0),
                "rouge1":           r.get("rouge1", 0.0),
                "rouge2":           r.get("rouge2", 0.0),
                "bertscore":        r.get("bertscore_f1", 0.0),
                "latency_seconds":  r.get("latency_seconds", 0.0),
            })
        payload[variant] = rows
        log.info("  [build] %s: %d models from %s",
                 variant, len(rows), src.name)

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    enforce_size(out_path)
    log.info("[build] %s (%d KB)", out_path.relative_to(SITE_DIR),
             out_path.stat().st_size // 1024)


def copy_jsons() -> None:
    log.info("\n== JSON: headlines / stats / summaries / GT / report ==")
    pairs = [
        (V6_DIR / "output/final/news_items.json",
         DATA_DIR / "headlines/news_items.json"),
        (V6_DIR / "output/final/news_items_cleaned.json",
         DATA_DIR / "headlines/news_items_cleaned.json"),
        (V6_DIR / "output/final/pipeline_stats.json",
         DATA_DIR / "stats/pipeline_stats.json"),
        (RESULTS / "validation_report.json",
         DATA_DIR / "stats/validation_report.json"),
        (GT_DIR / "slice_A_headlines.txt",
         DATA_DIR / "ground_truth/slice_A_headlines.txt"),
        (GT_DIR / "slice_B_headlines.txt",
         DATA_DIR / "ground_truth/slice_B_headlines.txt"),
        (LLM_DIR / "output/latest.json",
         DATA_DIR / "summaries/visual_27min.json"),
        (LLM_DIR / "output_cleaned/latest.json",
         DATA_DIR / "summaries/visual_14h_cleaned.json"),
        (LLM_DIR / "output_asr/latest.json",
         DATA_DIR / "summaries/asr_final.json"),
    ]
    for src, out in pairs:
        copy_file(src, out)


# ── stage 2: panorama crops (Slice A + B) ──────────────────

def crop_panorama(slice_letter: str) -> None:
    src = GT_DIR / f"slice_{slice_letter}_panorama.png"
    if not src.exists():
        log.warning("missing panorama: %s", src)
        return

    out_dir = DATA_DIR / "panorama" / f"slice_{slice_letter}"
    out_dir.mkdir(parents=True, exist_ok=True)

    thumb_path = out_dir / "full_thumb.webp"
    micro_path = out_dir / "micro.webp"
    seg_paths = [out_dir / f"segment_{i:03d}.webp"
                 for i in range(PANO_SEGMENT_COUNT)]

    if all(is_up_to_date(src, p) for p in [thumb_path, micro_path] + seg_paths):
        log.info("[skip] panorama crops slice_%s", slice_letter)
        return

    log.info("[crop] panorama slice_%s (%.1f MB) -> webp",
             slice_letter, src.stat().st_size / 1024 / 1024)

    img = Image.open(src).convert("RGB")
    w, h = img.size

    # full thumbnail
    thumb_h = max(1, int(h * PANO_THUMB_WIDTH / w))
    img.resize((PANO_THUMB_WIDTH, thumb_h), Image.LANCZOS)\
       .save(thumb_path, format="WEBP", quality=80)
    enforce_size(thumb_path)

    # micro placeholder for above-the-fold loading
    img.resize((200, max(1, int(h * 200 / w))), Image.LANCZOS)\
       .save(micro_path, format="WEBP", quality=60)

    # 10 evenly spaced full-resolution segments for the OCR stage animation
    if w < PANO_SEGMENT_WIDTH:
        log.warning("  panorama narrower than segment width; widening sample")
        seg_w = w // PANO_SEGMENT_COUNT
    else:
        seg_w = PANO_SEGMENT_WIDTH
    spacing = max(1, (w - seg_w) // (PANO_SEGMENT_COUNT - 1))
    for i, p in enumerate(seg_paths):
        x0 = i * spacing
        x1 = min(w, x0 + seg_w)
        img.crop((x0, 0, x1, h))\
           .save(p, format="WEBP", quality=78, method=6)
        enforce_size(p)

    log.info("  wrote thumb + micro + %d segments", PANO_SEGMENT_COUNT)


# ── stage 3: ASR transcript + chunks ───────────────────────

def copy_transcripts() -> None:
    log.info("\n== ASR: transcripts + chunks ==")
    pairs = [
        (ASR_DIR / "transcript_slice_A.json",
         DATA_DIR / "transcript/transcript_slice_A.json"),
        (ASR_DIR / "transcript_slice_A.txt",
         DATA_DIR / "transcript/transcript_slice_A.txt"),
        (ASR_DIR / "transcript_slice_A.srt",
         DATA_DIR / "transcript/transcript_slice_A.srt"),
    ]
    for src, out in pairs:
        copy_file(src, out)

    # Sample 12 chunk .txt files for the chunking-stage animation
    chunks_in = ASR_DIR / "chunks"
    chunks_out = DATA_DIR / "chunks"
    chunks_out.mkdir(parents=True, exist_ok=True)
    if chunks_in.exists():
        all_chunks = sorted(chunks_in.glob("chunk_*.txt"))
        sample_idx = np.linspace(0, len(all_chunks) - 1, 12).astype(int) \
                     if len(all_chunks) > 12 else range(len(all_chunks))
        for i in sample_idx:
            copy_file(all_chunks[i], chunks_out / all_chunks[i].name)
    else:
        log.warning("missing %s", chunks_in)


# ── stage 4: ASR Level-1 intermediate summaries ───────────

def copy_level1() -> None:
    log.info("\n== ASR: Level-1 intermediate summaries ==")
    matches = sorted((LLM_DIR / "output_asr").glob("level1_*.json"))
    if not matches:
        log.warning("no level1_*.json found in llm_summarization/output_asr/")
        return
    src = matches[-1]
    out = DATA_DIR / "summaries/asr_level1.json"
    copy_file(src, out)


# ── stage 5: waveform peaks for ASR-stage scrubber ─────────

def build_waveform_peaks() -> None:
    log.info("\n== Waveform peaks for ASR scrubber ==")
    src = VIDEOS / "demo_clip_30s.wav"
    out = DATA_DIR / "audio/peaks.json"
    if not src.exists():
        log.warning("demo WAV not found: %s "
                    "(run videos/extract_demo_clip.py first)", src)
        return
    if is_up_to_date(src, out):
        log.info("[skip] %s", out.relative_to(SITE_DIR))
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(src), "rb") as wf:
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    samples = np.frombuffer(raw, dtype=dtype)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    samples = samples.astype(np.float32) / np.iinfo(dtype).max

    n_buckets = WAVEFORM_PEAKS
    bucket_size = max(1, len(samples) // n_buckets)
    peaks = []
    for i in range(n_buckets):
        chunk = samples[i * bucket_size:(i + 1) * bucket_size]
        if len(chunk):
            peaks.append([float(chunk.min()), float(chunk.max())])
    payload = {
        "framerate": framerate,
        "duration_sec": n_frames / framerate,
        "peak_count": len(peaks),
        "peaks": peaks,
    }
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    enforce_size(out)
    log.info("[build] %s (%d KB)", out.relative_to(SITE_DIR),
             out.stat().st_size // 1024)


# ── stage 6: source video preview ──────────────────────────

def copy_demo_video() -> None:
    log.info("\n== Demo video preview ==")
    src = VIDEOS / "demo_clip_30s.mp4"
    out = VIDEO_DIR / "demo_clip_30s.mp4"
    if not src.exists():
        log.warning("demo MP4 not found: %s "
                    "(run videos/extract_demo_clip.py first)", src)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    copy_file(src, out)


# ── stage 7: total-size guard ──────────────────────────────

def total_data_bytes() -> int:
    total = 0
    for p in DATA_DIR.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    for p in VIDEO_DIR.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Site asset build starting (project=%s)", PROJECT_DIR)

    copy_jsons()
    aggregate_eval_scores()
    log.info("\n== Panorama crops ==")
    for letter in ("A", "B"):
        crop_panorama(letter)
    copy_transcripts()
    copy_level1()
    build_waveform_peaks()
    copy_demo_video()

    total = total_data_bytes()
    log.info("\nTotal site assets: %.2f MB (limit %.0f MB)",
             total / 1024 / 1024, MAX_TOTAL_BYTES / 1024 / 1024)
    if total > MAX_TOTAL_BYTES:
        log.error("Asset payload exceeds budget!")
        return 1
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
