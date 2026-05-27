"""
Extract two 5-minute audio slices that align with the existing OCR
ground-truth slices (Slice A 08:30:00–08:35:00, Slice B 13:00:00–13:05:00).

Used by the ASR evaluation pipeline to produce a modality-matched
ground-truth audio dataset. The 5-minute window per slice is short
enough for manual transcription (~25–35 minutes of careful work) and
long enough for a meaningful WER/CER estimate.

Format: 16 kHz mono PCM s16 WAV (matches Whisper's preprocessing).
Reuses the proven extractor from asr/extract_audio.py — that file is
imported, NOT modified.

Usage:
    python asr/extract_eval_slices.py                   # auto-detect 14h video
    python asr/extract_eval_slices.py --video path.mp4
    python asr/extract_eval_slices.py --force           # redo even if outputs exist
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
ASR_DIR = PROJECT_DIR / "asr"
EVAL_DIR = ASR_DIR / "eval"
LOGS_DIR = PROJECT_DIR / "logs"

# Reuse the existing helper from asr/extract_audio.py without modifying it.
sys.path.insert(0, str(ASR_DIR))
from extract_audio import extract_audio, find_video  # noqa: E402

# Slice timings (matched to ground-truth panoramas)
SLICE_A_START_SEC = 8 * 3600 + 30 * 60   # 08:30:00
SLICE_B_START_SEC = 13 * 3600            # 13:00:00
SLICE_DURATION_SEC = 5 * 60              # 5 minutes


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "extract_eval_slices.log"
    logger = logging.getLogger("asr.eval.extract")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def extract_slice(video: Path, slice_letter: str, start_sec: int,
                  out_path: Path, force: bool, log: logging.Logger) -> bool:
    """Returns True on success, False on skip/error."""
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1e6
        log.info(f"[skip] Slice {slice_letter} already exists "
                 f"({out_path.name}, {size_mb:.1f} MB)")
        return True

    if force and out_path.exists():
        log.info(f"  --force: removing existing {out_path.name}")
        out_path.unlink()

    log.info(f"\n=== Extracting Slice {slice_letter} ===")
    log.info(f"  Source: {video.name}")
    log.info(f"  Start:  {start_sec}s  ({start_sec//3600:02d}:"
             f"{(start_sec%3600)//60:02d}:{start_sec%60:02d})")
    log.info(f"  Length: {SLICE_DURATION_SEC}s "
             f"({SLICE_DURATION_SEC//60} min)")
    log.info(f"  Output: {out_path}")

    try:
        extract_audio(
            video_path=video,
            out_path=out_path,
            start_sec=start_sec,
            duration_sec=SLICE_DURATION_SEC,
        )
    except Exception as e:
        log.error(f"FAILED to extract Slice {slice_letter}: {e}")
        return False

    if not out_path.exists():
        log.error(f"FAILED — output file did not appear: {out_path}")
        return False

    size_mb = out_path.stat().st_size / 1e6
    log.info(f"  ✓ Wrote {out_path.name} ({size_mb:.2f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Slice A + B audio for ASR evaluation")
    parser.add_argument("--video", type=str,
                        help="Source MP4 (default: auto-detect 14h)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if output files exist")
    args = parser.parse_args()

    log = setup_logger()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        video = Path(args.video) if args.video else find_video()
    except FileNotFoundError as e:
        log.error(f"{e}")
        return 1
    if not video.exists():
        log.error(f"Video not found: {video}")
        return 1

    slices = [
        ("A", SLICE_A_START_SEC, EVAL_DIR / "slice_A_audio.wav"),
        ("B", SLICE_B_START_SEC, EVAL_DIR / "slice_B_audio.wav"),
    ]

    ok = True
    for letter, start, out in slices:
        if not extract_slice(video, letter, start, out, args.force, log):
            ok = False

    log.info("\n=== Summary ===")
    for letter, _, out in slices:
        if out.exists():
            log.info(f"  Slice {letter}: {out.relative_to(PROJECT_DIR)} "
                     f"({out.stat().st_size/1e6:.2f} MB)")
        else:
            log.info(f"  Slice {letter}: MISSING")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
