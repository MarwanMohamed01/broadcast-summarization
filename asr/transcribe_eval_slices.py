"""
Run faster-whisper on the two 5-minute eval slices and write timestamped
transcripts to asr/eval/slice_*_whisper.txt.

These short slices fit comfortably in faster-whisper's feature-extractor
memory, so we skip the chunked-streaming logic from asr/transcribe.py
and use a direct call. Same model and settings (small int8 English on CPU)
so the WER measurement is representative of the production transcribe.py.

Idempotent: skips a slice if its output file is newer than its input WAV.

Usage:
    python asr/transcribe_eval_slices.py
    python asr/transcribe_eval_slices.py --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
EVAL_DIR = PROJECT_DIR / "asr" / "eval"
LOGS_DIR = PROJECT_DIR / "logs"

MODEL_NAME = "small"
COMPUTE_TYPE = "int8"
LANG = "en"


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "transcribe_eval_slices.log"
    logger = logging.getLogger("asr.eval.whisper")
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


def fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def transcribe_slice(wav: Path, txt_out: Path, force: bool,
                     log: logging.Logger) -> bool:
    if (txt_out.exists() and not force
            and txt_out.stat().st_mtime >= wav.stat().st_mtime):
        log.info(f"[skip] {txt_out.name} is up-to-date")
        return True

    log.info(f"\n=== Whisper {MODEL_NAME} int8 on {wav.name} ===")
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        log.error(f"faster-whisper not installed: {e}")
        return False

    try:
        model = WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE)
    except Exception as e:
        log.error(f"failed to load model: {e}")
        return False

    log.info(f"  transcribing... (this is a 5-min CPU run, ~1-3 min)")
    try:
        segments, info = model.transcribe(
            str(wav),
            language=LANG,
            beam_size=5,
            vad_filter=False,
        )
    except Exception as e:
        log.error(f"transcribe() failed: {e}")
        return False

    txt_out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(f"# Whisper auto-transcript ({MODEL_NAME} int8, {LANG})\n")
        f.write(f"# Source: {wav.name}\n")
        f.write(f"# Format: [HH:MM:SS.mmm --> HH:MM:SS.mmm] text\n")
        f.write(f"# Detected language: {info.language} "
                f"(prob {info.language_probability:.2f})\n")
        f.write("\n")
        for seg in segments:
            f.write(f"[{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}] "
                    f"{seg.text.strip()}\n")
            n += 1

    log.info(f"  ✓ wrote {n} segments → {txt_out.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-transcribe even if output is up-to-date")
    args = parser.parse_args()

    log = setup_logger()
    pairs = [
        (EVAL_DIR / "slice_A_audio.wav",
         EVAL_DIR / "slice_A_whisper.txt"),
        (EVAL_DIR / "slice_B_audio.wav",
         EVAL_DIR / "slice_B_whisper.txt"),
    ]

    ok = True
    for wav, txt in pairs:
        if not wav.exists():
            log.error(f"missing input WAV: {wav} "
                      f"(run extract_eval_slices.py first)")
            ok = False
            continue
        if not transcribe_slice(wav, txt, args.force, log):
            ok = False

    log.info("\n=== Summary ===")
    for _, txt in pairs:
        if txt.exists():
            with open(txt, encoding="utf-8") as f:
                line_count = sum(1 for line in f if not line.startswith("#")
                                 and line.strip())
            log.info(f"  {txt.name}: {line_count} non-empty lines")
        else:
            log.info(f"  {txt.name}: MISSING")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
