"""
Compute WER / CER / MER / WIL for the two ASR eval slices against a
manually-transcribed ground truth.

Inputs:
    asr/eval/slice_A_groundtruth.txt   (manually written)
    asr/eval/slice_B_groundtruth.txt   (manually written)
    asr/eval/slice_A_whisper.txt       (auto, from transcribe_eval_slices.py)
    asr/eval/slice_B_whisper.txt       (auto)

Outputs:
    results/asr_evaluation.json
    Markdown table to stdout (paste-ready for the thesis)

Normalization (applied identically to both reference and hypothesis):
    - Strip header lines starting with '#'
    - Collapse newlines + multiple spaces into single space
    - Lowercase
    - Strip punctuation EXCEPT apostrophes (the ground-truth format
      already preserves apostrophes; we drop everything else for fair
      comparison even if Whisper added it)
    - Drop empty lines (blank line in GT marks a non-speech gap)

Run:
    python evaluation/asr_evaluate.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
EVAL_DIR = PROJECT_DIR / "asr" / "eval"
RESULTS_DIR = PROJECT_DIR / "results"

SLICES = [
    ("A", EVAL_DIR / "slice_A_groundtruth.txt", EVAL_DIR / "slice_A_whisper.txt"),
    ("B", EVAL_DIR / "slice_B_groundtruth.txt", EVAL_DIR / "slice_B_whisper.txt"),
]


# ── normalisation ──────────────────────────────────────────────────────

PUNCT_TO_DROP = re.compile(r"[^\w\s']+", flags=re.UNICODE)
WHITESPACE = re.compile(r"\s+")
TIMESTAMP_LINE = re.compile(r"^\s*\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*"
                            r"\d{2}:\d{2}:\d{2}\.\d{3}\]\s*")


def _strip_whisper_timestamp(line: str) -> str:
    """Remove the leading '[HH:MM:SS.mmm --> HH:MM:SS.mmm] ' from Whisper output."""
    return TIMESTAMP_LINE.sub("", line)


def normalize(raw: str, *, is_whisper: bool = False) -> str:
    out_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if is_whisper:
            s = _strip_whisper_timestamp(s)
        out_lines.append(s)
    text = " ".join(out_lines).lower()
    text = PUNCT_TO_DROP.sub(" ", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text


def load_text(path: Path, is_whisper: bool) -> str:
    if not path.exists():
        return ""
    return normalize(path.read_text(encoding="utf-8"), is_whisper=is_whisper)


# ── jiwer wrapper ──────────────────────────────────────────────────────

def compute_metrics(ref: str, hyp: str) -> dict:
    """Returns {wer, cer, mer, wil} as floats (not percentages)."""
    try:
        import jiwer
    except ImportError:
        raise SystemExit(
            "jiwer is not installed.  Run:  pip install jiwer\n"
            "(it is listed in requirements.txt)"
        )
    return {
        "wer": float(jiwer.wer(ref, hyp)),
        "cer": float(jiwer.cer(ref, hyp)),
        "mer": float(jiwer.mer(ref, hyp)),
        "wil": float(jiwer.wil(ref, hyp)),
        "ref_words": len(ref.split()),
        "hyp_words": len(hyp.split()),
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
    }


# ── main ───────────────────────────────────────────────────────────────

def main() -> int:
    per_slice = {}
    refs_combined: list[str] = []
    hyps_combined: list[str] = []
    missing = []

    for letter, gt_path, wh_path in SLICES:
        ref = load_text(gt_path, is_whisper=False)
        hyp = load_text(wh_path, is_whisper=True)

        if not ref:
            missing.append(f"slice_{letter}_groundtruth.txt is empty — "
                           f"manually transcribe it before running this script")
            continue
        if not hyp:
            missing.append(f"slice_{letter}_whisper.txt is empty/missing — "
                           f"run asr/transcribe_eval_slices.py")
            continue

        per_slice[f"slice_{letter}"] = compute_metrics(ref, hyp)
        refs_combined.append(ref)
        hyps_combined.append(hyp)

    if missing:
        print("ERROR — cannot compute metrics yet:")
        for m in missing:
            print(f"  • {m}")
        return 1

    combined = compute_metrics(" ".join(refs_combined), " ".join(hyps_combined))

    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model": "faster-whisper small int8 (English)",
        "slices": per_slice,
        "combined": combined,
        "notes": (
            "WER/CER/MER/WIL computed with jiwer. "
            "Both reference and hypothesis normalised to lowercase, "
            "apostrophes preserved, all other punctuation removed, "
            "comments and timestamps stripped."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "asr_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # ── Markdown table for the thesis ──
    print()
    print("## ASR evaluation — faster-whisper small int8 (English)")
    print()
    print("| Slice | Words (ref) | WER | CER | MER | WIL |")
    print("|---|---:|---:|---:|---:|---:|")
    for letter, _, _ in SLICES:
        key = f"slice_{letter}"
        if key not in per_slice:
            continue
        m = per_slice[key]
        print(f"| {letter} | {m['ref_words']} "
              f"| {m['wer']*100:.1f}% "
              f"| {m['cer']*100:.1f}% "
              f"| {m['mer']*100:.1f}% "
              f"| {m['wil']*100:.1f}% |")
    print(f"| **Combined** | **{combined['ref_words']}** "
          f"| **{combined['wer']*100:.1f}%** "
          f"| **{combined['cer']*100:.1f}%** "
          f"| **{combined['mer']*100:.1f}%** "
          f"| **{combined['wil']*100:.1f}%** |")
    print()
    print(f"Full JSON: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
