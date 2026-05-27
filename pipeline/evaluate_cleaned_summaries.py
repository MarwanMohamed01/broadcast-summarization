"""
Evaluate the 14-hour LLM-cleaned visual summaries against the human
reference (`llm_summarization/reference_summary.txt`).

Thin wrapper around `llm_summarization/evaluate.py`. Mirrors what
`asr/evaluate_audio_summaries.py` does for the audio path:
  1. Monkey-patches `config.OUTPUT_DIR` to point at
     `llm_summarization/output_cleaned/` so the eval picks up the
     cleaned-14h summaries (latest.json) and writes its
     `evaluation_*.json` next to them.
  2. Re-routes argparse defaults to point at the 14h summaries +
     the existing visual reference.
  3. Mirrors the freshly-written `output_cleaned/evaluation_latest.json`
     into `results/visual_14h_summary_evaluation.json` for symmetry
     with the other reports under `results/`.

Does NOT modify llm_summarization/ source — only patches at runtime.

Background — why this exists: the 27-min sample (output/) and the 14h
cleaned ticker (output_cleaned/) are different broadcasts. The
reference_summary.txt was written for the 14h broadcast (Iran/Hormuz
news day), not the 27-min sample (March 13, Saudi / Sudan news day).
The thesis-defensible visual leaderboard is THIS one, not the older
27-min eval.

Run AFTER `python summarize.py` has produced `output_cleaned/latest.json`
(already on disk from 2026-04-11).

Usage:
    python pipeline/evaluate_cleaned_summaries.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LLM_DIR = PROJECT_DIR / "llm_summarization"
RESULTS_DIR = PROJECT_DIR / "results"

REFERENCE_VISUAL = LLM_DIR / "reference_summary.txt"
SUMMARIES_CLEANED = LLM_DIR / "output_cleaned" / "latest.json"
OUTPUT_DIR_CLEANED = LLM_DIR / "output_cleaned"

sys.path.insert(0, str(LLM_DIR))


def _validate_inputs() -> bool:
    ok = True
    if not REFERENCE_VISUAL.exists():
        print(f"ERROR: missing visual reference: {REFERENCE_VISUAL}")
        ok = False
    elif len(REFERENCE_VISUAL.read_text(encoding="utf-8").strip()) < 200:
        print(f"ERROR: {REFERENCE_VISUAL.name} appears empty / too short.")
        ok = False
    if not SUMMARIES_CLEANED.exists():
        print(f"ERROR: missing 14h cleaned summaries: {SUMMARIES_CLEANED}")
        print("       Run: python llm_summarization/summarize.py "
              "(after pointing config.NEWS_ITEMS_PATH at news_items_cleaned.json)")
        ok = False
    return ok


def main() -> int:
    if not _validate_inputs():
        return 1

    import config as llm_config  # noqa: E402  (from llm_summarization/)
    orig_output_dir = llm_config.OUTPUT_DIR
    llm_config.OUTPUT_DIR = OUTPUT_DIR_CLEANED

    orig_argv = sys.argv[:]
    sys.argv = [
        "evaluate.py",
        "--summaries", str(SUMMARIES_CLEANED),
        "--reference", str(REFERENCE_VISUAL),
    ]
    try:
        import evaluate as llm_evaluate  # noqa: E402
        llm_evaluate.main()
    finally:
        sys.argv = orig_argv
        llm_config.OUTPUT_DIR = orig_output_dir

    src = OUTPUT_DIR_CLEANED / "evaluation_latest.json"
    if not src.exists():
        print(f"ERROR: evaluate.main() did not produce {src}")
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = RESULTS_DIR / "visual_14h_summary_evaluation.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["evaluation_kind"] = "visual_summary"
    payload["reference_kind"] = "visual_14h_human_reference"
    payload["news_items_source"] = "ticker_extraction_v6/output/final/news_items_cleaned.json"
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print(f"\n  Mirrored report -> {dst.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
