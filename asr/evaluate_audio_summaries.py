"""
Evaluate the AUDIO-PATH 9-LLM summaries against the audio-specific
human reference (`llm_summarization/reference_summary_audio.txt`).

This is a thin wrapper around `llm_summarization/evaluate.py` that:
  1. Monkey-patches `config.OUTPUT_DIR` to point at `llm_summarization/output_asr/`,
     so the eval picks up the audio-path summaries (latest.json) and
     writes its `evaluation_*.json` next to them.
  2. Re-routes argparse defaults via `sys.argv` to use the audio
     reference summary (so the visual reference file is untouched).
  3. After evaluate.main() runs, copies the freshly-written
     `output_asr/evaluation_latest.json` to `results/asr_summary_evaluation.json`
     for symmetry with the other reports under results/.

Does NOT modify llm_summarization/ source — only patches at runtime.

Run AFTER:
  - filling in `llm_summarization/reference_summary_audio.txt`
  - producing audio-path summaries (existing flow:
    `python asr/summarize_transcript.py` → output_asr/latest.json)

Usage:
    python asr/evaluate_audio_summaries.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LLM_DIR = PROJECT_DIR / "llm_summarization"
RESULTS_DIR = PROJECT_DIR / "results"

REFERENCE_AUDIO = LLM_DIR / "reference_summary_audio.txt"
SUMMARIES_AUDIO = LLM_DIR / "output_asr" / "latest.json"
OUTPUT_DIR_AUDIO = LLM_DIR / "output_asr"

# Add llm_summarization/ to path so we can import its evaluate module.
sys.path.insert(0, str(LLM_DIR))


def _validate_inputs() -> bool:
    ok = True
    if not REFERENCE_AUDIO.exists():
        print(f"ERROR: missing audio reference: {REFERENCE_AUDIO}")
        ok = False
    else:
        body = "\n".join(line for line in REFERENCE_AUDIO.read_text(
            encoding="utf-8").splitlines() if not line.startswith("#")).strip()
        if len(body) < 200:
            print(f"ERROR: {REFERENCE_AUDIO.name} appears empty / template-only "
                  f"(got {len(body)} non-comment chars). Fill it in first.")
            ok = False

    if not SUMMARIES_AUDIO.exists():
        print(f"ERROR: missing audio-path summaries: {SUMMARIES_AUDIO}")
        print("       Run: python asr/summarize_transcript.py")
        ok = False
    return ok


def main() -> int:
    if not _validate_inputs():
        return 1

    # Patch evaluate's config to write into the audio output folder
    import config as llm_config  # noqa: E402  (from llm_summarization/)
    orig_output_dir = llm_config.OUTPUT_DIR
    llm_config.OUTPUT_DIR = OUTPUT_DIR_AUDIO

    # Override sys.argv so evaluate.main()'s argparse picks up our paths
    orig_argv = sys.argv[:]
    sys.argv = [
        "evaluate.py",
        "--summaries", str(SUMMARIES_AUDIO),
        "--reference", str(REFERENCE_AUDIO),
    ]

    try:
        import evaluate as llm_evaluate  # noqa: E402
        llm_evaluate.main()
    finally:
        # Restore globals so a subsequent import in the same process
        # doesn't see our patches.
        sys.argv = orig_argv
        llm_config.OUTPUT_DIR = orig_output_dir

    # Mirror the canonical eval report into results/
    src = OUTPUT_DIR_AUDIO / "evaluation_latest.json"
    if not src.exists():
        print(f"ERROR: evaluate.main() did not produce {src}")
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = RESULTS_DIR / "asr_summary_evaluation.json"

    # Annotate the copy with which reference was used
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["evaluation_kind"] = "asr_summary"
    payload["reference_kind"] = "audio_human_reference"
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print(f"\n  Mirrored report -> {dst.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
