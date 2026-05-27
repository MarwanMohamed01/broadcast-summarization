"""
Merge the multiple ASR Level-2 retry files into a single canonical
`output_asr/latest.json` containing every model that ever produced a
successful Level-2 summary.

Precedence (later wins if both succeed):
  1. summaries_<TS>.json           (original full run)
  2. summaries_retry_<TS>.json     (single-call retry)
  3. summaries_retry_chunked_<TS>  (3-batch L2 retry — recovers Groq TPM failures)
  4. summaries_retry_ollama_<TS>   (Ollama L1+L2 backfill)

After merge, this script also re-runs `asr/evaluate_audio_summaries.py`
so the canonical evaluation `results/asr_summary_evaluation.json` and
`output_asr/evaluation_latest.json` reflect the full 9-model set.

Usage:
    python asr/merge_asr_runs.py            # merge + re-evaluate
    python asr/merge_asr_runs.py --no-eval  # merge only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LLM_DIR = PROJECT_DIR / "llm_summarization"
OUTPUT_DIR_ASR = LLM_DIR / "output_asr"

sys.path.insert(0, str(LLM_DIR))
import config as llm_config  # noqa: E402


def _load_runs() -> list[tuple[str, dict]]:
    """Return (label, payload) in precedence order."""
    candidates = [
        ("original", sorted(OUTPUT_DIR_ASR.glob("summaries_2026*.json"))),
        ("retry",    sorted(OUTPUT_DIR_ASR.glob("summaries_retry_2026*.json"))),
        ("chunked",  sorted(OUTPUT_DIR_ASR.glob("summaries_retry_chunked_*.json"))),
        ("ollama",   sorted(OUTPUT_DIR_ASR.glob("summaries_retry_ollama_*.json"))),
    ]
    out: list[tuple[str, dict]] = []
    for label, paths in candidates:
        for p in paths:
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  ! skip {p.name}: {e}")
                continue
            out.append((f"{label}:{p.name}", payload))
    return out


def _best_per_model(runs: list[tuple[str, dict]]) -> dict[str, dict]:
    """For each display name keep the latest successful record; if no
    successful record exists across any run, keep the latest failure.
    """
    best: dict[str, dict] = {}
    best_source: dict[str, str] = {}
    for label, payload in runs:
        for r in payload.get("results", []) or []:
            name = r.get("display_name")
            if not name:
                continue
            cur = best.get(name)
            r_ok = r.get("status") == "success" and (r.get("summary") or "").strip()
            cur_ok = (cur and cur.get("status") == "success"
                      and (cur.get("summary") or "").strip())
            # Later run wins if it's at least as good
            if not cur:
                best[name] = r; best_source[name] = label
            elif r_ok and (not cur_ok or label > best_source[name]):
                best[name] = r; best_source[name] = label
            elif not r_ok and not cur_ok and label > best_source[name]:
                best[name] = r; best_source[name] = label
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the re-evaluation step")
    args = ap.parse_args()

    runs = _load_runs()
    if not runs:
        print("No retry files found.")
        return 1
    print(f"Loaded {len(runs)} run files:")
    for label, _ in runs:
        print(f"  - {label}")

    best = _best_per_model(runs)

    # Re-order to match config.MODELS so output is canonical
    ordered: list[dict] = []
    for _, _, display in llm_config.MODELS:
        if display in best:
            ordered.append(best[display])
    # Anything in best that we didn't recognise (shouldn't happen) trail
    for name, rec in best.items():
        if not any(m[2] == name for m in llm_config.MODELS):
            ordered.append(rec)

    succ = sum(1 for r in ordered if r.get("status") == "success"
               and (r.get("summary") or "").strip())
    print(f"\nMerged set: {len(ordered)} models, {succ} successful Level-2 summaries.")

    merged = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "merged_from": [label for label, _ in runs],
        "models_successful": succ,
        "models_total": len(ordered),
        "results": ordered,
    }
    out_path = OUTPUT_DIR_ASR / "latest.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"Wrote canonical -> {out_path.relative_to(PROJECT_DIR)}")

    if args.no_eval:
        return 0

    # Trigger the existing wrapper, which calls llm_summarization/evaluate.py
    # against reference_summary_audio.txt and mirrors to results/.
    print("\nRe-running evaluation against the audio reference...")
    eval_script = PROJECT_DIR / "asr" / "evaluate_audio_summaries.py"
    r = subprocess.run([sys.executable, str(eval_script)],
                        cwd=str(PROJECT_DIR))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
