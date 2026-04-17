"""
Retry Level-2 (final summarization) for models that have complete Level-1 coverage.

Level-2 previously failed for several models because:
  - Groq TPM rate limits triggered on large batch (59 summaries concatenated)
  - Gemini free tier 429 rate limits
  - HuggingFace "Bad Request" (likely context length / input too long)

Fix: retry with retries+backoff, and accept whichever Level-1 summaries
we have (even partial coverage, 21+/59).

Usage:
    python asr/retry_level2.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LLM_DIR = PROJECT_DIR / "llm_summarization"
sys.path.insert(0, str(LLM_DIR))

import config as llm_config  # noqa: E402
import summarize as llm_summarize  # noqa: E402

OUTPUT_DIR_ASR = LLM_DIR / "output_asr"

# Find the most recent level1 file
level1_files = sorted(OUTPUT_DIR_ASR.glob("level1_*.json"))
if not level1_files:
    raise FileNotFoundError(f"No level1 files in {OUTPUT_DIR_ASR}")
level1_path = level1_files[-1]
print(f"Loading Level-1 intermediate: {level1_path.name}")
data = json.loads(level1_path.read_text(encoding="utf-8"))

LEVEL2_SYSTEM = (
    "You are a professional news editor. You receive a list of "
    "paragraph summaries, each covering 15 minutes of a ~14-hour TV "
    "news broadcast. Write a 5-7 paragraph final summary covering all "
    "the day's major stories and themes in chronological order. Each "
    "paragraph should focus on a distinct topic or time block. Write "
    "in clear neutral journalistic language. Do not add information "
    "not present in the input. Do not use bullet points or headings."
)


def call_with_retries(provider, model_id, display, system, user,
                      max_retries=3, retry_delay=30):
    """Call run_model with automatic retries and backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            result = llm_summarize.run_model(provider, model_id, display, system, user)
            if result["status"] == "success":
                return result
            # If failed, wait and retry
            err = (result.get("error") or "").lower()
            if "429" in err or "rate" in err or "too many" in err or "tpm" in err:
                print(f"    Rate-limited, waiting {retry_delay}s before retry ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            # Non-retryable error
            return result
        except Exception as e:
            print(f"    Exception: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
    return result


level2_results = []
for provider, model_id, display in llm_config.MODELS:
    if display not in data["results"]:
        continue
    chunk_summaries = [r["summary"] for r in data["results"][display]
                       if r["status"] == "success" and r.get("summary")]
    if len(chunk_summaries) < 10:
        print(f"\n{display}: only {len(chunk_summaries)} Level-1 summaries, skipping Level-2")
        continue

    print(f"\n--- {display} ({len(chunk_summaries)}/59 Level-1 OK) ---")
    numbered = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(chunk_summaries))
    user = (
        f"Below are {len(chunk_summaries)} paragraph summaries, each covering "
        f"a 15-minute segment of a ~14-hour news broadcast (in chronological "
        f"order). Write a final 5-7 paragraph summary covering the day's "
        f"major stories and themes.\n\nPARAGRAPH SUMMARIES:\n{numbered}"
    )

    result = call_with_retries(provider, model_id, display, LEVEL2_SYSTEM, user)
    result["level1_coverage"] = f"{len(chunk_summaries)}/59"
    level2_results.append(result)
    time.sleep(2)  # gentle pacing

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_path = OUTPUT_DIR_ASR / f"summaries_retry_{timestamp}.json"
latest_path = OUTPUT_DIR_ASR / "latest.json"
output = {
    "run_timestamp": datetime.now().isoformat(),
    "level1_source": str(level1_path),
    "retry_run": True,
    "results": level2_results,
}
for p in (final_path, latest_path):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print("  RETRY LEVEL-2 RESULTS")
print(f"{'='*70}")
successes = [r for r in level2_results if r["status"] == "success"]
for r in level2_results:
    status = "OK" if r["status"] == "success" else "FAIL"
    n = len(r["summary"]) if r["summary"] else 0
    cov = r.get("level1_coverage", "?")
    print(f"  [{status}] {r['display_name']:30s}  cov={cov:>6}  {r['latency_seconds']:6.1f}s  {n:5d} chars")
print(f"\n  {len(successes)}/{len(level2_results)} models succeeded")
print(f"  Saved: {final_path}")
