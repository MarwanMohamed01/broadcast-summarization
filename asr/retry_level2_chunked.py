"""
Two-stage Level-2 retry for models whose single-call Level-2 hits
free-tier TPM caps (Groq 8 B / Qwen3 32 B = 6 K TPM, HF Inference 503s).

Idea: instead of one big 12 K-token Level-2 call, split the 59 Level-1
summaries into 3 batches of ~20, summarise each batch (~3 K-token
input → ~600-token output), then summarise those 3 batch-summaries
into the final Level-2. Each call now fits inside the TPM budget for
one minute, and pacing handles the rest.

Targets the models that completed Level-1 fully (59/59) but failed
Level-2 in the original run + the existing single-call retry:
  - Llama 3.1 8B (Groq)
  - Qwen3 32B (Groq)
  - Llama 3 8B (HuggingFace)

Models with complete Level-2 already are skipped. Output is written to
`summaries_retry_chunked_<TS>.json`. The merge step (`merge_asr_runs.py`)
is what assembles the canonical `latest.json` from all retry runs.

Usage:
    python asr/retry_level2_chunked.py
"""

from __future__ import annotations

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

# Only target the 3 cloud models that still fail single-call L2.
TARGETS = {
    "Llama 3.1 8B (Groq)",
    "Qwen3 32B (Groq)",
    "Llama 3 8B (HuggingFace)",
}

MID_SYSTEM = (
    "You are a professional news editor. You receive a list of "
    "paragraph summaries, each covering 15 minutes of a long news "
    "broadcast. Write ONE consolidated paragraph (about 150 words) "
    "summarising the major stories these segments cover. Keep "
    "neutral journalistic tone. Do not invent facts."
)

FINAL_SYSTEM = (
    "You are a professional news editor. You receive a few paragraph "
    "summaries each covering several hours of a ~14-hour TV news "
    "broadcast. Write a 5-7 paragraph final summary covering all the "
    "day's major stories and themes. Use clear neutral journalistic "
    "language. Do not add information not present in the input. Do "
    "not use bullet points or headings."
)

BATCH_SIZE = 20            # 59 L1 → 3 batches (20/20/19)
INTER_CALL_PAUSE = 4.0     # seconds between sub-calls — keeps TPM under cap


def _load_level1() -> tuple[Path, dict]:
    files = sorted(OUTPUT_DIR_ASR.glob("level1_*.json"))
    if not files:
        raise FileNotFoundError(f"No level1 files in {OUTPUT_DIR_ASR}")
    p = files[-1]
    print(f"Loading Level-1: {p.name}")
    return p, json.loads(p.read_text(encoding="utf-8"))


def _call_with_backoff(provider, model_id, display, system, user,
                        max_retries=4, base_delay=20.0):
    delay = base_delay
    last = None
    for attempt in range(1, max_retries + 1):
        last = llm_summarize.run_model(provider, model_id, display, system, user)
        if last["status"] == "success":
            return last
        err = (last.get("error") or "").lower()
        if any(s in err for s in ("429", "rate", "tpm", "too many", "503")):
            print(f"      ! rate-limited / 503 on attempt {attempt} — sleeping {delay:.0f}s")
            time.sleep(delay)
            delay *= 1.7
            continue
        # Non-retryable
        return last
    return last


def main() -> int:
    level1_path, l1 = _load_level1()
    out_records: list[dict] = []

    for provider, model_id, display in llm_config.MODELS:
        if display not in TARGETS:
            continue
        l1_results = l1["results"].get(display, []) or []
        chunk_summaries = [
            r.get("summary", "") for r in l1_results
            if isinstance(r, dict) and r.get("status") == "success"
            and (r.get("summary") or "").strip()
        ]
        if len(chunk_summaries) < 20:
            print(f"\n[skip] {display}: only {len(chunk_summaries)} L1 done, "
                   "below the 20-summary threshold for chunked L2.")
            continue

        print(f"\n=== {display} ({len(chunk_summaries)} L1 summaries) ===")
        batches: list[list[str]] = [
            chunk_summaries[i:i + BATCH_SIZE]
            for i in range(0, len(chunk_summaries), BATCH_SIZE)
        ]
        print(f"  splitting into {len(batches)} batches of "
              f"{[len(b) for b in batches]}")

        # Stage A — one paragraph per batch.
        mid_summaries: list[str] = []
        a_started = time.time()
        for i, batch in enumerate(batches, 1):
            numbered = "\n\n".join(f"[{j+1}] {p}" for j, p in enumerate(batch))
            user = (
                f"Below are {len(batch)} paragraph summaries, each covering "
                f"a 15-minute segment of a TV news broadcast (chronological). "
                f"Write ONE consolidated paragraph covering the major stories.\n\n"
                f"PARAGRAPH SUMMARIES:\n{numbered}"
            )
            res = _call_with_backoff(provider, model_id, display,
                                      MID_SYSTEM, user)
            if res["status"] != "success" or not res.get("summary"):
                print(f"   [stage-A {i}/{len(batches)}] FAIL: "
                       f"{(res.get('error') or 'no summary')[:120]}")
                break
            mid_summaries.append(res["summary"].strip())
            print(f"   [stage-A {i}/{len(batches)}] ok "
                   f"({len(res['summary'].split())} words, "
                   f"{res.get('latency_seconds', 0):.1f}s)")
            time.sleep(INTER_CALL_PAUSE)
        a_dur = time.time() - a_started

        if len(mid_summaries) != len(batches):
            out_records.append({
                "provider": provider,
                "model_id": model_id,
                "display_name": display,
                "status": "error",
                "error": "stage-A incomplete",
                "summary": "",
                "latency_seconds": a_dur,
                "level1_coverage": f"{len(chunk_summaries)}/59",
            })
            continue

        # Stage B — collapse the mid-summaries into the final L2.
        numbered_mid = "\n\n".join(f"[{j+1}] {p}" for j, p in enumerate(mid_summaries))
        user = (
            f"Below are {len(mid_summaries)} paragraph summaries, each "
            f"covering several hours of a ~14-hour TV news broadcast "
            f"(chronological). Write a final 5-7 paragraph summary "
            f"covering the day's major stories and themes.\n\n"
            f"SUMMARIES:\n{numbered_mid}"
        )
        b_started = time.time()
        final = _call_with_backoff(provider, model_id, display,
                                    FINAL_SYSTEM, user)
        b_dur = time.time() - b_started

        if final["status"] == "success" and final.get("summary"):
            print(f"   [stage-B] ok "
                   f"({len(final['summary'].split())} words, "
                   f"{final.get('latency_seconds', 0):.1f}s)")
            final["latency_seconds"] = round(a_dur + b_dur, 2)
            final["level1_coverage"] = f"{len(chunk_summaries)}/59"
            final["strategy"] = f"chunked_L2_{len(batches)}batches"
            out_records.append(final)
        else:
            print(f"   [stage-B] FAIL: "
                   f"{(final.get('error') or 'no summary')[:120]}")
            out_records.append({
                "provider": provider,
                "model_id": model_id,
                "display_name": display,
                "status": "error",
                "error": final.get("error", "stage-B failed"),
                "summary": "",
                "latency_seconds": round(a_dur + b_dur, 2),
                "level1_coverage": f"{len(chunk_summaries)}/59",
            })

    if not out_records:
        print("\nNothing to retry.")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR_ASR / f"summaries_retry_chunked_{ts}.json"
    out_path.write_text(json.dumps({
        "run_timestamp": datetime.now().isoformat(),
        "level1_source": str(level1_path),
        "strategy": "chunked_level2",
        "results": out_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  CHUNKED-L2 RETRY DONE  ({len([r for r in out_records if r['status']=='success'])}/{len(out_records)} ok)")
    print("=" * 70)
    for r in out_records:
        flag = "OK  " if r["status"] == "success" else "FAIL"
        n = len((r.get("summary") or "").split())
        print(f"  [{flag}] {r['display_name']:32s} {n:5d} words  "
              f"{r.get('latency_seconds', 0):6.1f}s")
    print(f"\nSaved -> {out_path.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
