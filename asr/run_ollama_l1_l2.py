"""
Resume-safe Level-1 + Level-2 Ollama backfill.

Run order:
  1. Read the canonical Level-1 file (level1_<TS>.json). Sparse entries
     for the two Ollama models (Llama 3.2 3B has 1/59 OK, Llama 3.1 8B
     has 0/59) are the ones we need to fill.
  2. For each missing chunk, call Ollama via summarize.call_ollama with
     a 600 s read timeout (CPU model loading is the slow part).
  3. Save progress to `level1_ollama_inflight.json` every 5 chunks so a
     crash / reboot doesn't lose work. On restart, only chunks still
     status=error or missing are retried.
  4. After all chunks for a model are done, run Level-2 once on the
     combined L1 summaries (no chunking needed at L2 because Ollama is
     local — there's no TPM cap).
  5. Emit `summaries_retry_ollama_<TS>.json` for the merge step.

Estimated wall time on commodity CPU:
  - Llama 3.2 3B (2 GB):  ~2.5 h  (≈ 150 s per chunk × 58 left)
  - Llama 3.1 8B (4.9 GB): ~7 h    (≈ 400 s per chunk × 59 left)
  Total: ~9-10 hours overnight.

Usage:
    python asr/run_ollama_l1_l2.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LLM_DIR = PROJECT_DIR / "llm_summarization"
CHUNKS_DIR = PROJECT_DIR / "asr" / "output" / "chunks"
OUTPUT_DIR_ASR = LLM_DIR / "output_asr"
INFLIGHT_PATH = OUTPUT_DIR_ASR / "level1_ollama_inflight.json"

sys.path.insert(0, str(LLM_DIR))
import config as llm_config  # noqa: E402
import summarize as llm_summarize  # noqa: E402
import requests  # noqa: E402

# Override the default 300 s timeout used by summarize.call_ollama —
# cold-load + 3 K-token input on a CPU 8 B model can take 600-900 s.
OLLAMA_READ_TIMEOUT = 1200


def _ollama_call(model_id: str, system: str, user: str,
                 timeout: int = OLLAMA_READ_TIMEOUT) -> tuple[str, dict]:
    r = requests.post(
        f"{llm_config.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"], {
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
    }


def _warmup(model_id: str) -> None:
    """Force Ollama to load the model into memory so the first real call
    doesn't include cold-load time in its budget."""
    print(f"  warmup {model_id} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        requests.post(
            f"{llm_config.OLLAMA_BASE_URL}/api/generate",
            json={"model": model_id, "prompt": "warmup", "stream": False,
                  "options": {"num_predict": 1}},
            timeout=300,
        ).raise_for_status()
        print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", flush=True)

OLLAMA_MODELS = [
    ("ollama", "llama3.2",      "Llama 3.2 3B (Ollama local)"),
    ("ollama", "llama3.1:8b",   "Llama 3.1 8B (Ollama local)"),
]

L1_SYSTEM = (
    "You are a professional news editor. You receive ~15 minutes of a "
    "TV news broadcast transcript. Summarise it in ONE paragraph "
    "(about 80-100 words) covering the major stories. Use neutral "
    "journalistic language. Do not add facts not in the input."
)
L2_SYSTEM = (
    "You are a professional news editor. You receive a list of "
    "paragraph summaries, each covering 15 minutes of a ~14-hour TV "
    "news broadcast. Write a 5-7 paragraph final summary covering all "
    "the day's major stories and themes in chronological order. Use "
    "clear neutral journalistic language. Do not add facts. No "
    "bullet points or headings."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_canonical_l1() -> tuple[Path, dict]:
    files = sorted(OUTPUT_DIR_ASR.glob("level1_2026*.json"))
    if not files:
        raise FileNotFoundError(f"no level1 file in {OUTPUT_DIR_ASR}")
    p = files[-1]
    return p, json.loads(p.read_text(encoding="utf-8"))


def _load_inflight(seed: dict) -> dict:
    """Return the inflight state, seeding from canonical L1 on first run."""
    if INFLIGHT_PATH.exists():
        return json.loads(INFLIGHT_PATH.read_text(encoding="utf-8"))
    # First run: copy just the two Ollama entries (and chunks list) into
    # a fresh inflight file so we never touch the canonical level1 file.
    inflight = {
        "started_at": _now(),
        "chunks": seed["chunks"],
        "results": {
            display: [dict(e) if isinstance(e, dict) else
                      {"chunk_file": seed["chunks"][i],
                       "chunk_idx": i, "status": "error", "summary": ""}
                      for i, e in enumerate(seed["results"].get(display, []))]
            for _, _, display in OLLAMA_MODELS
        },
    }
    return inflight


def _save_inflight(state: dict) -> None:
    INFLIGHT_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                              encoding="utf-8")


def _missing_chunks(state: dict, display: str, total: int) -> list[int]:
    entries = state["results"].get(display, [])
    out: list[int] = []
    for i in range(total):
        if i >= len(entries):
            out.append(i); continue
        e = entries[i]
        if not isinstance(e, dict) or e.get("status") != "success" \
                or not (e.get("summary") or "").strip():
            out.append(i)
    return out


def _run_l1(state: dict, provider: str, model_id: str, display: str
            ) -> None:
    total = len(state["chunks"])
    todo = _missing_chunks(state, display, total)
    print(f"\n=== {display}: {total - len(todo)}/{total} already done, "
          f"{len(todo)} to fill ===", flush=True)
    save_every = 5
    for idx, chunk_idx in enumerate(todo, 1):
        chunk_file = state["chunks"][chunk_idx]
        chunk_path = CHUNKS_DIR / chunk_file
        chunk_text = chunk_path.read_text(encoding="utf-8")
        # Drop header comment lines so the model gets clean prose.
        chunk_text = "\n".join(
            ln for ln in chunk_text.splitlines() if not ln.startswith("#")
        ).strip()
        user = (
            "Summarise the following 15-minute news transcript in ONE "
            "paragraph (80-100 words) covering the main stories.\n\n"
            f"TRANSCRIPT:\n{chunk_text}"
        )
        t0 = time.time()
        try:
            summary, meta = _ollama_call(model_id, L1_SYSTEM, user)
            dt = time.time() - t0
            entry = {
                "chunk_file": chunk_file,
                "chunk_idx": chunk_idx,
                "summary": summary.strip(),
                "status": "success" if summary.strip() else "error",
                "error": None,
                "latency_seconds": round(dt, 2),
                "input_tokens": meta.get("input_tokens", 0),
                "output_tokens": meta.get("output_tokens", 0),
            }
            print(f"  [{idx:2d}/{len(todo)}] chunk_{chunk_idx:03d} "
                   f"{entry['status']:>7s}  {dt:5.0f}s  "
                   f"{len(summary.split()):4d} words", flush=True)
        except Exception as exc:
            dt = time.time() - t0
            entry = {
                "chunk_file": chunk_file,
                "chunk_idx": chunk_idx,
                "summary": "",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_seconds": round(dt, 2),
            }
            print(f"  [{idx:2d}/{len(todo)}] chunk_{chunk_idx:03d}    FAIL  "
                   f"{dt:5.0f}s  {type(exc).__name__}", flush=True)

        # update in-place
        entries = state["results"][display]
        while len(entries) <= chunk_idx:
            entries.append({})
        entries[chunk_idx] = entry

        if idx % save_every == 0 or idx == len(todo):
            _save_inflight(state)


def _run_l2(state: dict, provider: str, model_id: str, display: str
            ) -> dict:
    entries = state["results"].get(display, [])
    paras = [e["summary"] for e in entries
             if isinstance(e, dict) and e.get("status") == "success"
             and (e.get("summary") or "").strip()]
    if len(paras) < 10:
        print(f"\n  [{display}] only {len(paras)} L1 done, skipping L2")
        return {
            "provider": provider, "model_id": model_id,
            "display_name": display, "status": "error",
            "summary": "", "error": f"only {len(paras)} L1 summaries",
            "latency_seconds": 0,
        }
    print(f"\n  [{display}] L2 over {len(paras)} L1 paragraphs", flush=True)
    numbered = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(paras))
    user = (
        f"Below are {len(paras)} paragraph summaries, each covering a "
        f"15-minute segment of a ~14-hour news broadcast (chronological). "
        f"Write a final 5-7 paragraph summary covering the day's major "
        f"stories and themes.\n\nPARAGRAPH SUMMARIES:\n{numbered}"
    )
    t0 = time.time()
    try:
        summary, meta = _ollama_call(model_id, L2_SYSTEM, user)
        dt = time.time() - t0
        print(f"   ok  {dt:.0f}s  {len(summary.split())} words", flush=True)
        return {
            "provider": provider, "model_id": model_id,
            "display_name": display, "status": "success",
            "summary": summary.strip(),
            "error": None,
            "latency_seconds": round(dt, 2),
            "level1_coverage": f"{len(paras)}/{len(state['chunks'])}",
            "input_tokens": meta.get("input_tokens", 0),
            "output_tokens": meta.get("output_tokens", 0),
        }
    except Exception as exc:
        dt = time.time() - t0
        return {
            "provider": provider, "model_id": model_id,
            "display_name": display, "status": "error",
            "summary": "",
            "error": f"{type(exc).__name__}: {exc}",
            "latency_seconds": round(dt, 2),
        }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2-only", action="store_true",
                    help="skip L1 entirely; run L2 over whatever the "
                         "inflight file already has (use after an "
                         "8B-style L1 abort).")
    args = ap.parse_args()

    canon_path, canon = _load_canonical_l1()
    print(f"Seeded from {canon_path.name}")
    state = _load_inflight(canon)
    print(f"Inflight: {INFLIGHT_PATH.name}")

    overall_start = time.time()
    if not args.l2_only:
        for provider, model_id, display in OLLAMA_MODELS:
            _warmup(model_id)
            try:
                _run_l1(state, provider, model_id, display)
            except Exception as exc:
                print(f"\n!! L1 loop crashed for {display}: {exc}", flush=True)
                _save_inflight(state)
    else:
        # L2 path still needs the model loaded.
        for _, model_id, _ in OLLAMA_MODELS:
            _warmup(model_id)

    # All L1 work persisted — now do L2 per model.
    l2_records = []
    for provider, model_id, display in OLLAMA_MODELS:
        l2_records.append(_run_l2(state, provider, model_id, display))

    # Save L2 retry file in the same shape as the chunked retry script.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR_ASR / f"summaries_retry_ollama_{ts}.json"
    out_path.write_text(json.dumps({
        "run_timestamp": _now(),
        "level1_source": str(INFLIGHT_PATH),
        "strategy": "ollama_local_l1_l2",
        "results": l2_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    dur = (time.time() - overall_start) / 60
    print(f"\n{'='*70}\n  OLLAMA L1+L2 DONE  ({dur:.0f} min total)\n{'='*70}")
    for r in l2_records:
        flag = "OK  " if r["status"] == "success" else "FAIL"
        n = len((r.get("summary") or "").split())
        print(f"  [{flag}] {r['display_name']:32s} {n:5d} words  "
              f"{r.get('latency_seconds', 0):6.1f}s  "
              f"cov={r.get('level1_coverage', '?')}")
    print(f"\nSaved -> {out_path.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
