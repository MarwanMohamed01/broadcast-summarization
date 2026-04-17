"""
Two-level LLM summarization of a Whisper transcript.

Level 1: each 15-min chunk -> 1 short summary paragraph
Level 2: all Level-1 summaries from one LLM -> final multi-paragraph summary

Uses the same 9 LLMs as llm_summarization/ via monkey-patched imports
(does NOT modify the frozen module). Output goes to a separate folder:
llm_summarization/output_asr/.

Usage:
    python asr/summarize_transcript.py
    python asr/summarize_transcript.py --chunks-dir asr/output/chunks
    python asr/summarize_transcript.py --model 6     (run only model index 6 = Gemini)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LLM_DIR = PROJECT_DIR / "llm_summarization"
CHUNKS_DIR = PROJECT_DIR / "asr" / "output" / "chunks"

sys.path.insert(0, str(LLM_DIR))

# Import the frozen llm_summarization module so we can use its PROVIDERS
import config as llm_config  # noqa: E402
import summarize as llm_summarize  # noqa: E402


OUTPUT_DIR_ASR = LLM_DIR / "output_asr"

# ── Custom prompts for ASR summarization ─────────────

LEVEL1_SYSTEM = (
    "You are a professional news editor. You receive a raw transcript "
    "of a 15-minute segment of a TV news broadcast (produced by "
    "automatic speech recognition). Write a single concise paragraph "
    "(3-5 sentences) summarizing the key stories, events, and claims "
    "discussed in this segment. Write in clear neutral journalistic "
    "language. Do not add information that isn't in the transcript. "
    "Do not use bullet points."
)

LEVEL2_SYSTEM = (
    "You are a professional news editor. You receive a list of "
    "paragraph summaries, each covering 15 minutes of a ~14-hour TV "
    "news broadcast. Write a 5-7 paragraph final summary covering all "
    "the day's major stories and themes in chronological order. Each "
    "paragraph should focus on a distinct topic or time block. Write "
    "in clear neutral journalistic language. Do not add information "
    "not present in the input. Do not use bullet points or headings."
)


def load_chunk(path: Path) -> str:
    """Load a chunk's plain text, stripping the header comment lines."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        lines.append(line)
    return " ".join(lines)


def level1_summarize(text: str, provider: str, model_id: str, display: str) -> dict:
    """Run one LLM on one chunk, returning the summary + metadata."""
    user = (
        f"Below is a 15-minute segment of a TV news broadcast transcript. "
        f"Summarize it in one concise paragraph (3-5 sentences).\n\n"
        f"TRANSCRIPT:\n{text}"
    )
    return llm_summarize.run_model(provider, model_id, display, LEVEL1_SYSTEM, user)


def level2_summarize(paragraphs: list[str], provider: str, model_id: str, display: str) -> dict:
    """Combine all Level-1 summaries from one LLM into a final summary."""
    numbered = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(paragraphs))
    user = (
        f"Below are {len(paragraphs)} paragraph summaries, each covering "
        f"a 15-minute segment of a ~14-hour news broadcast (in chronological "
        f"order). Write a final 5-7 paragraph summary covering the day's "
        f"major stories and themes.\n\nPARAGRAPH SUMMARIES:\n{numbered}"
    )
    return llm_summarize.run_model(provider, model_id, display, LEVEL2_SYSTEM, user)


def main():
    parser = argparse.ArgumentParser(description="ASR transcript two-level LLM summarization")
    parser.add_argument("--chunks-dir", type=str, default=str(CHUNKS_DIR))
    parser.add_argument("--model", type=int, default=None,
                        help="Run only model at this index (default: all 9)")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
    if not chunk_files:
        print(f"ERROR: no chunks in {chunks_dir}. Run chunk_transcript.py first.")
        return

    print(f"Loaded {len(chunk_files)} chunks from {chunks_dir}")

    if args.model is not None:
        models_to_run = [llm_config.MODELS[args.model]]
    else:
        models_to_run = llm_config.MODELS

    OUTPUT_DIR_ASR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Level 1: summarize each chunk with each model ───
    print(f"\n{'='*60}")
    print(f"  LEVEL 1: {len(chunk_files)} chunks x {len(models_to_run)} models")
    print(f"{'='*60}")

    # Structure: { model_display_name: [ {chunk_idx, summary, ...}, ... ] }
    level1_results = {m[2]: [] for m in models_to_run}

    for chunk_idx, chunk_path in enumerate(chunk_files):
        text = load_chunk(chunk_path)
        print(f"\n[chunk {chunk_idx+1}/{len(chunk_files)}] {chunk_path.name} "
              f"({len(text.split())} words)")

        for provider, model_id, display in models_to_run:
            try:
                result = level1_summarize(text, provider, model_id, display)
                level1_results[display].append({
                    "chunk_file": chunk_path.name,
                    "chunk_idx": chunk_idx,
                    "summary": result["summary"],
                    "status": result["status"],
                    "latency_seconds": result["latency_seconds"],
                    "error": result.get("error"),
                })
            except Exception as e:
                print(f"    {display} FAILED: {e}")
                level1_results[display].append({
                    "chunk_file": chunk_path.name,
                    "chunk_idx": chunk_idx,
                    "summary": None,
                    "status": "error",
                    "latency_seconds": 0,
                    "error": str(e),
                })
            # Small sleep to avoid hammering rate-limited providers
            time.sleep(0.5)

    # Save Level-1 intermediate results
    level1_path = OUTPUT_DIR_ASR / f"level1_{timestamp}.json"
    with open(level1_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "chunks": [p.name for p in chunk_files],
            "models": [m[2] for m in models_to_run],
            "results": level1_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Level-1 saved: {level1_path}")

    # ── Level 2: combine Level-1 summaries per model ────
    print(f"\n{'='*60}")
    print(f"  LEVEL 2: final multi-paragraph summary per model")
    print(f"{'='*60}")

    level2_results = []
    for provider, model_id, display in models_to_run:
        chunk_summaries = [r["summary"] for r in level1_results[display]
                           if r["status"] == "success" and r["summary"]]
        if not chunk_summaries:
            print(f"\n  {display}: no successful chunk summaries, skipping")
            continue

        try:
            result = level2_summarize(chunk_summaries, provider, model_id, display)
            level2_results.append(result)
        except Exception as e:
            print(f"  {display} LEVEL-2 FAILED: {e}")

    # Save final output
    final_path = OUTPUT_DIR_ASR / f"summaries_{timestamp}.json"
    latest_path = OUTPUT_DIR_ASR / "latest.json"
    output_data = {
        "run_timestamp": datetime.now().isoformat(),
        "chunks_count": len(chunk_files),
        "chunks_source": str(chunks_dir),
        "level1_intermediate": str(level1_path),
        "level2_system_prompt": LEVEL2_SYSTEM,
        "results": level2_results,
    }

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # ── Summary table ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ASR SUMMARIZATION RESULTS")
    print(f"{'='*70}")
    successes = [r for r in level2_results if r["status"] == "success"]
    for r in level2_results:
        status = "OK" if r["status"] == "success" else "FAIL"
        summary_len = len(r["summary"]) if r["summary"] else 0
        print(f"  [{status}] {r['display_name']:30s} | "
              f"{r['latency_seconds']:6.1f}s | "
              f"{int(r['output_tokens']):4d} tokens | "
              f"{summary_len:5d} chars")
    print(f"\n  {len(successes)}/{len(level2_results)} models succeeded")
    print(f"\n  Final output: {final_path}")
    print(f"  Latest:       {latest_path}")


if __name__ == "__main__":
    main()
