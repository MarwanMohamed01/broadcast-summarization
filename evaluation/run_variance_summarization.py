"""
Multi-run variance experiment for the VISUAL pipeline's 9-LLM
summarisation step.

For each of 3 independent runs, calls llm_summarization.summarize.main()
on news_items_cleaned.json and writes the result into
llm_summarization/output_cleaned_runs/run_<K>/.

Resume-safe:
  - A run is considered "complete" when its run_K/latest.json exists AND
    contains a non-empty results array. Already-complete runs are skipped.
  - Inside a run, summarize.main() handles its own per-model errors and
    writes one combined latest.json — partial successes are preserved.

Throttle:
  - Sleeps THROTTLE_BETWEEN_RUNS_S between runs to keep the same model
    well below the per-minute rate limit on every provider's free tier.

Logs:
  - Per-event log to logs/variance_visual.log (timestamps + model lines).

Does NOT modify llm_summarization/ — uses the same monkey-patch pattern
as pipeline/summarize_cleaned.py.

Usage:
    python evaluation/run_variance_summarization.py
    python evaluation/run_variance_summarization.py --runs 3 --start-from 1
    python evaluation/run_variance_summarization.py --force   # redo all runs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LLM_DIR = PROJECT_DIR / "llm_summarization"
LOGS_DIR = PROJECT_DIR / "logs"

CLEANED_INPUT = (
    PROJECT_DIR / "ticker_extraction_v6" / "output" / "final" / "news_items_cleaned.json"
)
RUNS_ROOT = LLM_DIR / "output_cleaned_runs"

THROTTLE_BETWEEN_RUNS_S = 15   # well under any provider's per-minute limit

# Make llm_summarization importable
sys.path.insert(0, str(LLM_DIR))


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "variance_visual.log"
    logger = logging.getLogger("variance.visual")
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


def is_run_complete(run_dir: Path) -> bool:
    latest = run_dir / "latest.json"
    if not latest.exists():
        return False
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("results")) and len(data["results"]) > 0


def execute_run(run_idx: int, log: logging.Logger, force: bool) -> dict:
    run_dir = RUNS_ROOT / f"run_{run_idx}"

    if is_run_complete(run_dir) and not force:
        log.info(f"[skip] run_{run_idx} already complete "
                 f"({(run_dir / 'latest.json').stat().st_size // 1024} KB)")
        return {"run": run_idx, "status": "skipped"}

    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"\n=== RUN {run_idx}/3 — output: {run_dir.relative_to(PROJECT_DIR)} ===")

    # Monkey-patch the (already-imported) config module so summarize.main()
    # writes into our run-specific folder.
    import config as llm_config
    orig_input = llm_config.NEWS_ITEMS_PATH
    orig_output = llm_config.OUTPUT_DIR
    llm_config.NEWS_ITEMS_PATH = CLEANED_INPUT
    llm_config.OUTPUT_DIR = run_dir

    # Override sys.argv so summarize's argparse sees no extra flags
    orig_argv = sys.argv[:]
    sys.argv = ["run_variance_summarization.py"]

    t0 = time.time()
    try:
        # First import triggers module load; subsequent calls reuse it.
        import summarize as llm_summarize  # noqa: E402
        llm_summarize.main()
        status = "ok"
    except Exception as e:
        log.error(f"  run_{run_idx} FAILED: {e}")
        status = "error"
    finally:
        sys.argv = orig_argv
        llm_config.NEWS_ITEMS_PATH = orig_input
        llm_config.OUTPUT_DIR = orig_output

    elapsed = time.time() - t0
    log.info(f"  run_{run_idx} {status} in {elapsed/60:.1f} min")

    # Lightweight per-run summary
    successes = 0
    failures = 0
    if (run_dir / "latest.json").exists():
        try:
            data = json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))
            for r in data.get("results", []):
                if r.get("status") == "success":
                    successes += 1
                else:
                    failures += 1
        except Exception:
            pass
    log.info(f"  models: {successes} OK, {failures} failed")
    return {"run": run_idx, "status": status,
            "successes": successes, "failures": failures,
            "elapsed_minutes": round(elapsed / 60, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of independent runs (default: 3)")
    parser.add_argument("--start-from", type=int, default=1,
                        help="First run index to execute (default: 1)")
    parser.add_argument("--force", action="store_true",
                        help="Redo runs even if their latest.json exists")
    args = parser.parse_args()

    log = setup_logger()

    if not CLEANED_INPUT.exists():
        log.error(f"Missing input: {CLEANED_INPUT}")
        log.error("Run pipeline/clean_news_items.py first.")
        return 1

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    log.info(f"Variance experiment: {args.runs} runs × 9 models = "
             f"{args.runs * 9} model calls")
    log.info(f"Input:  {CLEANED_INPUT.relative_to(PROJECT_DIR)}")
    log.info(f"Output: {RUNS_ROOT.relative_to(PROJECT_DIR)}/run_{{1..{args.runs}}}/")

    summary = []
    for k in range(args.start_from, args.runs + 1):
        info = execute_run(k, log, args.force)
        summary.append(info)
        if k < args.runs:
            log.info(f"  throttling {THROTTLE_BETWEEN_RUNS_S}s before next run...")
            time.sleep(THROTTLE_BETWEEN_RUNS_S)

    log.info("\n=== Variance experiment finished ===")
    for info in summary:
        log.info(f"  run_{info['run']}: {info}")
    log.info(f"Aggregator next: python evaluation/aggregate_variance.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
