"""
Aggregate the 3-run variance experiment for the visual pipeline.

For each (model, run) it loads the summary text from
    llm_summarization/output_cleaned_runs/run_<K>/latest.json
scores it against reference_summary.txt with ROUGE-1/2/L + BERTScore F1
(same metric implementations as llm_summarization/evaluate.py — imported,
not duplicated), and reports per-model mean ± std across the 3 runs.

n is reported explicitly per model. Models that failed on some runs
contribute n<3 with a note. NEVER silently averages over partial data.

Outputs:
    results/variance_report.json   — machine-readable
    results/variance_report.md     — LaTeX-pasteable table

Run AFTER `python evaluation/run_variance_summarization.py`:
    python evaluation/aggregate_variance.py
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LLM_DIR = PROJECT_DIR / "llm_summarization"
RESULTS_DIR = PROJECT_DIR / "results"

RUNS_ROOT = LLM_DIR / "output_cleaned_runs"
REFERENCE_PATH = LLM_DIR / "reference_summary.txt"

METRIC_KEYS = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]

# Reuse llm_summarization's metric implementations
sys.path.insert(0, str(LLM_DIR))
import evaluate as llm_evaluate  # noqa: E402


def collect_runs() -> dict:
    """
    Returns:
        {
          "<display_name>": {
            "runs": [
              {"run": 1, "summary": "...", "latency_s": 4.6, "status": "success"},
              {"run": 2, "summary": "...", "latency_s": 4.7, "status": "success"},
              {"run": 3, "summary": null,  "latency_s": 0.0, "status": "error"},
            ]
          },
          ...
        }
    """
    by_model: dict[str, dict] = {}
    for k in (1, 2, 3):
        latest = RUNS_ROOT / f"run_{k}" / "latest.json"
        if not latest.exists():
            print(f"  [warn] run_{k}/latest.json missing — skipping")
            continue
        data = json.loads(latest.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            name = r.get("display_name", "unknown")
            by_model.setdefault(name, {
                "provider": r.get("provider", ""),
                "model_id": r.get("model_id", ""),
                "runs": [],
            })
            by_model[name]["runs"].append({
                "run": k,
                "status": r.get("status"),
                "summary": r.get("summary"),
                "latency_s": r.get("latency_seconds", 0.0),
            })
    return by_model


def score_one(summary: str, reference: str) -> dict:
    """ROUGE-1/2/L + BERTScore F1 for a single summary vs the reference."""
    rouge = llm_evaluate.compute_rouge(summary, reference)
    bs = llm_evaluate.compute_bertscore([summary], [reference])[0]
    return {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bertscore_f1": bs["f1"],
    }


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = float(statistics.mean(xs))
    s = float(statistics.stdev(xs)) if len(xs) > 1 else 0.0
    return m, s


def main() -> int:
    if not REFERENCE_PATH.exists():
        print(f"ERROR: reference summary missing: {REFERENCE_PATH}")
        return 1
    reference = REFERENCE_PATH.read_text(encoding="utf-8").strip()

    print(f"Reference: {REFERENCE_PATH.name} ({len(reference.split())} words)")
    print(f"Loading runs from {RUNS_ROOT.relative_to(PROJECT_DIR)}/...")
    by_model = collect_runs()
    if not by_model:
        print("ERROR: no run data found.")
        return 2

    print(f"Found {len(by_model)} models across runs.\n")

    # Score every (model, successful run) pair, then aggregate
    print("Scoring (this calls ROUGE + BERTScore per summary, may take a "
          "couple of minutes on first run while BERTScore loads its model)...")

    aggregated = []
    for name, info in by_model.items():
        per_run_scores = []
        statuses = []
        for entry in info["runs"]:
            statuses.append((entry["run"], entry["status"]))
            if entry["status"] != "success" or not entry["summary"]:
                continue
            try:
                scores = score_one(entry["summary"], reference)
            except Exception as e:
                print(f"  [warn] {name} run {entry['run']}: scoring failed ({e})")
                continue
            scores["run"] = entry["run"]
            scores["latency_s"] = entry["latency_s"]
            per_run_scores.append(scores)

        n = len(per_run_scores)
        agg = {
            "model": name,
            "provider": info["provider"],
            "model_id": info["model_id"],
            "n_successful_runs": n,
            "run_statuses": statuses,
            "per_run": per_run_scores,
        }
        for k in METRIC_KEYS:
            vals = [r[k] for r in per_run_scores]
            m, s = mean_std(vals)
            agg[f"{k}_mean"] = round(m, 4)
            agg[f"{k}_std"] = round(s, 4)

        if n < 3:
            agg["note"] = (f"n={n} (of 3) — partial data. Mean computed over "
                           f"{n} successful run{'s' if n != 1 else ''}; std "
                           f"requires n≥2 (else 0).")
        aggregated.append(agg)

        # Console line per model
        m_str = " | ".join(
            f"{k.upper()} {agg[f'{k}_mean']*100:.1f}±{agg[f'{k}_std']*100:.1f}"
            for k in METRIC_KEYS
        )
        print(f"  {name:<32} n={n}  {m_str}")

    # Sort by BERTScore F1 mean, descending
    aggregated.sort(key=lambda r: r["bertscore_f1_mean"], reverse=True)

    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "reference_summary": str(REFERENCE_PATH.relative_to(PROJECT_DIR)),
        "reference_length_words": len(reference.split()),
        "runs_root": str(RUNS_ROOT.relative_to(PROJECT_DIR)),
        "n_runs_target": 3,
        "metrics": ["rouge1", "rouge2", "rougeL", "bertscore_f1"],
        "models": aggregated,
        "notes": (
            "Mean ± std reported only over models that succeeded on ≥1 run. "
            "n is shown explicitly per model. std=0 when n=1 (insufficient "
            "samples for spread). std requires n≥2."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "variance_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    # ── Markdown table (paste-ready for thesis / LaTeX) ──
    md_lines = []
    md_lines.append(f"# Variance report — visual pipeline (3 runs)\n")
    md_lines.append(f"_Generated {report['evaluation_timestamp']}_  ")
    md_lines.append(f"_Reference: `{report['reference_summary']}` "
                    f"({report['reference_length_words']} words)_\n")
    md_lines.append("| Model | n | ROUGE-1 (mean ± std) | ROUGE-2 | ROUGE-L | BERTScore F1 |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for r in aggregated:
        n = r["n_successful_runs"]
        if n == 0:
            md_lines.append(f"| {r['model']} | 0 | — | — | — | — |")
            continue
        cells = []
        for k in METRIC_KEYS:
            m = r[f"{k}_mean"] * 100
            s = r[f"{k}_std"] * 100
            cells.append(f"{m:.1f} ± {s:.1f}")
        md_lines.append(f"| {r['model']} | {n} | "
                        f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    md_lines.append("")
    md_lines.append("_All values shown as **percent**. n=number of successful "
                    "runs out of 3. std requires n≥2 (else reported as 0)._")

    md_path = RESULTS_DIR / "variance_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print("\n".join(md_lines))
    print()
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
