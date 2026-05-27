"""Open-source VLM extraction + evaluation entry point.

Currently wires the Mistral Ministral 3 14B adapter (open-weights, free
Experiment-plan tier) into the same `extract.run_one_triple` pipeline
Gemini uses, so the comparison is on byte-identical tiles and the
byte-identical extraction prompt (`vlm_extraction/prompts.py`). No
prompt edits, no pipeline redesign.

Usage:
    # full 1845-tile run, idempotent / resume-safe
    python -m vlm_extraction.opensource_vlm run --slice full_video

    # smoke run on a single panorama (~25 tiles)
    python -m vlm_extraction.opensource_vlm run --slice full_video --max-tiles 25

    # rebuild results/vlm_opensource_evaluation.md once a run exists
    python -m vlm_extraction.opensource_vlm evaluate

The runner mirrors `extract.py`'s interface — same VLMResponse dataclass,
same per-tile output schema (`headlines_<tile_id>.json`), same
aggregator (`headlines_combined.json`). The only difference is the
final report file: `results/vlm_opensource_evaluation.md` instead of
`results/vlm_evaluation.md`, with an extra head-to-head table comparing
the open-source VLM against Gemini (existing run) and the OCR pipeline
(from `results/validation_report.json`).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from vlm_extraction import config  # noqa: E402
from vlm_extraction.extract import run_one_triple  # noqa: E402
from vlm_extraction.evaluate import (  # noqa: E402
    evaluate_triple,
    aggregate_runs,
)

# Default open-source VLM registered in `config.MODEL_REGISTRY`. Keep this
# in one place so the runner, evaluator and report all agree.
OPENSOURCE_VLM = "mistral"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPORT_MD = RESULTS_DIR / "vlm_opensource_evaluation.md"
REPORT_JSON = RESULTS_DIR / "vlm_opensource_evaluation.json"

VLM_EVAL_JSON = RESULTS_DIR / "vlm_evaluation.json"
OCR_VALIDATION_JSON = RESULTS_DIR / "validation_report.json"


# ---------------------------------------------------------------- run

def run(slice_name: str, runs: int, max_tiles: int | None,
        force: bool, vlm: str) -> int:
    spec = config.MODEL_REGISTRY.get(vlm)
    if spec is None:
        print(f"Unknown VLM: {vlm}", file=sys.stderr)
        return 2
    if not spec["available"]:
        print(f"VLM '{vlm}' unavailable: {spec['skip_reason']}", file=sys.stderr)
        return 2
    print(f"Open-source VLM extraction: model={spec['model_id']} "
          f"slice={slice_name} runs={runs} max_tiles={max_tiles}")
    for run_idx in range(1, runs + 1):
        run_one_triple(vlm, slice_name, run_idx,
                       force=force, max_tiles=max_tiles)
    return 0


# ---------------------------------------------------------------- eval

def _read_gemini_canonical() -> dict | None:
    """Pull the canonical Gemini full_video / vs slice_A run from
    `results/vlm_evaluation.json` so the comparison table reproduces
    the exact same numbers already reported there.
    """
    if not VLM_EVAL_JSON.exists():
        return None
    payload = json.loads(VLM_EVAL_JSON.read_text(encoding="utf-8"))
    # Prefer the per-run record for (gemini, full_video, run 1, raw, slice_A
    # GT) — that's the row currently quoted in the thesis tables.
    for r in payload.get("per_run", []):
        if (r.get("vlm") == "gemini" and r.get("slice") == "full_video"
                and r.get("run") == 1 and r.get("variant") == "raw"
                and r.get("gt_slice") == "slice_A"):
            return r
    return None


def _ocr_baseline() -> dict:
    """Pull OCR LLM-cleaned numbers from `results/validation_report.json`.

    Fallback to the constants in CLAUDE.md if the file is missing.
    """
    fallback = {"f1_at_70": 0.881, "recall_at_70": 0.923,
                "precision_at_70": 0.842, "cer_at_70": 0.139,
                "time_minutes": 12, "cost_usd": 0.05}
    if not OCR_VALIDATION_JSON.exists():
        return fallback
    try:
        payload = json.loads(OCR_VALIDATION_JSON.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    # Find the cleaned-combined entry. Schema varies — try a few common
    # shapes before giving up.
    for entry in payload if isinstance(payload, list) else payload.get("stages", []):
        if not isinstance(entry, dict):
            continue
        stage = (entry.get("stage") or entry.get("name") or "").lower()
        if "clean" in stage and ("combined" in stage or "overall" in stage
                                  or entry.get("slice") in (None, "combined")):
            t = entry.get("thresholds", {}).get("70", {})
            if t:
                return {
                    "f1_at_70": t.get("f1", fallback["f1_at_70"]),
                    "recall_at_70": t.get("recall", fallback["recall_at_70"]),
                    "precision_at_70": t.get("precision", fallback["precision_at_70"]),
                    "cer_at_70": entry.get("avg_cer_at_70",
                                            fallback["cer_at_70"]),
                    "time_minutes": fallback["time_minutes"],
                    "cost_usd": fallback["cost_usd"],
                }
    return fallback


def _per_run_metrics(vlm: str, slice_name: str) -> list[dict]:
    out: list[dict] = []
    slice_dir = config.OUTPUT_DIR / vlm / slice_name
    if not slice_dir.exists():
        return out
    gt_keys = (("slice_A", "slice_B")
               if slice_name == "full_video"
               else (slice_name,))
    for run_dir in sorted(slice_dir.glob("run_*")):
        try:
            run_idx = int(run_dir.name.split("_")[-1])
        except ValueError:
            continue
        for gt_key in gt_keys:
            m = evaluate_triple(vlm, slice_name, run_idx, gt_slice=gt_key,
                                cleaned=False)
            if m is None:
                continue
            m["vlm"] = vlm
            m["slice"] = slice_name
            m["gt_slice"] = gt_key
            out.append(m)
    return out


def _pct(x) -> str:
    if x is None: return "n/a"
    if isinstance(x, float) and math.isnan(x): return "n/a"
    return f"{x * 100:.1f}%"


def _mean_pct(vs: list[float | None]) -> str:
    vs = [v for v in vs
          if v is not None
          and not (isinstance(v, float) and math.isnan(v))]
    return _pct(statistics.mean(vs)) if vs else "n/a"


def evaluate() -> int:
    vlm = OPENSOURCE_VLM
    per_run = _per_run_metrics(vlm, "full_video") or _per_run_metrics(vlm, "slice_A") \
              + _per_run_metrics(vlm, "slice_B")
    if not per_run:
        print(f"No runs found for vlm='{vlm}'. Run extraction first.",
              file=sys.stderr)
        return 2
    # Re-aggregate across (gt_slice) pairs the same way evaluate.py does.
    by_gt = {}
    for r in per_run:
        by_gt.setdefault(r["gt_slice"], []).append(r)
    aggs = {gt: aggregate_runs(rs) for gt, rs in by_gt.items()}

    gemini_row = _read_gemini_canonical()
    ocr_row = _ocr_baseline()
    spec = config.MODEL_REGISTRY[vlm]

    # Tables ------------------------------------------------------------
    md: list[str] = []
    md.append("# Open-Source VLM Evaluation\n")
    md.append("Auto-generated by `vlm_extraction/opensource_vlm.py evaluate`.\n")
    md.append("Adds an open-weights VLM to the existing Gemini comparison, "
              "using the byte-identical tiles, prompt and aggregator. "
              f"Model: **{spec['model_id']}** "
              f"({'open-weights, ' if not spec['is_paid'] else ''}"
              f"served via {spec['provider']} free tier).\n")

    md.append("\n## Per-run results\n")
    md.append("| Run | GT slice | Extracted | F1@70 | Recall@70 | Precision@70 | CER@70 | Halluc@60 | Wall (s) | Cost (USD) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(per_run, key=lambda x: (x["run"], x["gt_slice"])):
        t70 = r["thresholds"]["70"]
        md.append(f"| {r['run']} | {r['gt_slice']} | {r['extracted_count']} | "
                  f"{_pct(t70['f1'])} | {_pct(t70['recall'])} | "
                  f"{_pct(t70['precision'])} | "
                  f"{_pct(r.get('avg_cer_at_70'))} | "
                  f"{_pct(r.get('hallucination_rate_at_60'))} | "
                  f"{r.get('wall_seconds', float('nan')):.0f} | "
                  f"${r.get('cost_usd', 0):.4f} |")

    md.append("\n## Slice-level aggregate (mean across runs)\n")
    md.append("| GT slice | n_runs | F1 mean ± std | CER mean ± std | Halluc mean ± std |")
    md.append("|---|---|---|---|---|")
    for gt, a in aggs.items():
        f1m, f1s = a["f1_mean"], a["f1_std"]
        cerm, cers = a["cer_mean"], a["cer_std"]
        hm, hs = a["halluc_mean"], a["halluc_std"]
        def pm(m, s):
            if m is None or (isinstance(m, float) and math.isnan(m)):
                return "n/a"
            return f"{m*100:.1f}% ± {s*100:.1f}%"
        md.append(f"| {gt} | {a['n_runs']} | {pm(f1m, f1s)} | "
                  f"{pm(cerm, cers)} | {pm(hm, hs)} |")

    # Head-to-head: Gemini (existing run) vs new open-source VLM vs OCR
    md.append("\n## Comparison: open-source VLM vs Gemini vs OCR pipeline\n")
    md.append("Columns at threshold = 70% fuzzy match. Cost = USD. "
              "Time = wall-clock for the full run. OCR row from "
              "`results/validation_report.json` / CLAUDE.md.\n")
    md.append("| Model | F1 | Recall | Precision | Hallucination Rate | Cost | Time | Notes |")
    md.append("|---|---|---|---|---|---|---|---|")

    # OCR row
    md.append(f"| OCR pipeline (v6 + LLM-cleaned) | "
              f"{_pct(ocr_row['f1_at_70'])} | "
              f"{_pct(ocr_row['recall_at_70'])} | "
              f"{_pct(ocr_row['precision_at_70'])} | "
              f"n/a | $0.05 | ~12 min | "
              f"Apples-to-apples vs 39-headline GT from `validation/ground_truth/` |")

    # Gemini row
    if gemini_row:
        g70 = gemini_row["thresholds"]["70"]
        md.append(f"| Gemini 2.5 Flash (paid) | "
                  f"{_pct(g70['f1'])} | "
                  f"{_pct(g70['recall'])} | "
                  f"{_pct(g70['precision'])} | "
                  f"{_pct(gemini_row.get('hallucination_rate_at_60'))} | "
                  f"${gemini_row.get('cost_usd', 0):.4f} | "
                  f"{gemini_row.get('wall_seconds', 0)/60:.0f} min | "
                  f"Closed-source, scored against {gemini_row['gt_slice']} GT |")
    else:
        md.append("| Gemini 2.5 Flash | n/a | n/a | n/a | n/a | n/a | n/a | "
                  "_results/vlm_evaluation.json missing_ |")

    # New open-source VLM rows (one per GT slice + a mean row)
    n_extracted = per_run[0].get("extracted_count")
    for gt, a in aggs.items():
        # Pull the matching per_run record for recall + precision since
        # aggregate_runs() only carries F1.
        rs = by_gt[gt]
        f1m, f1s = a["f1_mean"], a["f1_std"]
        r_at_70 = statistics.mean(r["thresholds"]["70"]["recall"] for r in rs)
        p_at_70 = statistics.mean(r["thresholds"]["70"]["precision"] for r in rs)
        cerm = a["cer_mean"]
        hm = a["halluc_mean"]
        walls = [r.get("wall_seconds") for r in rs
                 if r.get("wall_seconds") and not math.isnan(r["wall_seconds"])]
        wall_mean = statistics.mean(walls) if walls else float("nan")
        md.append(f"| {spec['model_id']} (open-weights) vs {gt} | "
                  f"{_pct(f1m)} | {_pct(r_at_70)} | {_pct(p_at_70)} | "
                  f"{_pct(hm)} | $0.00 | "
                  f"{wall_mean/60:.0f} min | "
                  f"{n_extracted} extracted; "
                  f"served free via {spec['provider']} |")

    # Mean across GT slices for the open-source VLM
    f1s_all = [a["f1_mean"] for a in aggs.values()]
    cer_all = [a["cer_mean"] for a in aggs.values()
               if a["cer_mean"] is not None
               and not (isinstance(a["cer_mean"], float) and math.isnan(a["cer_mean"]))]
    halluc_all = [a["halluc_mean"] for a in aggs.values()]
    rec_all = [statistics.mean(r["thresholds"]["70"]["recall"] for r in by_gt[gt])
               for gt in aggs]
    prec_all = [statistics.mean(r["thresholds"]["70"]["precision"] for r in by_gt[gt])
                for gt in aggs]
    walls_all = [r.get("wall_seconds") for r in per_run
                 if r.get("wall_seconds") and not math.isnan(r["wall_seconds"])]
    md.append(f"| **{spec['model_id']} (mean across GT slices)** | "
              f"**{_pct(statistics.mean(f1s_all))}** | "
              f"**{_pct(statistics.mean(rec_all))}** | "
              f"**{_pct(statistics.mean(prec_all))}** | "
              f"**{_pct(statistics.mean(halluc_all))}** | "
              f"**$0.00** | "
              f"**{(statistics.mean(walls_all) if walls_all else float('nan'))/60:.0f} min** | "
              f"**Free open-weights baseline** |")

    md.append("\n---")
    md.append(f"_Open-source model: {spec['model_id']}. "
              f"Per-tile output under "
              f"`vlm_extraction/output/{vlm}/<slice>/run_<n>/`._\n")

    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps({
        "model": spec["model_id"],
        "vlm_key": vlm,
        "per_run": per_run,
        "aggregates": aggs,
        "gemini_canonical": gemini_row,
        "ocr_baseline": ocr_row,
    }, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    return 0


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run open-source VLM extraction")
    run_p.add_argument("--slice", default="full_video",
                       choices=list(config.SLICE_BOUNDARIES.keys()))
    run_p.add_argument("--runs", type=int, default=1)
    run_p.add_argument("--max-tiles", type=int, default=None)
    run_p.add_argument("--force", action="store_true")
    run_p.add_argument("--vlm", default=OPENSOURCE_VLM,
                       choices=[k for k, v in config.MODEL_REGISTRY.items()
                                if not v["is_paid"]])

    sub.add_parser("evaluate", help="rebuild results/vlm_opensource_evaluation.md")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return run(args.slice, args.runs, args.max_tiles, args.force, args.vlm)
    if args.cmd == "evaluate":
        return evaluate()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
