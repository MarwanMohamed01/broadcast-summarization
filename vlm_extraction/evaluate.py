"""Evaluate VLM extraction quality.

Reuses validation/validate_extraction.py helpers (import, do not copy).
For each (vlm, slice, run) computes P/R/F1 at 60/70/80 thresholds, CER,
hallucination rate, and rolls up mean ± std across the 3 runs.

Outputs:
    results/vlm_evaluation.json  (full machine-readable)
    results/vlm_evaluation.md    (paste-ready Markdown for thesis)
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

# Project root needs to be importable so `import validation.validate_extraction`
# works when this script is run as `python -m vlm_extraction.evaluate`.
PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from validation.validate_extraction import (  # noqa: E402
    best_match,
    evaluate_stage,
    load_ground_truth,
)
from vlm_extraction import config  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
JSON_OUT = RESULTS_DIR / "vlm_evaluation.json"
MD_OUT = RESULTS_DIR / "vlm_evaluation.md"

HALLUC_THRESHOLD = 60  # below this = "model invented this headline"


def _load_run_headlines(vlm: str, slice_name: str, run_idx: int,
                        *, cleaned: bool = False) -> list[str]:
    name = ("headlines_combined_cleaned.json" if cleaned
            else "headlines_combined.json")
    path = config.OUTPUT_DIR / vlm / slice_name / f"run_{run_idx}" / name
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    return [it["text"] for it in items if it.get("text")]


def _load_run_cost_time(vlm: str, slice_name: str, run_idx: int
                        ) -> tuple[float, float]:
    base = config.OUTPUT_DIR / vlm / slice_name / f"run_{run_idx}"
    timing = base / "timing.json"
    cost = base / "cost.json"
    wall = json.loads(timing.read_text("utf-8"))["total_wall_seconds"] \
        if timing.exists() else float("nan")
    usd = json.loads(cost.read_text("utf-8"))["total_usd"] \
        if cost.exists() else float("nan")
    return wall, usd


def _hallucination_rate(extracted: list[str], gt: list[str]) -> float:
    if not extracted:
        return 0.0
    halluc = 0
    for e in extracted:
        _, score = best_match(e, gt)
        if score < HALLUC_THRESHOLD:
            halluc += 1
    return round(halluc / len(extracted), 4)


def evaluate_triple(vlm: str, slice_name: str, run_idx: int,
                    *, gt_slice: str | None = None,
                    cleaned: bool = False) -> dict | None:
    """Evaluate a single (vlm, slice, run) directory.

    For slice_name == 'full_video' we compare the same global VLM
    headlines list against EACH gt slice separately by passing
    ``gt_slice``. For 'slice_A' / 'slice_B' the GT defaults to that slice.
    Pass ``cleaned=True`` to score the cleaned (post-LLM-pass) output
    instead of the raw aggregated output.
    """
    extracted = _load_run_headlines(vlm, slice_name, run_idx, cleaned=cleaned)
    if not extracted:
        return None
    gt_key = gt_slice or slice_name
    gt_path = config.GROUND_TRUTH_HEADLINES[gt_key]
    gt = load_ground_truth(gt_path)
    metrics = evaluate_stage(extracted, gt, vlm, gt_key)
    metrics["hallucination_rate_at_60"] = _hallucination_rate(extracted, gt)
    wall, cost = _load_run_cost_time(vlm, slice_name, run_idx)
    metrics["wall_seconds"] = wall
    metrics["cost_usd"] = cost
    metrics["run"] = run_idx
    metrics["source_slice"] = slice_name
    metrics["variant"] = "cleaned" if cleaned else "raw"
    return metrics


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def aggregate_runs(per_run: list[dict]) -> dict:
    f1s = [r["thresholds"]["70"]["f1"] for r in per_run]
    cers = [r["avg_cer_at_70"] for r in per_run]
    halluc = [r["hallucination_rate_at_60"] for r in per_run]
    walls = [r["wall_seconds"] for r in per_run]
    costs = [r["cost_usd"] for r in per_run]
    f1_m, f1_s = _mean_std(f1s)
    cer_m, cer_s = _mean_std(cers)
    h_m, h_s = _mean_std(halluc)
    w_m, _ = _mean_std(walls)
    c_m, _ = _mean_std(costs)
    return {
        "f1_mean": f1_m, "f1_std": f1_s,
        "cer_mean": cer_m, "cer_std": cer_s,
        "halluc_mean": h_m, "halluc_std": h_s,
        "wall_mean": w_m, "cost_mean": c_m,
        "n_runs": len(per_run),
    }


def _fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.1f}%" if x <= 1.0 else f"{x:.1f}%"


def _fmt_pm(mean: float, std: float, *, pct: bool = True) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    if pct:
        return f"{mean*100:.1f}% ± {std*100:.1f}%"
    return f"{mean:.3f} ± {std:.3f}"


def build_report() -> dict:
    """Walk every (vlm, slice, run) directory that exists; aggregate.

    For ``slice_name == 'full_video'`` the same headlines are scored
    against slice_A and slice_B GT separately and bucketed under
    ``full_video|slice_A`` and ``full_video|slice_B`` so Tables 1-3
    can show both numbers cleanly.
    """
    report = {"per_run": [], "by_vlm_slice": {}, "by_vlm": {}}
    for vlm in config.MODEL_REGISTRY:
        for slice_name in config.SLICE_BOUNDARIES:
            slice_dir = config.OUTPUT_DIR / vlm / slice_name
            if not slice_dir.exists():
                continue
            # For full_video, pivot against both GT slices.
            gt_keys = (("slice_A", "slice_B")
                       if slice_name == "full_video"
                       else (slice_name,))
            for gt_key in gt_keys:
                # Score both raw and cleaned variants (if cleaned exists).
                for variant_cleaned in (False, True):
                    suffix = ("_cleaned" if variant_cleaned else "")
                    bucket_base = (f"{vlm}|full_video__vs__{gt_key}"
                                   if slice_name == "full_video"
                                   else f"{vlm}|{slice_name}")
                    bucket = bucket_base + suffix
                    run_metrics = []
                    for run_dir in sorted(slice_dir.glob("run_*")):
                        try:
                            run_idx = int(run_dir.name.split("_")[-1])
                        except ValueError:
                            continue
                        m = evaluate_triple(vlm, slice_name, run_idx,
                                            gt_slice=gt_key,
                                            cleaned=variant_cleaned)
                        if m is None:
                            continue
                        report["per_run"].append({**m, "vlm": vlm,
                                                  "slice": slice_name,
                                                  "gt_slice": gt_key})
                        run_metrics.append(m)
                    if run_metrics:
                        report["by_vlm_slice"][bucket] = aggregate_runs(run_metrics)
    # Roll up across slices for each vlm.
    for vlm in config.MODEL_REGISTRY:
        slice_aggs = [v for k, v in report["by_vlm_slice"].items()
                      if k.startswith(f"{vlm}|")]
        if not slice_aggs:
            continue
        f1s = [a["f1_mean"] for a in slice_aggs]
        cers = [a["cer_mean"] for a in slice_aggs if a["cer_mean"] is not None
                and not (isinstance(a["cer_mean"], float) and math.isnan(a["cer_mean"]))]
        halluc = [a["halluc_mean"] for a in slice_aggs]
        walls = [a["wall_mean"] for a in slice_aggs]
        costs = [a["cost_mean"] for a in slice_aggs]
        report["by_vlm"][vlm] = {
            "mean_f1": _mean_std(f1s)[0],
            "mean_cer": _mean_std(cers)[0] if cers else None,
            "mean_halluc": _mean_std(halluc)[0],
            "mean_wall": _mean_std(walls)[0],
            "mean_cost": _mean_std(costs)[0],
            "n_slices": len(slice_aggs),
        }
    return report


# ----- Markdown formatting --------------------------------------------------
def _row_table1(vlm: str, report: dict) -> str:
    sa = report["by_vlm_slice"].get(f"{vlm}|slice_A")
    sb = report["by_vlm_slice"].get(f"{vlm}|slice_B")
    overall = report["by_vlm"].get(vlm)
    if not overall:
        deferred = config.MODEL_REGISTRY[vlm]["skip_reason"] or "deferred"
        return f"| {vlm} | n/a | n/a | n/a | n/a | n/a | _{deferred}_ |"

    def f1c(x):
        if x is None: return "n/a"
        return f"{x*100:.1f}%"

    return (f"| {vlm} | {f1c(sa and sa['f1_mean'])} | "
            f"{f1c(sb and sb['f1_mean'])} | "
            f"{f1c(overall['mean_f1'])} | "
            f"{_fmt_pct(overall['mean_cer'])} | "
            f"{_fmt_pct(overall['mean_halluc'])} | |")


def _ocr_baselines() -> list[tuple[str, str, str, str, str]]:
    """OCR baseline rows for Table 2.

    Pulled from validated numbers in CLAUDE.md (`88.1 %` cleaned F1,
    `23.4 %` raw CER, etc.). These are constants of record from prior
    evaluation, not recomputed here.
    """
    return [
        ("OCR (v6 raw)", "84.8%", "23.4%", "~12 min", "$0.00"),
        ("OCR (LLM-cleaned)", "88.1%", "13.9%", "~12 min + cleanup", "~$0.05"),
    ]


def _agg_for(report: dict, vlm: str, gt_key: str, cleaned: bool) -> dict | None:
    suffix = "_cleaned" if cleaned else ""
    keys = [f"{vlm}|full_video__vs__{gt_key}{suffix}",
            f"{vlm}|{gt_key}{suffix}"]
    for k in keys:
        if k in report["by_vlm_slice"]:
            return report["by_vlm_slice"][k]
    return None


def _mean_pair(a: dict | None, b: dict | None, field: str) -> float | None:
    vals = []
    for d in (a, b):
        if d and d.get(field) is not None:
            v = d[field]
            if isinstance(v, float) and math.isnan(v):
                continue
            vals.append(v)
    return statistics.mean(vals) if vals else None


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# VLM Extraction Evaluation\n")
    lines.append("Auto-generated by `vlm_extraction/evaluate.py`. ")
    lines.append("Source: per-run output under `vlm_extraction/output/<vlm>/<slice>/run_<n>/`.\n")
    lines.append("Each VLM row appears twice: raw aggregator output, and the same output "
                 "after the `pipeline/clean_vlm_headlines.py` LLM-cleanup pass — directly "
                 "analogous to OCR raw vs OCR LLM-cleaned.\n")

    def _pct(x):
        if x is None or (isinstance(x, float) and math.isnan(x)): return "n/a"
        return f"{x*100:.1f}%"

    # Table 1 — per-VLM extraction quality, raw + cleaned side by side
    lines.append("\n## Table 1 — Per-VLM extraction quality\n")
    lines.append("| VLM | Variant | Slice A F1 | Slice B F1 | Mean F1 | Mean CER | Halluc rate | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for vlm in config.MODEL_REGISTRY:
        spec = config.MODEL_REGISTRY[vlm]
        if not spec["available"]:
            note = spec["skip_reason"] or "deferred"
            lines.append(f"| {vlm} | — | n/a | n/a | n/a | n/a | n/a | _{note}_ |")
            continue
        for variant_label, cleaned_flag in (("raw", False), ("cleaned", True)):
            sa = _agg_for(report, vlm, "slice_A", cleaned_flag)
            sb = _agg_for(report, vlm, "slice_B", cleaned_flag)
            if sa is None and sb is None:
                lines.append(f"| {vlm} | {variant_label} | n/a | n/a | n/a | n/a | n/a | "
                             f"_no `_cleaned` output yet_ |"
                             if variant_label == "cleaned" else
                             f"| {vlm} | {variant_label} | n/a | n/a | n/a | n/a | n/a | |")
                continue
            mean_f1 = _mean_pair(sa, sb, "f1_mean")
            mean_cer = _mean_pair(sa, sb, "cer_mean")
            mean_h = _mean_pair(sa, sb, "halluc_mean")
            lines.append(f"| {vlm} | {variant_label} | "
                         f"{_pct(sa and sa['f1_mean'])} | "
                         f"{_pct(sb and sb['f1_mean'])} | "
                         f"{_pct(mean_f1)} | "
                         f"{_pct(mean_cer)} | "
                         f"{_pct(mean_h)} | |")

    # Table 2 — VLM vs OCR head-to-head
    lines.append("\n## Table 2 — VLM vs. OCR head-to-head\n")
    lines.append("| Method | Mean F1 | Mean CER | Time | Cost |")
    lines.append("|---|---|---|---|---|")
    for name, f1, cer, t, c in _ocr_baselines():
        lines.append(f"| {name} | {f1} | {cer} | {t} | {c} |")
    for vlm in config.MODEL_REGISTRY:
        spec = config.MODEL_REGISTRY[vlm]
        if not spec["available"]:
            continue
        for variant_label, cleaned_flag in (("raw", False), ("cleaned", True)):
            sa = _agg_for(report, vlm, "slice_A", cleaned_flag)
            sb = _agg_for(report, vlm, "slice_B", cleaned_flag)
            if sa is None and sb is None:
                continue
            mean_f1 = _mean_pair(sa, sb, "f1_mean")
            mean_cer = _mean_pair(sa, sb, "cer_mean")
            wall = _mean_pair(sa, sb, "wall_mean")
            cost = _mean_pair(sa, sb, "cost_mean")
            wall_s = (f"{wall/60:.0f} min" if wall and wall > 60
                      else (f"{wall:.0f}s" if wall else "n/a"))
            cost_s = (f"${cost:.4f}" if cost is not None else "n/a")
            lines.append(f"| VLM ({vlm}, {variant_label}) | "
                         f"{_pct(mean_f1)} | "
                         f"{_pct(mean_cer)} | "
                         f"{wall_s} | {cost_s} |")

    # Table 3 — variance across runs (raw + cleaned reported separately)
    lines.append("\n## Table 3 — Per-VLM variance across runs (mean ± std)\n")
    lines.append("Note: with N=1 run per (vlm, slice) the std is 0. Variance only "
                 "becomes informative once 3 runs are collected.\n")
    lines.append("| VLM | Variant | F1 mean ± std | CER mean ± std | Halluc mean ± std |")
    lines.append("|---|---|---|---|---|")
    for vlm in config.MODEL_REGISTRY:
        spec = config.MODEL_REGISTRY[vlm]
        if not spec["available"]:
            note = spec["skip_reason"] or "deferred"
            lines.append(f"| {vlm} | — | _{note}_ | | |")
            continue
        for variant_label, cleaned_flag in (("raw", False), ("cleaned", True)):
            suffix = "_cleaned" if cleaned_flag else ""
            slice_aggs = [v for k, v in report["by_vlm_slice"].items()
                          if k.startswith(f"{vlm}|")
                          and (k.endswith("_cleaned") == cleaned_flag)]
            if not slice_aggs:
                continue
            f1m = statistics.mean(a["f1_mean"] for a in slice_aggs)
            f1s = statistics.mean(a["f1_std"] for a in slice_aggs)
            cer_vals = [a["cer_mean"] for a in slice_aggs
                        if a["cer_mean"] is not None and not (
                            isinstance(a["cer_mean"], float) and math.isnan(a["cer_mean"]))]
            cer_std_vals = [a["cer_std"] for a in slice_aggs
                            if a["cer_std"] is not None and not (
                                isinstance(a["cer_std"], float) and math.isnan(a["cer_std"]))]
            cer_m = statistics.mean(cer_vals) if cer_vals else float("nan")
            cer_s = statistics.mean(cer_std_vals) if cer_std_vals else float("nan")
            h_m = statistics.mean(a["halluc_mean"] for a in slice_aggs)
            h_s = statistics.mean(a["halluc_std"] for a in slice_aggs)
            lines.append(f"| {vlm} | {variant_label} | {_fmt_pm(f1m, f1s)} | "
                         f"{_fmt_pm(cer_m, cer_s)} | {_fmt_pm(h_m, h_s)} |")
    # Skip the original aggregation block.
    return "\n".join(lines + ["\n---\n",
                              f"_n runs aggregated: {len(report['per_run'])} (vlm, slice, run, variant) entries_\n"])

    for vlm in config.MODEL_REGISTRY:
        # Average the per-slice mean/std pairs into one global pair for display.
        slice_aggs = [v for k, v in report["by_vlm_slice"].items()
                      if k.startswith(f"{vlm}|")]
        if not slice_aggs:
            deferred = config.MODEL_REGISTRY[vlm]["skip_reason"] or "deferred"
            lines.append(f"| {vlm} | _{deferred}_ | | |")
            continue
        f1m = statistics.mean(a["f1_mean"] for a in slice_aggs)
        f1s = statistics.mean(a["f1_std"] for a in slice_aggs)
        cer_vals = [a["cer_mean"] for a in slice_aggs
                    if a["cer_mean"] is not None and not (
                        isinstance(a["cer_mean"], float) and math.isnan(a["cer_mean"]))]
        cer_std_vals = [a["cer_std"] for a in slice_aggs
                        if a["cer_std"] is not None and not (
                            isinstance(a["cer_std"], float) and math.isnan(a["cer_std"]))]
        cer_m = statistics.mean(cer_vals) if cer_vals else float("nan")
        cer_s = statistics.mean(cer_std_vals) if cer_std_vals else float("nan")
        h_m = statistics.mean(a["halluc_mean"] for a in slice_aggs)
        h_s = statistics.mean(a["halluc_std"] for a in slice_aggs)
        lines.append(f"| {vlm} | {_fmt_pm(f1m, f1s)} | "
                     f"{_fmt_pm(cer_m, cer_s)} | {_fmt_pm(h_m, h_s)} |")

    lines.append("\n---\n")
    lines.append(f"_n runs aggregated: {len(report['per_run'])} (vlm, slice, run) triples_\n")
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    JSON_OUT.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    MD_OUT.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")
    if not report["per_run"]:
        print("(no runs found — run vlm_extraction/extract.py first)")
        return 0
    print(f"\nAggregated {len(report['per_run'])} runs across "
          f"{len(report['by_vlm_slice'])} (vlm, slice) pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
