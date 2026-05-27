"""Re-aggregate cached per-tile VLM outputs with fuzzy dedup.

The per-tile JSONs under
    output/<vlm>/<slice>/run_<n>/headlines_<tile_id>.json
are written by extract.py at run time. The original aggregator does a
case-insensitive exact-match dedup which leaves tile-edge fragments in
the final list. This script applies rapidfuzz substring dedup
(longest version wins) to produce a clean headlines_combined.json
without any new API calls.

Idempotent: overwrites headlines_combined.json each call.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rapidfuzz import fuzz

from . import config

# Headlines whose stripped form is shorter than this are too short to be a
# real headline — likely a fragment artefact (e.g. "ZA ON MONDAY").
MIN_HEADLINE_LEN = 30

# Headlines whose stripped form is longer than this are not real ticker
# headlines — they're garbled multi-headline mashups Gemini occasionally
# emits when a panorama seam confuses it. Real Al Jazeera tickers max ~120 chars.
MAX_HEADLINE_LEN = 150

# Two headlines whose token-set similarity exceeds this are duplicates.
TOKEN_SET_DUP = 80

# A shorter headline is a fragment of a longer one if its partial_ratio
# against the longer one exceeds this.
PARTIAL_DUP = 88


_DIGIT_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _norm(text: str) -> str:
    """Normalise text for dedup comparison: upper, digits-stripped, punct-stripped."""
    t = text.upper()
    t = _DIGIT_RE.sub("", t)
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def fuzzy_dedup(headlines: list[str]) -> list[str]:
    """Return a deduped list, preferring the longest version of each headline.

    Strategy: sort by length descending; for each candidate, drop if it's a
    fragment of (or near-duplicate of) any already-kept headline.
    """
    # Sort longest first so fragments lose to complete versions.
    candidates = sorted({h.strip() for h in headlines if h.strip()},
                        key=len, reverse=True)
    kept: list[str] = []
    kept_norm: list[str] = []
    for h in candidates:
        h_norm = _norm(h)
        if len(h_norm) < MIN_HEADLINE_LEN or len(h_norm) > MAX_HEADLINE_LEN:
            continue
        is_dup = False
        for k_norm in kept_norm:
            # Token-set match catches reorderings + minor digit drift
            if fuzz.token_set_ratio(h_norm, k_norm) >= TOKEN_SET_DUP:
                is_dup = True
                break
            # Substring match catches "ZA ON MONDAY" inside "...GAZA ON MONDAY"
            if fuzz.partial_ratio(h_norm, k_norm) >= PARTIAL_DUP:
                is_dup = True
                break
        if not is_dup:
            kept.append(h)
            kept_norm.append(h_norm)
    return kept


def aggregate_run(vlm: str, slice_name: str, run_idx: int) -> Path:
    run_dir = config.OUTPUT_DIR / vlm / slice_name / f"run_{run_idx}"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    tile_files = sorted(run_dir.glob("headlines_*.json"))
    # Don't accidentally pick up the aggregated outputs themselves
    # (they're top-level lists, not dicts, and trip the dict access below).
    tile_files = [f for f in tile_files
                  if not f.stem.startswith("headlines_combined")]

    raw_headlines: list[str] = []
    err_tiles = 0
    for f in tile_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            err_tiles += 1
            continue
        if data.get("error"):
            err_tiles += 1
            continue
        for h in data.get("headlines", []):
            text = (h.get("text") or "").strip()
            if text:
                raw_headlines.append(text)

    print(f"  loaded {len(tile_files)} tile files ({err_tiles} errored)")
    print(f"  raw headlines (pre-dedup): {len(raw_headlines)}")

    deduped = fuzzy_dedup(raw_headlines)
    print(f"  deduped headlines: {len(deduped)}")

    out = [{"id": i + 1, "text": text} for i, text in enumerate(deduped)]
    out_path = run_dir / "headlines_combined.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"  wrote {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", default="all")
    parser.add_argument("--slice", default="all")
    parser.add_argument("--runs", default="all",
                        help="run number, comma-separated, or 'all'")
    args = parser.parse_args()

    vlms = (list(config.MODEL_REGISTRY) if args.vlm == "all" else [args.vlm])
    slices = (list(config.SLICE_BOUNDARIES) if args.slice == "all"
              else [args.slice])

    found_any = False
    for vlm in vlms:
        for slice_name in slices:
            slice_dir = config.OUTPUT_DIR / vlm / slice_name
            if not slice_dir.exists():
                continue
            run_dirs = sorted(slice_dir.glob("run_*"))
            for run_dir in run_dirs:
                try:
                    run_idx = int(run_dir.name.split("_")[-1])
                except ValueError:
                    continue
                if args.runs != "all":
                    keep = {int(s) for s in args.runs.split(",")}
                    if run_idx not in keep:
                        continue
                print(f"=== aggregating {vlm} / {slice_name} / run_{run_idx} ===")
                aggregate_run(vlm, slice_name, run_idx)
                found_any = True
    if not found_any:
        print("no runs found")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
