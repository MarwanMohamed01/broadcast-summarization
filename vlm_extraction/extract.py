"""Top-level VLM extraction runner.

Usage:
    python -m vlm_extraction.extract --vlm gemini --slice slice_A --runs 3
    python -m vlm_extraction.extract --vlm all   --slice all      --runs 3

Idempotent: re-running picks up where a previous run left off — any
panorama whose `headlines_<id>.json` already exists is skipped unless
--force is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from . import config
from .adapters.base import VLMAdapter, VLMResponse
from .prompts import EXTRACT_PROMPT
from .tiler import tiles_for_slice


def _setup_logger(provider: str) -> logging.Logger:
    log = logging.getLogger(f"vlm.{provider}")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(config.LOGS_DIR / f"vlm_{provider}.log",
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    log.addHandler(sh)
    return log


def _build_adapter(name: str) -> VLMAdapter:
    spec = config.MODEL_REGISTRY[name]
    if not spec["available"]:
        raise RuntimeError(f"{name} unavailable: {spec['skip_reason']}")
    if name == "gemini":
        from .adapters.gemini_adapter import GeminiAdapter
        return GeminiAdapter(config.GOOGLE_API_KEY,
                             temperature=config.TEMPERATURE,
                             timeout_s=config.REQUEST_TIMEOUT_S)
    if name == "gpt4o_mini":
        from .adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(config.OPENAI_API_KEY,
                             temperature=config.TEMPERATURE,
                             timeout_s=config.REQUEST_TIMEOUT_S)
    if name == "claude":
        from .adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(config.ANTHROPIC_API_KEY,
                                temperature=config.TEMPERATURE,
                                timeout_s=config.REQUEST_TIMEOUT_S)
    if name == "qwen":
        from .adapters.huggingface_adapter import HuggingFaceAdapter
        return HuggingFaceAdapter(config.HF_TOKEN,
                                  temperature=config.TEMPERATURE,
                                  timeout_s=config.REQUEST_TIMEOUT_S)
    if name == "groq_scout":
        from .adapters.groq_adapter import GroqAdapter
        return GroqAdapter(config.GROQ_API_KEY,
                           temperature=config.TEMPERATURE,
                           timeout_s=config.REQUEST_TIMEOUT_S)
    if name == "mistral":
        from .adapters.mistral_adapter import MistralAdapter
        return MistralAdapter(config.MISTRAL_API_KEY,
                              temperature=config.TEMPERATURE,
                              timeout_s=config.REQUEST_TIMEOUT_S)
    raise ValueError(f"Unknown VLM: {name}")


def _throttle_seconds(rpm: int) -> float:
    return 60.0 / max(1, rpm)


def _aggregate(headlines_per_pano: Iterable[list[dict]]) -> list[dict]:
    """Dedup across panoramas in a slice, then assign 1-indexed ids.

    Schema-compatible with news_items_cleaned.json: confidence stripped.
    """
    seen: list[str] = []
    out: list[dict] = []
    next_id = 1
    for pano in headlines_per_pano:
        for h in pano:
            text = h.get("text", "").strip()
            if not text:
                continue
            # Light dedup: case-insensitive exact match suffices at the
            # slice level; cross-cycle fuzzy dedup is what v6 does and
            # we'd re-introduce that later if full-video extraction runs.
            key = text.upper()
            if key in seen:
                continue
            seen.append(key)
            out.append({"id": next_id, "text": key})
            next_id += 1
    return out


def _cost_usd(spec: dict, in_tok: int | None, out_tok: int | None,
              think_tok: int | None = None) -> float:
    if in_tok is None or out_tok is None:
        return 0.0
    cost = (in_tok / 1_000_000) * spec["input_per_1m_tokens_usd"]
    cost += (out_tok / 1_000_000) * spec["output_per_1m_tokens_usd"]
    # Add thinking tokens at the higher thinking rate, if present.
    if think_tok and "thinking_per_1m_tokens_usd" in spec:
        cost += (think_tok / 1_000_000) * spec["thinking_per_1m_tokens_usd"]
    return round(cost, 6)


def run_one_triple(vlm: str, slice_name: str, run_idx: int,
                   force: bool = False,
                   max_tiles: int | None = None) -> dict:
    """Run a single (vlm, slice, run) triple. Returns a status dict."""
    log = _setup_logger(vlm)
    spec = config.MODEL_REGISTRY[vlm]
    adapter = _build_adapter(vlm)
    tiles = tiles_for_slice(slice_name)
    if max_tiles:
        tiles = tiles[:max_tiles]
    log.info("starting %s/%s/run_%d: %d tiles, RPM=%d, throttle=%.1fs",
             vlm, slice_name, run_idx, len(tiles), spec["rpm"],
             _throttle_seconds(spec["rpm"]))

    out_dir = config.OUTPUT_DIR / vlm / slice_name / f"run_{run_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    headlines_per_tile: list[list[dict]] = []
    timings: list[dict] = []
    costs: list[dict] = []
    total_in_tok = total_out_tok = 0
    total_cost = 0.0
    t_slice0 = time.time()

    for tile_path in tiles:
        tile_id = f"{tile_path.parent.name}_{tile_path.stem}"  # unique per panorama
        per_file = out_dir / f"headlines_{tile_id}.json"
        if per_file.exists() and not force:
            try:
                existing = json.loads(per_file.read_text(encoding="utf-8"))
            except Exception:
                existing = None
            # Retry on prior error — transient HF/Gemini failures are common
            # and shouldn't be permanently cached.
            if existing and not existing.get("error"):
                log.info("skip already-done %s/%s/run_%d/%s",
                         vlm, slice_name, run_idx, tile_id)
                headlines_per_tile.append(existing.get("headlines", []))
                timings.append(existing.get("timing", {}))
                costs.append(existing.get("cost", {}))
                total_cost += existing.get("cost", {}).get("usd", 0.0)
                continue
            log.info("retry errored %s/%s/run_%d/%s: %r",
                     vlm, slice_name, run_idx, tile_id,
                     (existing or {}).get("error"))

        log.info("call %s on %s (slice=%s run=%d)",
                 vlm, tile_path.name, slice_name, run_idx)
        resp: VLMResponse = adapter.extract_headlines(tile_path, EXTRACT_PROMPT)
        idx = tiles.index(tile_path) + 1
        status = "ERR" if resp.error else f"{len(resp.headlines):>2}h"
        elapsed = time.time() - t_slice0
        eta_s = (elapsed / idx) * (len(tiles) - idx) if idx > 0 else 0
        print(f"  [{idx:>4}/{len(tiles)}] {tile_path.parent.name}/{tile_path.stem} "
              f"-> {status}  ({resp.wall_seconds:.1f}s, "
              f"elapsed {elapsed/60:.1f}m, eta {eta_s/60:.0f}m)"
              + (f"  err={resp.error[:80]}" if resp.error else ""),
              flush=True)

        # persist raw
        (raw_dir / f"raw_{tile_id}.txt").write_text(
            resp.raw_text or (resp.error or ""), encoding="utf-8")

        think_tok = (resp.metadata or {}).get("thoughts_token_count", 0) or 0
        cost_usd = _cost_usd(spec, resp.input_tokens, resp.output_tokens, think_tok)
        total_cost += cost_usd
        total_in_tok += resp.input_tokens or 0
        total_out_tok += resp.output_tokens or 0

        timing = {"wall_seconds": resp.wall_seconds,
                  "input_tokens": resp.input_tokens,
                  "output_tokens": resp.output_tokens}
        cost = {"usd": cost_usd}
        per_file.write_text(json.dumps({
            "tile": str(tile_path),
            "model_id": spec["model_id"],
            "headlines": resp.headlines,
            "timing": timing,
            "cost": cost,
            "error": resp.error,
        }, indent=2), encoding="utf-8")

        headlines_per_tile.append(resp.headlines)
        timings.append(timing)
        costs.append(cost)

        # Throttle to provider RPM unless this was the last tile.
        delay = _throttle_seconds(spec["rpm"])
        if tile_path is not tiles[-1] and delay > 0:
            time.sleep(delay)

    aggregated = _aggregate(headlines_per_tile)
    (out_dir / "headlines_combined.json").write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "timing.json").write_text(
        json.dumps({"per_tile": timings,
                    "total_wall_seconds": time.time() - t_slice0,
                    "total_input_tokens": total_in_tok,
                    "total_output_tokens": total_out_tok}, indent=2),
        encoding="utf-8")
    (out_dir / "cost.json").write_text(
        json.dumps({"per_tile": costs,
                    "total_usd": round(total_cost, 6)}, indent=2),
        encoding="utf-8")

    n_tiles = len(tiles)
    n_h = len(aggregated)
    wall = time.time() - t_slice0
    msg = (f"[done] {vlm} {slice_name} run_{run_idx}: "
           f"{n_tiles} tiles, {n_h} headlines, "
           f"{wall:.0f}s wall, ${total_cost:.4f} cost")
    log.info(msg)
    print(msg)
    return {"vlm": vlm, "slice": slice_name, "run": run_idx,
            "tiles": n_tiles, "headlines": n_h,
            "wall_seconds": wall, "cost_usd": total_cost}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", required=True,
                        choices=list(config.MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--slice", default="all",
                        choices=list(config.SLICE_BOUNDARIES.keys()) + ["all"])
    parser.add_argument("--runs", type=int, default=config.DEFAULT_RUNS)
    parser.add_argument("--force", action="store_true",
                        help="re-run even if outputs exist")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="cap total tiles per (vlm, slice, run) — for smoke tests")
    args = parser.parse_args(argv)

    vlms = (config.available_vlms() if args.vlm == "all"
            else [args.vlm])
    slices = (list(config.SLICE_BOUNDARIES) if args.slice == "all"
              else [args.slice])

    if not vlms:
        deferred = config.deferred_vlms()
        print("No VLMs available. Reasons:")
        for k, why in deferred.items():
            print(f"  - {k}: {why}")
        return 2

    print(f"Running VLMs: {vlms}  slices: {slices}  runs: {args.runs}")
    deferred = config.deferred_vlms()
    if deferred:
        print("Deferred (not run):")
        for k, why in deferred.items():
            print(f"  - {k}: {why}")

    summary = []
    for vlm in vlms:
        for slice_name in slices:
            for run_idx in range(1, args.runs + 1):
                try:
                    summary.append(run_one_triple(vlm, slice_name, run_idx,
                                                  force=args.force,
                                                  max_tiles=args.max_tiles))
                except Exception as exc:
                    logging.getLogger(f"vlm.{vlm}").exception(
                        "triple failed: %s/%s/run_%d", vlm, slice_name, run_idx)
                    print(f"[fail] {vlm} {slice_name} run_{run_idx}: {exc}")
    print("\n=== summary ===")
    for s in summary:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
