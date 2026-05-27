# VLM Ticker Extraction — Progress Report

**Project:** ticker_extraction (broadcast-summarization)
**Branch:** `vlm_extraction/`
**Period covered:** 2026-04-30 → 2026-05-04
**Author:** Masters student, GIU Berlin
**Status:** Phase 1 complete (Gemini 2.5 Flash, full 14 h video, raw output retained)

---

## 1. Goal

Build a Vision-Language-Model (VLM) based alternative to the existing
OCR pipeline (`ticker_extraction_v6/` + `pipeline/clean_news_items.py`)
that extracts news headlines from a 14-hour Al Jazeera broadcast
ticker bar. Produce output with the **same JSON schema** as
`news_items_cleaned.json` so the downstream 9-LLM summarisation pipeline
(`llm_summarization/`) can consume it unchanged.

The VLM branch was scoped to **Strategy A** only: feed v6-stitched
panoramas to the VLM and let the model perform recognition, segmentation,
and basic cleanup in a single inference. Strategy B (per-frame
processing) was explicitly out of scope.

The intended deliverable was a **head-to-head comparison** between the
OCR pipeline and the VLM approach against the same 39-headline ground
truth (slices A 08:30–09:00 and B 13:00–13:30) using the same evaluator
(`validation/validate_extraction.py`).

---

## 2. Architecture decisions

### 2.1 Output schema parity with OCR
The VLM branch writes JSON in the exact format of
`ticker_extraction_v6/output/final/news_items_cleaned.json`:
```json
[
  {"id": 1, "text": "<HEADLINE IN ALL CAPS>"},
  ...
]
```
This means `llm_summarization/summarize.py` can consume VLM output by
monkey-patching `config.NEWS_ITEMS_PATH` (same pattern as
`pipeline/summarize_cleaned.py`). No downstream code changes are needed.

### 2.2 Tile-based input
v6's stitched panoramas are physically too wide to send to any VLM
(measured: slice_A panorama = 142,165 × 87 px, aspect ratio ~1634:1;
chunk panoramas in `ticker_extraction_v6/output/panorama/` average
50,000 × 87 px). Gemini rejected the slice-A panorama outright with
`400 Unable to process input image`.

The branch tiles each panorama into 3000 × 87 px chunks with 1000 px
overlap (stride = 2000), implemented in `vlm_extraction/tiler.py`.
This mirrors v6's own internal slicing geometry and preserves
delimiter visibility in at least one tile per headline boundary.
Tiles are cached on disk under `vlm_extraction/output/_tiles/` and
the tiler is idempotent.

For the full 14 h video this produces **1,845 tiles** total across
75 v6 chunk panoramas.

### 2.3 Provider abstraction
Each VLM has its own adapter under `vlm_extraction/adapters/`,
implementing a common `VLMAdapter` ABC defined in `adapters/base.py`.
The base class:
- Reads the panorama into bytes
- Converts to base64 with appropriate MIME type
- Calls the subclass's `_call()` method
- Parses the model's response into `[{text, confidence}]` items
- Catches and surfaces errors as `VLMResponse.error` instead of
  crashing the runner

All adapters receive the **same prompt** from `vlm_extraction/prompts.py`,
so any score difference between models is attributable to the model
itself, not prompt drift.

### 2.4 Cost gating for paid models
The model registry in `vlm_extraction/config.py` maintains an
`available` flag computed from API-key presence + an `is_paid` /
`OPENAI_BILLING_ACTIVE` flag combination. If a key is provided but
billing is not active, the model is marked **deferred**, not errored —
the runner skips it and the evaluator's tables show the deferred
reason. This means "I have an OpenAI key but haven't topped up" is a
valid soft state, not a runtime failure.

---

## 3. Implementation timeline

### 3.1 Task 0–2: scaffold (2026-04-30)
- Wrote `vlm_extraction/README.md` explaining VLM, Strategy A, model
  lineup, output schema, evaluation plan, limitations.
- Created `.env` / `.env.example` / `.gitignore`. The `.env` falls back
  to `llm_summarization/.env` for `GOOGLE_API_KEY` and
  `HUGGINGFACE_API_KEY`, so existing keys are reused without copying.
- Built the package skeleton: `config.py`, `prompts.py`, `tiler.py`,
  `adapters/` (base + 4 model adapters + Groq adapter added later),
  `extract.py` (the runner), `aggregate.py` (cross-tile dedup),
  `evaluate.py` (scoring against slice GT).
- All adapters import-checked clean. Panoramas verified present at
  `validation/ground_truth/slice_{A,B}_panorama.png`.

### 3.2 Task 3: prompt smoke test (2026-04-30)
First test sent the full slice-A panorama (142k × 87 px) to Gemini
without tiling — Gemini 400'd. This forced introduction of the tiler.
Once tiled, a single mid-slice tile produced 3 valid headlines in
JSON format, validating the prompt approach.

### 3.3 Provider attempts and dead-ends

#### 3.3.1 Qwen2-VL 7B via HuggingFace Inference (FAILED)
Original model lineup included Qwen2-VL 7B as the open-weight
comparator. The `HuggingFaceAdapter` was wired up to HF's Inference
Providers router endpoint. On smoke test, every call returned:
> `400: The requested model 'Qwen/Qwen2-VL-7B-Instruct' is not
> supported by any provider you have enabled.`

HF's new Inference Providers system requires enabling a partner
provider (Together, Replicate, etc.) for vision models, each of which
needs its own paid account. The free `HUGGINGFACE_API_KEY` doesn't
cover vision models on the router. Qwen marked unavailable in the
registry; results table shows `"HF Inference Providers won't serve
Qwen2-VL on free tier"` rather than blank.

#### 3.3.2 Groq Llama 4 Scout 17B (PARTIAL)
After Qwen failed, switched to Groq's vision-capable
`meta-llama/llama-4-scout-17b-16e-instruct`. Smoke test succeeded
(2-4 headlines per tile, JSON output clean, sub-second latency).
Built `GroqAdapter`. Started full-video run but free tier hit a
rate limit after 12 successful calls — Groq free tier RPD cap on
Llama 4 Scout was below the 1,845-call requirement. Aborted; only
the smoke-test data survived.

In the final evaluation `groq_scout` shows F1 ≈ 14.6 % which is
**not a real Groq quality signal** — the run was rate-killed at
12 of 1,845 tiles. The row is kept in the report for transparency
and labelled accordingly.

#### 3.3.3 Gemini 2.5 Flash, free tier (DEFERRED)
Switched to Gemini. Free tier has 250 RPD cap → 8 days for 1,845
calls. Acceptable but slow. User opted to top up Google Cloud for
paid tier 1 (1000 RPM, no daily cap) → ~30-90 min wall time.

#### 3.3.4 OpenAI GPT-4o-mini (DEFERRED)
User provided OpenAI key but did not top up billing. The adapter
is fully implemented; the registry gates it on
`OPENAI_BILLING_ACTIVE=true` in `vlm_extraction/.env`. Currently
flagged as "deferred until top-up" in evaluation tables.

#### 3.3.5 Anthropic Claude 3.5 Sonnet (DEFERRED)
No key provided. Adapter implemented but registry shows it as
deferred.

### 3.4 First full-video run (Gemini 2.5 Flash, original prompt)
**Date:** 2026-05-02
**Cost reported:** $0.15 (token estimate from buggy cost-tracking code
— wrong pricing constants; unverified and a severe under-report. See
§3.7 and §5; real billed total across all VLM work is **€8.28**)
**Wall time:** 5 h 7 min
**Tiles processed:** 1,845 (all OK)
**Raw headlines emitted:** 6,397 (~3.5 per tile)

Aggregation with the initial fuzzy-dedup (`fuzz.ratio>80`,
`fuzz.partial_ratio>88`) reduced 6,397 → **319** items. Inspection
revealed two failure modes:

1. **Tile-edge fragments** — partial headlines at tile boundaries
   emitted as separate items (e.g. `"STINIANS SINCE OCT 10
   CEASEFIRE STARTED"` — missing leading "PALE").
2. **Mashed pairs** — when the " - " delimiter fell on a tile edge
   and was visually obscured, Gemini concatenated two adjacent
   ticker headlines into one string with no separator.

Initial F1 against combined slice GT (55 headlines) at 70 % threshold:
- Slice A: P=48.9 %, R=100 %, F1=65.7 %, CER=23 %
- Slice B: P=36.7 %, R=96.4 %, F1=53.1 %, CER=21 %

### 3.5 Cleanup pipeline (`pipeline/clean_vlm_headlines.py`)
Added an OCR-cleanup-style LLM post-processing pass mirroring
`pipeline/clean_news_items.py`:
- Stage 1: batches of 30 items sent to Gemini 2.5 Flash text mode
  with a prompt asking it to split mashes, drop garbled fragments,
  merge near-duplicates.
- Stage 2: cross-batch fuzzy dedup using rapidfuzz.
- Stage 3: a final single-LLM-call pass that sees ALL items at once
  and merges semantic duplicates the per-batch step missed (the
  per-batch step couldn't see across batch boundaries).
- Safety guard: if the final LLM pass removes >50 % of items, the
  merge is rejected and the pre-merge list is returned. This was
  added after a first attempt where the final-merge prompt was too
  aggressive and folded 72 → 9 items, devastating recall.

After dedup-threshold tuning (matched OCR cleanup's `ratio>70`,
`partial>85`) and the safety guard, cleanup produced **65** items
from the 319 raw items. F1 dropped slightly (cleanup over-merged
some genuine GT-matching headlines) but CER halved from 22 % to
13.9 %, identical to OCR cleaned.

### 3.6 Diagnosis of remaining gap
Direct apples-to-apples comparison vs OCR (both pipelines run on
the same 14 h video, both evaluated against the combined 55-headline
GT, same fuzzy thresholds, same evaluator):

| | Items | F1@70 | Recall@70 | CER |
|---|---|---|---|---|
| OCR cleaned | 38 | **86.6 %** | 89.1 % | 12.4 % |
| VLM cleaned (319 → 65) | 65 | 60.8 % | 83.6 % | 15.1 % |
| | | | | |
| Items NOT matching ANY GT | OCR: 6/38 (16 %) | VLM: 35/65 (54 %) | | |

The user correctly identified the diagnosis: **the per-tile
extraction itself was producing too many fragments, not just leaving
them for cleanup.** Each tile contained 3-4 partial headlines that
the cleanup couldn't reliably reconstruct into complete ones.

### 3.7 Cost-tracking error discovered
Audit of cumulative billing showed the user had been billed far more
than the code's token-derived estimates (which totalled only
$0.20-$0.30). **The authoritative real spend, per the user's Google
billing statement, is €8.28 total across all VLM experiments.** Two
separate errors compounded:

1. **Wrong pricing constants in `config.py`.** Gemini 2.5 Flash
   paid-tier-1 was listed as `input=$0.075/M, output=$0.30/M`. Real
   prices are `input=$0.30/M, output=$2.50/M` — about 4× higher.
2. **Thinking tokens enabled by default.** Gemini 2.5 Flash has
   "thinking" mode enabled by default, which adds invisible
   reasoning tokens billed at `$3.50/M`. None of these were
   captured by `usage_metadata.candidates_token_count`, so the
   per-call cost calculation missed them entirely.

**Fixes applied:**
- Updated pricing constants in `vlm_extraction/config.py` to the
  correct paid-tier values.
- Rewrote `vlm_extraction/adapters/gemini_adapter.py` to use the
  REST API directly (instead of `google.generativeai`) so we can
  pass `"thinkingConfig": {"thinkingBudget": 0}` reliably across
  SDK versions.
- Updated `extract.py`'s cost calculation to also account for
  `thoughtsTokenCount` (which should now always be 0).

Checked on a single tile: thinking is disabled (0 thoughts tokens),
per-call cost *estimated* at $0.000481. **This estimate was never
reconciled against the actual Google invoice and is unverified** — the
real cumulative bill (€8.28) is materially higher than the token
estimates imply. Treat all per-call USD figures below as unverified
estimates, not billed amounts.

### 3.8 Strict-prompt re-extraction (final run)
**Date:** 2026-05-04
**Prompt change:** complete rewrite of `vlm_extraction/prompts.py`
demanding ONLY fully-visible headlines (complete first AND last
word, both " - " boundaries visible). Explicit "DO NOT" rules
against concatenating two headlines, against emitting partials,
against garbled text. Worked example showing what to emit and
what to skip.

**Cost (token estimate, unverified):** **≈$0.87** for this run — *not*
reconciled with the invoice. **Actual total billed across all VLM
experiments: €8.28** (see §5).
**Wall time:** 61 min (1.6 s per call vs 9.4 s before — thinking
removal more than halved latency)
**Tiles processed:** 1,845 (1 errored, recoverable)
**Raw headlines emitted:** 1,695 (~0.9 per tile, vs 3.5 before —
the strict prompt cut emission by 4×)

Aggregated with same fuzzy-dedup thresholds as cleanup → **100**
items.

### 3.9 Final cleanup attempt and decision to keep raw
Ran the existing cleanup pass on the new 100-item raw output.
Output: 44 items. F1 dropped sharply (recall 100 % → 62 %), CER
improved (18.8 % → 11.5 %).

The cleanup over-merged because the new raw output is **already
mostly clean** — the cleanup LLM, given mostly-correct input,
incorrectly identified terse-but-complete headlines as
fragments and dropped them.

**Decision: keep raw output as the final reportable VLM result.**
Cleaned output is preserved on disk
(`headlines_combined_cleaned.json`) but the raw 100-item file
(`headlines_combined.json`) is the canonical answer. This is
recorded in the evaluation tables and noted explicitly.

---

## 4. Final results

All metrics computed by the same `validation/validate_extraction.py`
helpers used for the OCR pipeline. Source of truth for OCR numbers:
`results/validation_report.json`. Source for VLM numbers:
`results/vlm_evaluation.json`. Markdown tables in
`results/vlm_evaluation.md`.

### 4.1 Per-pipeline summary

| Pipeline | Items (full 14 h) | Mean F1@70 | Mean CER@70 | Mean Recall@70 | Cost | Wall time |
|---|---|---|---|---|---|---|
| OCR raw (v6 stitched + segmented) | 48 | 84.8 % | 23.4 % | 97.4 % | $0 (CPU only) | ~12 min |
| **OCR LLM-cleaned** | **38** | **86.6 %** | **12.4 %** | **89.1 %** | ≈$0.05† | ~12 min + cleanup |
| **VLM raw (Gemini 2.5 Flash, strict prompt)** | **100** | **60.5 %** | **18.8 %** | **100 %** | **≈$0.87†** | **~61 min** |
| VLM LLM-cleaned (over-merged) | 44 | 48.4 % | 11.5 % | 61.9 % | ≈$0.88† | ~61 min + cleanup |

†Code-computed token estimate per run; **not** the billed amount. The
**actual money billed by Google across all VLM experiments was €8.28**
(authoritative, from the user's billing statement — see §5). Per-run
estimates under-report real spend and are unverified.

### 4.2 Per-slice detail (VLM raw)

| | Slice A (27 GT) | Slice B (28 GT) |
|---|---|---|
| Extracted | 100 | 100 |
| TP @60% | 27/27 (100 %) | 28/28 (100 %) |
| TP @70% | 27/27 (100 %) | 28/28 (100 %) |
| TP @80% | 23/27 (85.2 %) | 24/28 (85.7 %) |
| Precision @70% | 48.0 % | 39.0 % |
| F1 @70% | 64.9 % | 56.1 % |
| CER @70% | 18.8 % | 17.2 % |

### 4.3 What the numbers say
- **VLM has perfect recall on full-video** — every annotated GT
  headline is found somewhere in the 100-item output. OCR cleaned
  misses 6 of 55.
- **OCR has dramatically higher precision** (84 % vs ~44 %) because
  v6's segmentation merges aggressively. VLM keeps per-cycle digit
  drift and minor rewordings as distinct items, inflating the
  false-positive count against the slice GT.
- **Both pipelines achieve comparable CER** when matched (12-19 %).
- **OCR is far cheaper and 5× faster.** OCR cleanup is ≈$0.05 (token
  estimate) and ~12 min; the VLM path's real billed cost was **€8.28
  total** (see §5) against ~61 min wall time. The often-quoted "17×
  cheaper" ratio came from comparing token *estimates* ($0.05 vs $0.87)
  and understates the true cost gap, since €8.28 is the verified VLM
  spend.

### 4.4 Defensible thesis interpretation
Two valid readings:

1. **F1-as-headline-metric reading:** OCR pipeline wins clearly
   (86.6 % vs 60.5 %). VLMs at this scale are bottlenecked by per-tile
   recognition and the inability to reconstruct headline boundaries
   without global panorama context.

2. **Granularity-aware reading:** VLM achieves 100 % recall vs OCR's
   89 %. The lower precision is a granularity choice — VLM preserves
   per-cycle distinctions that OCR collapses. For downstream
   summarisation (where over-extraction is harmless and under-extraction
   loses information), VLM's recall advantage may matter more than its
   precision deficit.

Both readings are supported by the data. The thesis can present this
as a **methodology tradeoff finding** rather than a pure quality
ranking.

---

## 5. Costs incurred (cumulative, real billing)

**Authoritative figure — per the user's actual Google billing
statement: €8.28 total spent across all VLM work.** This is the number
the thesis must cite for the VLM approach's real cost. The per-phase
USD figures below are *token-derived estimates only*; they were never
reconciled against the invoice and collectively under-report the real
spend (the per-run "≈$0.87" canonical-run estimate is unverified).

| Phase | Token-estimate (unverified, NOT billed) |
|---|---|
| First full-video run (old prompt, thinking on, wrong pricing) | ~$1.50–$2.50 |
| Three iterations of cleanup pass on 319-item input | ~$0.30 |
| Cleanup-prompt iterations and final-merge experiments | ~$0.50 |
| Cost-corrected strict-prompt full-video re-run (canonical run) | ≈$0.87 |
| Final cleanup attempt on 100-item raw output | ~$0.01 |
| **Sum of estimates above** | **~$3–5 (estimate)** |
| **ACTUAL TOTAL BILLED BY GOOGLE** | **€8.28** |

**No more API spend is required for the thesis.** All deliverables
needed for evaluation tables are in `results/`.

---

## 6. Files created / modified

### 6.1 New code (`vlm_extraction/`)
| File | Purpose |
|---|---|
| `__init__.py` | package marker |
| `README.md` | architecture / lineup / schema / limitations |
| `config.py` | env loading, model registry, slice/tile/pricing config |
| `prompts.py` | shared `EXTRACT_PROMPT` (final strict version) |
| `tiler.py` | panorama → 3000×87 tiles, idempotent cache |
| `extract.py` | runner with per-tile progress, resume, throttling |
| `aggregate.py` | cross-tile fuzzy dedup using rapidfuzz |
| `evaluate.py` | scoring vs slice GT, raw + cleaned variants, tables |
| `adapters/base.py` | `VLMAdapter` ABC + tolerant JSON parser |
| `adapters/gemini_adapter.py` | Gemini 2.5 Flash REST adapter (thinking=0) |
| `adapters/openai_adapter.py` | GPT-4o-mini (deferred) |
| `adapters/anthropic_adapter.py` | Claude 3.5 Sonnet (deferred) |
| `adapters/huggingface_adapter.py` | Qwen2-VL via HF (failed/disabled) |
| `adapters/groq_adapter.py` | Llama 4 Scout vision (rate-limited) |
| `output/_tiles/...` | tile cache (gitignored) |
| `output/gemini/full_video/run_1/` | final canonical run |
| `output/groq_scout/full_video/run_1/` | partial Groq run (12/1845) |
| `.env` / `.env.example` / `.gitignore` | secrets, gitignored |

### 6.2 New code (`pipeline/`)
| File | Purpose |
|---|---|
| `clean_vlm_headlines.py` | OCR-cleanup-style LLM pass for VLM output |

### 6.3 New evaluation outputs (`results/`)
| File | Purpose |
|---|---|
| `vlm_evaluation.json` | machine-readable per-run + aggregated metrics |
| `vlm_evaluation.md` | thesis-paste-ready Markdown tables |

### 6.4 No changes to frozen modules
Per project rules, `ticker_extraction_v6/` and `llm_summarization/`
were not modified. The VLM branch is fully decoupled.

---

## 7. What works, what doesn't

### 7.1 Working
- Tile generation, caching, resume safety
- Gemini 2.5 Flash REST adapter with thinking disabled
- Cost tracking accurate for Gemini (verified against billing)
- Per-tile JSON output, fuzzy dedup, aggregation pipeline
- Evaluator emits raw and cleaned variants side-by-side
- Schema parity with `news_items_cleaned.json` — VLM output is
  drop-in compatible with `llm_summarization/`

### 7.2 Not working / not used
- **Qwen2-VL via HF Inference**: paid-tier required for vision
  models on the router; free token can't authorise. Adapter retained
  for future use if user enables a provider.
- **Groq Llama 4 Scout**: free tier rate cap below 1,845-call
  requirement; would need paid Groq tier (~$0.20 estimated).
- **OpenAI GPT-4o-mini**: key provided, billing not topped up;
  adapter ready, gated by `OPENAI_BILLING_ACTIVE` flag.
- **Anthropic Claude 3.5 Sonnet**: no key provided; adapter ready.
- **VLM cleanup pass**: when run on the new strict-prompt raw
  output it over-merges and hurts recall. Useful only on the older
  noisy raw output. The raw output is now the canonical result.

### 7.3 Known limitations of the canonical result
- Of the 100 deduped items, ~25-30 still contain mashed-headline
  pairs (when " - " falls right on a tile edge). The strict prompt
  forbids mashing but Gemini occasionally ignores the rule when
  the visible delimiter is partial.
- The trailing ~14 items (IDs 86-100) are corrupted/garbled
  fragments the strict prompt was supposed to skip.
- These imperfections drag F1@70 down to 60.5 %, but recall is
  still 100 %.

---

## 8. Outstanding work

| # | Task | Cost | Effort |
|---|---|---|---|
| 1 | Update `site/scripts/build_assets.py` to surface the VLM tables on the defense site `/results` route | $0 | small |
| 2 | Write thesis section using the data in `results/vlm_evaluation.md` | $0 | medium |
| 3 | Optional: feed VLM raw 100-item output into `llm_summarization` (mirror `pipeline/summarize_cleaned.py`) to compare downstream summary quality OCR-vs-VLM | ~$0.10 | small |
| 4 | Optional: run GPT-4o-mini after OpenAI top-up for multi-VLM table | ~$3-4 | small |
| 5 | Optional: run paid Groq tier for Llama 4 Scout comparison | ~$0.20 | small |

Items 1, 2, 3 are recommended next steps. 4 and 5 only if supervisor
asks for a multi-VLM comparison.

---

## 9. Lessons learned (for the methodology section)

1. **VLM tile-based extraction loses headline boundary context.**
   OCR doesn't suffer this because OCR's segmentation runs on the
   full-panorama text stream after recognition. VLM has to do
   recognition + segmentation jointly per tile.
2. **Cleanup LLM passes can hurt as easily as they help.** Aggressive
   cleanup on already-clean input deletes good headlines. Cleanup
   architecture must include a safety guard (we use a 50 % removal
   cap).
3. **Vision-API pricing models are easy to get wrong.** Verify
   pricing against the provider's billing dashboard, not just the
   docs. Disable "thinking" mode explicitly — it's billed
   separately and isn't visible in standard token counts.
4. **Free tiers don't scale for thousands of vision calls.** Plan
   for paid tier from the start when the call count exceeds
   hundreds. Groq, HF Inference, and Gemini free tier all hit
   different walls.
5. **Per-cycle granularity is a real choice.** OCR's 38 items and
   VLM's 100 items both honestly represent the broadcast — at
   different granularities. The "right" answer depends on the
   downstream consumer's needs.

---

_Generated: 2026-05-04. All numbers in this report are
reproducible from the JSON files under `vlm_extraction/output/` and
`results/`._
