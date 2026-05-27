# VLM Extraction — Vision-Language-Model-Based Ticker Headlines

Parallel alternative to the OCR pipeline (`ticker_extraction_v6/` +
`pipeline/clean_news_items.py`). Instead of detecting and recognising
ticker text glyph-by-glyph and then cleaning the result with an LLM,
we hand a **stitched panorama image** straight to a multimodal model
and ask it to produce structured JSON headlines in one shot.

The output schema is **identical** to
`ticker_extraction_v6/output/final/news_items_cleaned.json`, so the
existing 9-LLM summarisation pipeline (`llm_summarization/`) consumes
VLM output unchanged — no downstream code needs to know which branch
produced the headlines.

---

## 1. What is a VLM?

A **Vision-Language Model (VLM)** is a large language model whose
input layer accepts **images alongside text**. The model is trained on
paired image-text data so it can reason jointly about pixels and words
— it can describe scenes, read text inside images, follow instructions
about visual content, and emit structured outputs the same way a
text-only LLM does. Examples used here: Google Gemini 2.5 Flash,
OpenAI GPT-4o-mini, Anthropic Claude 3.5 Sonnet, and the open-weight
Qwen2-VL 7B served via HuggingFace Inference. From the calling code's
perspective the API surface looks just like a chat LLM, except one of
the message parts is an image rather than text.

---

## 2. Why VLMs for ticker extraction?

**Pros**
- **One call replaces a five-stage pipeline.** No frame extraction,
  scroll detection, panorama stitching, OCR, or rule-based
  segmentation — the VLM does perception + segmentation + cleanup in
  a single inference.
- **Robust to OCR failure modes.** Stylised ticker fonts, anti-aliased
  glyphs at panorama seam boundaries, and the thin " - " delimiter
  that makes EasyOCR/PaddleOCR/TrOCR collapse to 0 % F1 in our pure
  comparison are usually handled correctly because the model treats
  text as a visual concept, not a pixel pattern.
- **Native handling of merged / truncated headlines.** The same
  cross-headline reasoning that `pipeline/clean_news_items.py` invokes
  Gemini for *after* OCR is folded into extraction itself.
- **Structured output guaranteed by prompt + JSON-mode.** No regex on
  free-form OCR streams.

**Cons**
- **Cost.** Every panorama is a paid API call. For the 14 h video
  (≈ hundreds of stitched cycles) the per-VLM bill is non-trivial; we
  budget ~$10 across all paid models.
- **Rate limits.** Free tiers throttle aggressively (Gemini ~10 RPM,
  HF Inference variable). Long runs need throttling and resume logic.
- **Determinism.** Even with `temperature=0`, vision models have
  shown run-to-run drift; we measure it explicitly with 3 runs per
  (model, slice).
- **Hallucination.** A VLM can confidently emit a headline that was
  never in the ticker. This is the OCR pipeline's hardest-to-make
  failure mode and the VLM's easiest, so we measure hallucination
  rate as a first-class metric.
- **Internet dependency.** Three of four models require an external
  API; Qwen2-VL via HF Inference is the only option that comes close
  to "local-ish".

**OCR vs. VLM at a glance**

| Dimension | OCR pipeline (v6 + cleaning LLM) | VLM (Strategy A) |
|---|---|---|
| Stages | 5 (frames → scroll → stitch → OCR → segment) + 1 cleanup | 1 |
| Per-cycle cost | CPU only, ~free | $0.001 – $0.05 per call depending on model |
| Failure mode | Missed glyphs, merged headlines | Hallucinated headlines |
| Determinism | Fully deterministic | Stochastic; needs variance measurement |
| Validated F1 (39-headline GT) | 88.1 % cleaned | TBD by this experiment |

---

## 3. Architecture (parallel branches)

```
                 raw broadcast video (.mp4, 14 h)
                              │
                              ▼
              ┌───────────────┴────────────────┐
              │                                │
     ┌────────▼──────────┐          ┌──────────▼──────────┐
     │ OCR branch        │          │ VLM branch          │
     │ (existing)        │          │ (this folder)       │
     │                   │          │                     │
     │ ticker_extraction │          │ v6 panoramas        │
     │ _v6/  (5 stages)  │          │ (re-used as input)  │
     │        │          │          │       │             │
     │        ▼          │          │       ▼             │
     │  news_items.json  │          │  vlm_extraction/    │
     │        │          │          │  extract.py         │
     │        ▼          │          │       │             │
     │ pipeline/clean_   │          │       ▼             │
     │ news_items.py     │          │ headlines_*.json    │
     │        │          │          │ (same schema as     │
     │        ▼          │          │  news_items_        │
     │ news_items_       │          │  cleaned.json)      │
     │ cleaned.json      │          │       │             │
     └────────┬──────────┘          └───────┬─────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
              llm_summarization/  (9-LLM, frozen, unchanged)
              ├─ output_cleaned/        (from OCR branch)
              └─ output_vlm_<provider>/ (from VLM branch)
                             │
                             ▼
                        evaluate.py
                        ROUGE + BERTScore
                        head-to-head ranking
```

**Audio branch** (`asr/` + `llm_summarization/output_asr/`) sits
alongside both visual branches and is unchanged by this work.

---

## 4. Strategy A — panorama input, one call per cycle

The v6 pipeline already produces the most useful intermediate artefact
for a VLM: the **stitched panorama**. Each panorama is the result of
detecting one full ticker cycle, dewarping the scrolling crop, and
horizontally stitching frames so all the text within one cycle is
visible as a single wide image (~3000 px × ticker height per slice).

Strategy A re-uses these panoramas as VLM input:
- One panorama → one VLM API call → one JSON headline list.
- Multiple panoramas in a slice / video → headlines aggregated and
  deduplicated (re-using v6's segmentation dedup helpers where useful).

Strategy B (sending individual frames and forcing the VLM to handle
scroll-tracking itself) is **explicitly out of scope** — it multiplies
API cost by 30–60× per cycle and pushes work the v6 pipeline already
solves correctly back onto the VLM.

For the slice-level evaluation (Slice A 08:30–09:00, Slice B
13:00–13:30) the panoramas at
`validation/ground_truth/slice_{A,B}_panorama.png` are used directly,
so OCR and VLM see byte-identical input.

---

## 5. VLM lineup and rationale

| # | Model | Provider | Why |
|---|---|---|---|
| 1 | **Gemini 2.5 Flash** | Google AI Studio | Strong free tier; same model `pipeline/clean_news_items.py` already uses for OCR cleanup, so direct comparison "what does Gemini produce when given pixels vs. cleaned OCR text". |
| 2 | **GPT-4o-mini** | OpenAI | Cheapest competent OpenAI vision model (~$0.15 / 1 M input tokens, image at 85 tokens base). Industry baseline. |
| 3 | **Claude 3.5 Sonnet** | Anthropic | Strongest documented OCR-on-image performance among frontier models; the "if Claude can't, none of them can" anchor. |
| 4 | **Qwen2-VL 7B** | HF Inference | Optional. Open-weight comparator — answers "do we need a frontier closed model, or is a 7B open VLM sufficient?". Will be skipped if the HF token is not provided or the endpoint is cold. |

All four use the same prompt, same panorama files, same schema,
same evaluator. Differences in score therefore measure model quality,
not protocol differences.

---

## 6. Output schema

Identical to `ticker_extraction_v6/output/final/news_items_cleaned.json`:

```json
[
  {"id": 1, "text": "PRESIDENT TRUMP TELLS U.S. MEDIA THAT DEAL IS POSSIBLE BY MONDAY AS IRAN 'NEGOTIATING NOW'"},
  {"id": 2, "text": "LEBANON HEALTH MINISTRY: 1,461 PEOPLE KILLED AND INJURED IN ISRAELI ATTACKS SINCE START OF WAR"}
]
```

`id` is 1-indexed and assigned at aggregation time (after dedup
across panoramas in a slice). `text` is the headline as the VLM read
it, normalised to upper case to match v6 convention.

Internally the VLM is prompted to also emit a `confidence` field
(`high|medium|low`), which is preserved in the per-panorama files
under `output/<provider>/<slice>/run_<n>/headlines_<pano_id>.json` for
ablation, but **stripped from the aggregated `headlines_combined.json`**
so the file is byte-compatible with what `llm_summarization` already
consumes.

---

## 7. Evaluation plan

Same ground truth as the OCR comparison (`validation/ground_truth/`,
39 headlines across two 30-min slices), same fuzzy-match thresholds
(60 / 70 / 80 %), same metric helpers
(`validation/validate_extraction.py`).

Per (model, slice, run) we record:
- **Precision / Recall / F1** at all three thresholds.
- **Character Error Rate (CER)** across matched headlines.
- **Hallucination rate** = fraction of model-emitted headlines that
  match no GT headline at the 60 % threshold (lowest-bar match — if
  it doesn't match here, the model invented it).
- **Wall-clock time per call.**
- **Cost per call** (USD) using each provider's published rates,
  computed from returned token counts where available, estimated from
  panorama dimensions otherwise.

Per (model, slice) we aggregate **mean ± std across the 3 runs** to
quantify the VLM's run-to-run variance.

Final outputs:
- `results/vlm_evaluation.json` — full machine-readable results.
- `results/vlm_evaluation.md` — three thesis-paste-ready tables:
  1. Per-VLM extraction quality (slice A / B / mean F1, CER,
     hallucination).
  2. VLM vs. OCR head-to-head (OCR raw, OCR cleaned, all VLMs;
     F1 / CER / time / cost).
  3. Per-VLM variance (F1 mean ± std, CER mean ± std).

Downstream summarisation quality (ROUGE / BERTScore against the human
reference) is evaluated separately by `llm_summarization/evaluate.py`
on `output_vlm_<provider>/`, mirroring the existing OCR/audio
evaluation.

---

## 8. Limitations

- **Rate limits.** Gemini 2.5 Flash free tier ≈ 10 RPM, OpenAI
  tier-1 vision ≈ 500 RPM, Anthropic tier-1 ≈ 50 RPM, HF Inference
  highly variable. The runner throttles per provider; long runs may
  take hours of wall time even when total compute is small.
- **Internet dependency.** All four models are remote APIs; a
  network outage stalls the run. Resume-safe per-panorama checkpointing
  mitigates this.
- **Cost.** Estimated upper bound for the slice-level evaluation
  (2 panoramas × 4 models × 3 runs = 24 calls) is well under $1.
  Full-video extraction (~hundreds of panoramas × 3 paid models) is
  estimated at $5–10 total and will be confirmed with the user
  **before** Task 6 runs.
- **Determinism.** With `temperature=0` we still observe non-trivial
  output variance for vision tasks. The 3-run protocol exists to
  measure this rather than hide it.
- **Hallucination.** Unlike OCR, the VLM can produce a perfectly
  formatted headline that was never on screen. Hallucination rate is
  a primary reported metric, not a footnote.
- **Image size.** Some providers downscale images above ~2048 px on
  the long edge. v6 panoramas are 3000 px wide, so we send them at
  native size and let each provider apply its own scaling — this is
  recorded but not normalised away, since it's part of the
  provider's real-world behaviour.
- **Open-weight comparator is optional.** If the HF token is not
  provided or the endpoint is cold-starting, Qwen2-VL is silently
  skipped; tables note "n/a".
