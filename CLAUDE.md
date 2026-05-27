# CLAUDE.md

## Project
**Name:** ticker_extraction (repo: `broadcast-summarization` on GitHub)
**Thesis:** Multimodal Vision- and Audio-Based Extraction and LLM-Powered Summarization of TV News
**Author:** Masters student, GIU Berlin

This project implements three parallel extraction pipelines that feed one shared
9-LLM summarisation roster:

  1. **Visual path (OCR)** — 5-stage CV pipeline → LLM cleaning → 9-LLM summarisation
  2. **Visual path (VLM)** — single Vision-Language-Model call per panorama tile → JSON headlines
  3. **Audio path** — Whisper ASR of spoken broadcast → 2-level hierarchical 9-LLM summarisation

The OCR and VLM paths are independently evaluated against the same 39-headline
manually-annotated ground truth. ASR transcription quality is evaluated against
two separately-annotated audio slices. Each summarisation modality is scored
against its OWN human reference summary (`reference_summary.txt` for the visual
path, `reference_summary_audio.txt` for the audio path) using ROUGE + BERTScore.
A separate multi-engine OCR comparison (`ocr_comparison/`) benchmarks Tesseract,
EasyOCR, PaddleOCR, docTR, and CRAFT+TrOCR on the same panoramas.

## Project Structure

```
ticker_extraction/
├── ticker_extraction_v6/        ← FROZEN — 5-step OCR pipeline (DO NOT MODIFY)
│   ├── config.py, main.py, step1..step5_*.py
│   └── output/
│       ├── chunks/              (per-chunk outputs when running in chunked mode)
│       └── final/
│           ├── news_items.json              (improved raw v6 output, 48 items)
│           ├── news_items_cleaned.json      (LLM-cleaned output, 38 items)
│           └── pipeline_stats.json
│
├── llm_summarization/           ← FROZEN — 9-LLM summarization + evaluation (DO NOT MODIFY)
│   ├── config.py, prompt.py, summarize.py, evaluate.py
│   ├── reference_summary.txt        (human reference, 14h broadcast — visual path)
│   ├── reference_summary_audio.txt  (human reference, 14h broadcast — audio path)
│   ├── output/                  (summaries for original 27-min sample — DEV ONLY,
│   │                             scored against the wrong reference — see Phase 4)
│   ├── output_cleaned/          (canonical 14h cleaned-headlines summaries +
│   │                             evaluation_latest.json — scored against
│   │                             reference_summary.txt)
│   └── output_asr/              (canonical 14h ASR summaries + evaluation_latest.json,
│                                 merged latest.json from chunked + Ollama retries,
│                                 scored against reference_summary_audio.txt)
│
├── videos/                      ← raw input videos (.mp4) — gitignored except:
│   └── extract_demo_clip.py     (PyAV — produces 30s demo MP4 + WAV for site/)
│
├── pipeline/                    ← ticker post-processing scripts (NEW CODE)
│   ├── improve_segmentation.py  (re-segments v6 OCR text with better dedup)
│   ├── clean_news_items.py      (LLM error correction on raw v6 output)
│   ├── clean_vlm_headlines.py   (LLM error correction on VLM output)
│   ├── summarize_cleaned.py     (runs llm_summarization on cleaned items)
│   └── evaluate_cleaned_summaries.py  (wrapper: scores output_cleaned/ vs
│                                       reference_summary.txt — the canonical
│                                       14h visual summarisation eval)
│
├── validation/                  ← ground-truth validation (Part 1)
│   ├── regenerate_panorama.py   (rebuilds panorama PNG for chosen slice)
│   ├── validate_extraction.py   (computes P/R/F1/CER/WER against GT)
│   └── ground_truth/
│       ├── README.md            (annotation instructions)
│       ├── slice_A_panorama.png (hours 8:30–9:00, 15 MB)
│       ├── slice_A_headlines.txt (27 manually-annotated headlines)
│       ├── slice_B_panorama.png (hours 13:00–13:30, 13.6 MB)
│       └── slice_B_headlines.txt (28 manually-annotated headlines)
│
├── vlm_extraction/              ← VLM-based ticker extraction (parallel to OCR)
│   ├── README.md, PROGRESS_REPORT.md
│   ├── config.py                (model registry, tile geometry, API keys)
│   ├── prompts.py               (single shared prompt for all VLM providers)
│   ├── tiler.py                 (3000×87 px tiles with 1000 px overlap)
│   ├── extract.py               (top-level runner: --vlm X --slice Y --runs N)
│   ├── aggregate.py             (cross-tile dedup → canonical headlines_combined.json)
│   ├── evaluate.py              (P/R/F1 + CER + halluc rate vs slice GT;
│   │                             emits results/vlm_evaluation.{json,md})
│   ├── opensource_vlm.py        (open-source-VLM runner + bespoke evaluator that
│   │                             builds results/vlm_opensource_evaluation.md with
│   │                             the 3-way OCR vs Gemini vs Ministral comparison)
│   ├── adapters/                (base + gemini, openai, anthropic, huggingface,
│   │                             groq, mistral — all share the same prompt)
│   ├── output/
│   │   ├── _tiles/              (1845 cached 3000×87 tile PNGs)
│   │   ├── gemini/full_video/run_1/    (canonical paid Gemini run, 100 items)
│   │   ├── mistral/full_video/run_1/   (canonical open-weights Ministral 3 14B
│   │   │                                 run, 3214 raw items, 100% recall)
│   │   └── groq_scout/                  (partial / abandoned — kept for record)
│   └── .env (gitignored), .env.example
│
├── ocr_comparison/              ← multi-OCR comparison experiment (NEW)
│   ├── config.py                (engine list, paths, slice/tile params)
│   ├── run_ocr_engine.py        (top-level runner: --engine X --slice Y)
│   ├── segment_ocr_output.py    (wraps v6 step5 segmentation, no v6 edits)
│   ├── augment_dashes.py        (Tesseract PSM 6 dash augmentation pass)
│   ├── regenerate_pure.py       (rebuild pure-engine seg from cached words)
│   ├── evaluate_all.py          (emits BOTH tables: pure + augmented)
│   ├── resource_tracking.py     (psutil RSS probe)
│   ├── engines/
│   │   ├── base.py              (Word + EngineResult dataclasses)
│   │   ├── tesseract_engine.py
│   │   ├── easyocr_engine.py    (pure — no dash augmentation here)
│   │   ├── paddle_engine.py     (PaddleOCR v3 API, MKLDNN disabled)
│   │   ├── doctr_engine.py
│   │   ├── gemini_engine.py     (kept but excluded from results — free-tier RPM)
│   │   └── craft_trocr_engine.py (hybrid: CRAFT detector + TrOCR recognizer)
│   └── output/
│       ├── slice_A/             (per-engine raw_text, words, headlines,
│       │                         timing, segmentation_stats — both pure and
│       │                         _augmented variants; _tesseract_dashes cache)
│       ├── slice_B/ (same)
│       ├── comparison_report.json (both tables, machine-readable)
│       └── comparison_report.txt  (both tables, human-readable)
│
├── asr/                         ← audio pipeline (Part 2)
│   ├── extract_audio.py         (PyAV → 16 kHz mono WAV)
│   ├── transcribe.py            (faster-whisper chunked streaming)
│   ├── chunk_transcript.py      (time-based 15-min chunking)
│   ├── summarize_transcript.py  (2-level LLM summarization, 9 models)
│   ├── retry_level2.py          (retry Level-2 with simple backoff)
│   ├── retry_level2_chunked.py  (3-batch L2 retry for TPM-capped cloud models —
│   │                             recovers Llama 3.1 8B Groq, Qwen3 32B, HF Llama 3)
│   ├── run_ollama_l1_l2.py      (resume-safe L1+L2 backfill for the 2 Ollama
│   │                             models; supports --l2-only for partial runs;
│   │                             1200s read timeout + model warmup)
│   ├── merge_asr_runs.py        (combines original + all retries into a single
│   │                             canonical output_asr/latest.json then
│   │                             re-evaluates against reference_summary_audio.txt)
│   ├── evaluate_audio_summaries.py  (wrapper: scores output_asr/latest.json vs
│   │                                  reference_summary_audio.txt)
│   ├── extract_eval_slices.py   (extracts WAV slices for ASR-quality GT)
│   ├── transcribe_eval_slices.py (runs Whisper on the GT slices for scoring)
│   ├── eval/                    (ASR transcription-quality ground truth)
│   │   ├── slice_A_audio.wav    (Slice A audio for jiwer scoring)
│   │   ├── slice_A_groundtruth.txt (manually-annotated reference transcript)
│   │   ├── slice_A_whisper.txt  (whisper output on this slice)
│   │   ├── slice_B_audio.wav, slice_B_groundtruth.txt, slice_B_whisper.txt
│   └── output/
│       ├── audio_full.wav       (1.68 GB, 14.55 h mono 16 kHz — gitignored)
│       ├── transcript_full.json/.txt/.srt (13,517 segments)
│       ├── transcript_slice_A.{json,txt,srt} (Slice-A subset for the site)
│       ├── chunks/              (59 × 15-min chunk .txt files)
│       └── _chunks_transcript_full/  (intermediate 20-min resume files — gitignored)
│
├── design-system/               ← shared visual identity (NEW)
│   ├── tokens.css               (CSS custom properties used by both sites)
│   ├── components/              (shared React components)
│   │   ├── shared/              (StagePanel, ResultsHero, GTOverlay, Explorer,
│   │   │                         VLMComparison, VLMHeadlineSamples)
│   │   ├── visual/              (8 visual-pipeline stage components)
│   │   └── audio/               (4 audio-pipeline stage components)
│   └── README.md
│
├── site/                        ← defense site (NEW — Astro static export)
│   ├── astro.config.mjs, tailwind.config.mjs, package.json
│   ├── src/{layouts,components,pages,styles}
│   ├── public/{data,video}      (BUNDLED — committed so `npm run dev` works
│   │                             after a fresh clone; ~2.8 MB; regenerable
│   │                             via build_assets.py + instrument_slice_A.py)
│   ├── scripts/
│   │   ├── build_assets.py      (idempotent asset pipeline)
│   │   └── instrument_slice_A.py (real EasyOCR bboxes + scroll deltas + frame
│   │                              counts for site's animated stages)
│   └── README.md
│
├── webapp/                      ← interactive workbench (live, with backend)
│   ├── backend/                 (FastAPI: upload, job queue, pipeline orchestration)
│   ├── frontend/                (React + Vite — redesigned UI using design-system/)
│   ├── jobs/, uploads/          (runtime state — gitignored)
│   └── README.md
│
├── evaluation/                  ← determinism / variance experiment (3-run repeat)
│   ├── asr_evaluate.py          (ASR-quality scoring vs asr/eval/ ground truth)
│   ├── run_variance_summarization.py
│   └── aggregate_variance.py    (writes results/variance_report.{json,md})
│
├── results/                     ← evaluation outputs (canonical thesis tables)
│   ├── validation_report.json   (OCR P/R/F1 vs slice GT, 88.1 % F1 cleaned)
│   ├── asr_evaluation.json      (Whisper transcription quality, WER 6.6 %)
│   ├── asr_summary_evaluation.json    (8 LLMs vs reference_summary_audio.txt)
│   ├── visual_14h_summary_evaluation.json  (8 LLMs vs reference_summary.txt
│   │                                         on the 14h cleaned headlines)
│   ├── vlm_evaluation.json + .md       (Gemini paid run, full numbers + tables)
│   ├── vlm_opensource_evaluation.json + .md  (3-way OCR vs Gemini vs Ministral)
│   ├── ocr_comparison_report.json/.txt (5-engine comparison, two tables)
│   └── variance_report.json + .md      (3-run determinism check)
│
├── CLAUDE.md, TODO.md, README.md, PROGRESS_REPORT.md, requirements.txt
└── .gitignore
```

## Tech Stack
- **Language:** Python 3.9
- **CV / OCR:** OpenCV (cv2), EasyOCR, pytesseract, numpy
- **ASR:** faster-whisper (CTranslate2, int8 CPU), PyAV (audio extraction)
- **Text dedup / fuzzy matching:** rapidfuzz
- **LLM providers:** Groq, Ollama (local), Google Gemini, HuggingFace Inference, OpenAI, Cohere
- **Evaluation:** rouge-score, bert-score
- **Utilities:** tqdm, python-dotenv, requests, python-Levenshtein

## Frozen Modules (DO NOT MODIFY)

### `ticker_extraction_v6/` — 5-Step OCR Pipeline
Pipeline: frame extraction → scroll detection → panorama stitching → OCR (EasyOCR + Tesseract for dash detection) → segmentation.
Entry point: `python main.py --video <path> [--chunk-minutes N]` → produces `output/final/news_items.json` + `pipeline_stats.json`.
Chunked mode processes long videos in N-minute slices and merges results across chunks.
Performance: 18/19 recall on short AlJazeera sample; **97.4% recall, 84.8% F1 on 14-hour sample (raw)**; **92.3% recall, 88.1% F1 after LLM cleaning** (measured against 39 GT headlines from two 30-min slices).

### `llm_summarization/` — 9-LLM Summarization + Evaluation
Reads `ticker_extraction_v6/output/final/news_items.json` (or `_cleaned.json` / ASR chunks via wrappers), sends to 9 configured LLMs,
scores with ROUGE + BERTScore against a human reference summary.
Entry points: `summarize.py`, `evaluate.py`. API keys in `llm_summarization/.env`.

**Two human references live here:**
- `reference_summary.txt` (687 words, 14h Iran/Hormuz broadcast) — used for the
  **visual** path. Despite the misleading legacy "27-min sample" history, this
  reference describes the SAME broadcast that produced `news_items_cleaned.json`.
  Score via `pipeline/evaluate_cleaned_summaries.py` against `output_cleaned/`.
- `reference_summary_audio.txt` (1,908 words, same 14h broadcast, voice-over
  content) — used for the **audio** path. Score via
  `asr/evaluate_audio_summaries.py` against `output_asr/`.

The legacy `output/` folder is a March-2026 dev run on a DIFFERENT broadcast
(Saudi / Sudan / Ukraine stories). Its `evaluation_latest.json` exists for
historical record but **is not surfaced on the defense site**; the canonical
visual leaderboard is the `output_cleaned/` one.

### 9-LLM Roster (shared across both paths)

| Provider | Model | Notes |
|---|---|---|
| Groq | `llama-3.3-70b-versatile` | cloud, free tier |
| Groq | `llama-3.1-8b-instant` | cloud, free tier |
| Groq | `qwen/qwen3-32b` | cloud, free tier |
| Groq | `meta-llama/llama-4-scout-17b-16e-instruct` | cloud, free tier |
| Ollama (local) | `llama3.2` (3B) | local, 300 s read timeout |
| Ollama (local) | `llama3.1:8b` | local, often times out on 14h workloads |
| Google | `gemini-2.5-flash` | cloud, free tier (heavy rate limiting) |
| HuggingFace | `meta-llama/Meta-Llama-3-8B-Instruct` | cloud, free tier |
| Cohere | `command-r-08-2024` | cloud, free trial |

## Visual Pipeline Scripts (in `pipeline/`)

### `pipeline/improve_segmentation.py`
Re-processes the per-chunk OCR text from `ticker_extraction_v6/output/chunks/` to produce an improved raw `news_items.json`.
Uses all cycles from all chunks (not just one per chunk), aggressive fuzzy dedup with digit-stripping, ideal-length
tiebreakers (prefers 50–130 char single headlines over 150+ char merged items), internal-dash re-splitting.
Writes to `ticker_extraction_v6/output/final/news_items.json` (overwrites raw output but NOT v6 code).

### `pipeline/clean_news_items.py`
LLM post-processing pass. Reads `news_items.json`, sends to Gemini 2.5 Flash in batches of 15 items, asks it to
split merged headlines, complete truncated ones using cross-item context, and fix OCR typos. Writes to
`news_items_cleaned.json` (separate file — raw output preserved for evaluation).

### `pipeline/summarize_cleaned.py`
Wrapper that runs `llm_summarization/summarize.py` on `news_items_cleaned.json` (not the raw file) by monkey-patching
`config.NEWS_ITEMS_PATH` and `config.OUTPUT_DIR` at import time. Writes summaries to `llm_summarization/output_cleaned/`.
Does NOT modify llm_summarization source.

### `pipeline/clean_vlm_headlines.py`
Same idea as `clean_news_items.py` but for VLM output. Reads
`vlm_extraction/output/<provider>/<slice>/run_<n>/headlines_combined.json`,
sends to Gemini for over-merge / typo fixes, writes
`headlines_combined_cleaned.json`. Has a 50%-removal safety guard after one
aggressive run dropped 72 → 9 items.

### `pipeline/evaluate_cleaned_summaries.py`
Wrapper around `llm_summarization/evaluate.py` that scores
`output_cleaned/latest.json` against `reference_summary.txt` (the correct
14h-broadcast pairing — NOT the legacy 27-min `output/` run). Monkey-patches
`config.OUTPUT_DIR` to point at `output_cleaned/` so the eval lands there,
then mirrors the report to `results/visual_14h_summary_evaluation.json`.

**Validated visual summarisation results (8 of 9 LLMs, 14h cleaned headlines):**
- Gemini 2.5 Flash wins BERT-F1 = 0.865, ROUGE-L = 0.238
- Qwen3 32B wins ROUGE-L = 0.305
- Llama 3.1 8B (Ollama) failed L1 (CPU timeout) — same 8/9 outcome as the audio path

## Validation Pipeline (in `validation/`)

### `validation/regenerate_panorama.py`
Re-runs v6 steps 1–3 on a specific frame range of the video (e.g., the 30-min window corresponding to chunk 17)
and produces a single panorama PNG suitable for manual ground-truth annotation. Output goes to
`validation/ground_truth/slice_<letter>_panorama.png`.

### `validation/validate_extraction.py`
Loads manually-annotated `ground_truth/slice_*_headlines.txt`, loads both `news_items.json` and
`news_items_cleaned.json`, computes Precision / Recall / F1 at multiple fuzzy-match thresholds (60 / 70 / 80%),
plus Character Error Rate (CER), Word Error Rate (WER), exact-match count, missed headlines list, and
false-positive list. Writes a JSON report to `results/validation_report.json` and prints a summary table.

**Validated results (14h video, combined two 30-min slices = 39 GT headlines):**
- Raw v6: 97.4% recall, 75.0% precision, 84.8% F1 @70% threshold, CER=23.4%
- LLM-cleaned: 92.3% recall, 84.2% precision, **88.1% F1** @70% threshold, CER=13.9%

## OCR Comparison Experiment (in `ocr_comparison/`)

Head-to-head comparison of multiple OCR engines on the **same two ground-truth panoramas**
(Slice A hours 8:30–9:00, Slice B hours 13:00–13:30) used by Part 1 validation. Engines:
**Tesseract**, **EasyOCR**, **PaddleOCR (PP-OCRv5)**, **docTR (db_resnet50 + crnn_vgg16_bn)**,
**CRAFT+TrOCR (microsoft/trocr-base-printed)**. Gemini 2.5 Flash Vision adapter is implemented
but excluded from reported results (free-tier 20 RPM made timing measurements non-comparable).
CPU-only execution; thesis notes this constraint affects PaddleOCR and TrOCR especially.

### `ocr_comparison/run_ocr_engine.py`
Top-level runner: `python -m ocr_comparison.run_ocr_engine --engine X --slice {slice_A|slice_B|all}`.
Per (engine, slice) pair, saves: `<engine>_raw_text.txt`, `<engine>_words.json` (token list with
left-x positions for downstream dash insertion), `<engine>_timing.json` (wall-clock + RSS delta).
Skips already-processed pairs unless `--force`. Uses v6's slicing geometry (3000 px wide, 1500 px
stride) so every engine sees identical input chunks.

### `ocr_comparison/regenerate_pure.py`
Reads cached `<engine>_words.json`, sorts by x, concatenates → pure text → runs v6 segmentation.
Produces the **engine-alone** results (no Tesseract help). Shows that scene-text recognizers
(EasyOCR, PaddleOCR, docTR, TrOCR) skip the thin " - " delimiter glyph entirely → 0% F1 for
EasyOCR. Tesseract is the only engine with native delimiter detection.

### `ocr_comparison/augment_dashes.py`
Runs Tesseract PSM 6 across the panorama once (cached as `_tesseract_dashes.json` per slice),
then for each engine merges the engine's words with synthetic " - " tokens at dash x-positions,
re-runs segmentation. Produces `_augmented` variants of raw text / headlines / stats. Matches v6
production architecture (EasyOCR for recognition + Tesseract for delimiter detection).

### `ocr_comparison/evaluate_all.py`
Reuses `validation/validate_extraction.py` metric functions. Emits **two side-by-side tables**
into `comparison_report.json` and `comparison_report.txt`:
- **Pure engines** — what each engine produces alone
- **Engines + dash augmentation** — apples-to-apples recognition quality

### Validated comparison results (combined two slices = 39 GT headlines)

**Pure engines (engine alone):**
| Engine | Mean F1 | Mean CER |
|---|---|---|
| Tesseract | 85.4% | 10.0% |
| docTR | 63.9% | 20.1% |
| PaddleOCR | 8.5% | 45.0% |
| EasyOCR | 0.0% | — (no delimiters) |
| CRAFT+TrOCR | 0.0% | — (no delimiters) |

**Engines + Tesseract dash augmentation (v6 production architecture):**
| Engine | Mean F1 | Mean CER | Time | Peak RAM |
|---|---|---|---|---|
| **EasyOCR** | **91.3%** | **8.8%** | 3.9 min | 1.8 GB |
| Tesseract | 85.4% | 10.2% | 0.6 min | 4 MB |
| docTR | 80.3% | 12.0% | 6.0 min | 2.1 GB |
| PaddleOCR | 23.0% | 27.8% | 21.9 min | 596 MB |
| CRAFT+TrOCR (hybrid) | 29.9% | 29.8% | 52.4 min | 2.8 GB |

EasyOCR's win in Table 2 validates v6's production choice as the primary recognizer. PaddleOCR and
TrOCR underperform because their training distributions favor document text, not TV-ticker fonts.

## VLM Extraction (in `vlm_extraction/`)

Parallel to the OCR pipeline. Same panoramas, same prompt, but the VLM emits
JSON headlines in one call per tile (no OCR / segmentation stack).

### `vlm_extraction/extract.py`
Top-level: `python -m vlm_extraction.extract --vlm gemini --slice all --runs 3`.
Idempotent — already-done `headlines_<tile_id>.json` files are skipped
unless `--force`. Aggregates per-tile JSON via cross-tile fuzzy dedup into
`headlines_combined.json` (canonical schema, byte-compatible with
`news_items_cleaned.json`).

### `vlm_extraction/opensource_vlm.py`
Entry point for the open-source-VLM comparison. Wraps `extract.run_one_triple`
for the Mistral Ministral 3 14B adapter (open-weights, free Mistral
Experiment-plan tier — 1 B tokens / mo, no card). Sub-commands:
- `run` — execute extraction (full_video / slice_A / slice_B); resume-safe.
- `evaluate` — rebuild `results/vlm_opensource_evaluation.{json,md}` with the
  3-way comparison table (OCR cleaned vs Gemini paid vs Ministral free).

### VLM adapter roster
| Adapter | Provider | Tier | Status |
|---|---|---|---|
| `gemini_adapter.py` | Google AI Studio | paid tier 1 | ✅ canonical run (100 items, ≈$0.87 token-estimate / **€8.28 actual total billed**, 61 min) |
| `mistral_adapter.py` | Mistral La Plateforme | free Experiment | ✅ canonical run (3214 items, $0.00, 114 min) |
| `openai_adapter.py` | OpenAI | paid (billing not active) | ⏸ deferred |
| `anthropic_adapter.py` | Anthropic | paid | ⏸ deferred |
| `groq_adapter.py` | Groq Llama 4 Scout | free | ⚠️ partial (12/1845) |
| `huggingface_adapter.py` | HF Inference (Qwen2-VL) | requires partner provider | ❌ blocked on free tier |

### Validated VLM extraction results (combined slice GT, 39 headlines)
| Pipeline | Items | F1@70 | Recall | Precision | Cost | Wall |
|---|---|---|---|---|---|---|
| OCR LLM-cleaned (v6 + Gemini cleaner) | 38 | **88.1%** | 92.3% | 84.2% | ≈$0.05† | ~12 min |
| Gemini 2.5 Flash (paid) | 100 | 64.9% | 100% | 48.0% | ≈$0.87† | 61 min |
| Ministral 3 14B (open-weights, free) | 3214 | **71.0%** | **100%** | 55.3% | **$0.00** | 114 min |

†Cost column = code-computed token estimate per run. The **actual money
billed by Google across all VLM experiments was €8.28** (authoritative
figure from the user's billing statement); the token estimates
under-report real spend and their per-run accuracy is unverified. Cite
**€8.28** as the VLM approach's real cost in the thesis.

The open-weights free VLM beats paid Gemini on F1 with the same prompt
and tiles, at zero cost; both VLMs still trail the OCR pipeline by ~17 pp
because they over-emit candidates (lower precision).

## Web Surfaces (in `site/`, `webapp/frontend/`, `design-system/`)

Two parallel web deliverables share one design system:

### `site/` — Defense site (read-only, static)
Astro 4 static export with React islands for animation. Six routes:
landing, `/visual` (8 animated OCR stage panels), `/vlm` (3-way VLM
comparison + per-model headline samples), `/audio` (5 animated stage
panels), `/results` (P/R/F1/CER + parallel ROUGE/BERTScore leaderboards
for visual + audio summarisation + GT-vs-extracted overlay), `/explore`
(tabbed data browser: headlines / transcript / summaries / raw OCR with
search). Zero runtime API calls — everything is precomputed by
`scripts/build_assets.py`. Uses a 30-sec demo MP4 + WAV extracted from
the source video at 08:30:00 (matches Slice A so the OCR walkthrough
lines up with the annotated panorama).

The `instrument_slice_A.py` companion script imports v6's modules read-only
and persists the artefacts v6 doesn't save (per-frame scroll deltas, real
EasyOCR word-level bboxes on the first 3000-px panorama segment, real frame
counts). The site's animated stages render the **real numbers** from these
when present (look for the small "live data" badge), and fall back to clearly
labelled representative numbers otherwise. The eval-bars chart on `/visual`
uses real ROUGE-L + BERTScore F1 from
`llm_summarization/output/evaluation_latest.json`.

Built bundle (`site/public/data/` + `site/public/video/`, ~2.8 MB total) is
committed so a fresh clone can run `cd site && npm install && npm run dev`
with no source video required.

### `webapp/frontend/` — Interactive workbench (live, with backend)
Existing React+Vite UI redesigned to match the defense site's visual
identity. Same FastAPI backend (`webapp/backend/`) — UI rewrite only, no
backend changes. Adds Framer Motion + Lucide + dark mode + animated
status pills + drag-drop upload zone. All API endpoints unchanged.

### `design-system/` — Shared (no build step)
Single `tokens.css` (light + dark, path-accent colors, motion tokens,
spacing/radii/shadows) imported by both projects. React stage components
live under `components/visual/` and `components/audio/`; both consumers
import them via relative path. Both projects' Tailwind configs glob the
design-system folder for utility extraction.

### `videos/extract_demo_clip.py`
PyAV-based extractor that produces `videos/demo_clip_30s.mp4` (480p H.264,
~1.3 MB) and `videos/demo_clip_30s.wav` (16 kHz mono PCM, ~940 KB) from
08:30:00→08:30:30 of the source video. Both files are gitignored.
Idempotent — skips if outputs already exist.

## Audio Pipeline (in `asr/`)

### `asr/extract_audio.py`
Uses PyAV (no ffmpeg CLI needed) to extract the audio track from a .mp4 and convert to 16 kHz mono WAV
(Whisper's preferred format). Supports `--start` and `--duration` for testing on time-bounded slices.
Full 14 h produces `audio_full.wav` (~1.68 GB, ~10 min extraction time).

### `asr/transcribe.py`
Runs faster-whisper (`small` int8 English model on CPU) with **internal chunked streaming** —
reads the WAV file in 20-minute chunks via the `wave` module to avoid the MemoryError that occurs
when faster-whisper tries to load the full 14 h audio array into feature-extractor memory (7.4 GB).
Per-chunk intermediate JSON files are saved to `asr/output/_chunks_transcript_full/` for resume-safety.
Produces `transcript_full.json/.txt/.srt`.
Full 14 h: **13,517 segments in 284 min (3.07× realtime)**.

### `asr/chunk_transcript.py`
Splits the transcript into **59 × 15-minute chunks** based on segment timestamps. Each chunk is ~2,000–2,600 words
of plain text. Output: `asr/output/chunks/chunk_NNN.txt`.

### `asr/summarize_transcript.py`
**Two-level hierarchical summarization** to handle the ~150 KB transcript that won't fit in any single LLM context:
- **Level 1**: each 15-min chunk × each of 9 LLMs → 1 short paragraph. 59 × 9 = 531 API calls.
- **Level 2**: each LLM's 59 Level-1 summaries concatenated and fed back to the same LLM →
  final 5–7 paragraph summary covering the day's major stories.

Monkey-patches llm_summarization's config path (does NOT modify the frozen module). Output in
`llm_summarization/output_asr/`. **5 of 9 models produced complete Level-2 summaries;**
Groq 70B and Gemini hit TPM limits on 12 K-token Level-2 input; Ollama local models timed out.

### `asr/retry_level2.py`
Retries Level-2 only (using already-computed Level-1 intermediate JSON) with exponential backoff
for rate-limited providers. Gets 4 additional successful Level-2 summaries beyond the initial run.

### `asr/retry_level2_chunked.py`
For models whose single-call Level-2 hits the free-tier TPM cap (6 K TPM on
Groq 8 B / Qwen3 32 B, HF Inference 503s) but whose **Level-1 is complete
(59/59)**: splits the 59 L1 summaries into 3 batches of ~20, summarises each
batch (~150 words), then summarises the 3 batch-summaries into the final L2.
Each call now fits inside the TPM budget. Recovered 3 of 3 targeted cloud
models in ~10 min total.

### `asr/run_ollama_l1_l2.py`
Resume-safe Ollama Level-1 + Level-2 backfill for the two local models.
Direct Ollama HTTP API call with **1200 s read timeout** + explicit model
warmup, persists `level1_ollama_inflight.json` every 5 chunks. Run with
`--l2-only` to skip L1 and run only the final reduction (useful when L1 is
already done from a previous run, or when one model's L1 has hopelessly
timed out). Llama 3.2 3B reached 59/59 L1 in ~6.5 h of CPU; Llama 3.1 8B
stalled at 4/59 (consistent 1200 s timeouts) and is reported as the single
documented failure.

### `asr/merge_asr_runs.py`
Combines the original `summaries_*.json`, `summaries_retry_*.json`,
`summaries_retry_chunked_*.json`, and `summaries_retry_ollama_*.json` into
one canonical `output_asr/latest.json`. Later, more-successful runs win on a
per-model basis. After merging, runs `asr/evaluate_audio_summaries.py` so
`evaluation_latest.json` and `results/asr_summary_evaluation.json` reflect
the merged 8-of-9-model set.

### `asr/evaluate_audio_summaries.py`
Thin wrapper around `llm_summarization/evaluate.py` that scores
`output_asr/latest.json` against `reference_summary_audio.txt` and mirrors
the report to `results/asr_summary_evaluation.json`.

### `asr/extract_eval_slices.py` + `asr/transcribe_eval_slices.py`
Extract two short WAV slices from the full audio, transcribe with Whisper,
then compare against the **manually-annotated** `slice_{A,B}_groundtruth.txt`
to compute Whisper-quality WER/CER. Result: **WER 6.6 %, CER 4.8 %** combined
on a 1,567-word reference (`results/asr_evaluation.json`).

### Validated audio summarisation results (8 of 9 LLMs, vs 1908-word audio reference)
- Command-R (Cohere) wins BERT-F1 = 0.778 (shortest summary, 367 words)
- Llama 4 Scout 17B (Groq) wins ROUGE-L = 0.144 (longest summary, 742 words)
- All 4 Groq models + Gemini + Cohere + HF + Ollama 3B succeeded (8 / 9)
- Llama 3.1 8B (Ollama) failed L1 — commodity-CPU latency exceeds 1200 s timeout

## Rules Claude MUST Follow
1. **Never modify** `ticker_extraction_v6/` or `llm_summarization/` — they are frozen working code.
   (Writing to their `output/` folders is allowed — that's data, not code.)
2. **Always use relative paths** from the project root. New Python scripts in `pipeline/`,
   `validation/`, `asr/`, `ocr_comparison/`, or `site/scripts/` should use
   `PROJECT_DIR = Path(__file__).parent.parent.resolve()` (or `.parent.parent.parent` for
   scripts nested two levels deep, e.g. `site/scripts/`).
3. **Always wrap** file I/O and API calls in `try/except` with meaningful error messages.
4. **Always log errors** to a `logs/` folder at project root (create if missing).
5. **Always skip already-processed files** — check for existing output before processing.
6. Before running any destructive command (delete, overwrite, force-push), confirm with the user.
7. When adding new code, explain the plan first before writing it.

## Folders To Ignore
- `ticker_extraction_v6/output/chunks/*/panorama/` (cleaned up to save disk)
- `ticker_extraction_v6/output/chunks/*/ticker_frames/` (gitignored intermediates)
- `asr/output/_chunks_transcript_full/` (intermediate resume files)
- `logs/instrument_slice_A_work/` (heavy temp frames from the instrumented run)
- any `ticker_extraction_v2/..v5/` legacy folders if they reappear
