# Progress Report: Vision-Based Extraction and LLM-Powered Summarization of TV News Tickers

**Project**: Master's Thesis — GIU Berlin
**Author**: Marwan Mohamed
**Last updated**: 2026-05-13

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1 — Vision-Based Ticker Extraction (v6 Pipeline)](#2-phase-1--vision-based-ticker-extraction-v6-pipeline)
3. [Phase 2 — Scaling to 14-Hour Video (Chunked Mode + Dual-Engine OCR)](#3-phase-2--scaling-to-14-hour-video-chunked-mode--dual-engine-ocr)
4. [Phase 3 — Post-Processing (Improved Segmentation + LLM Cleaning)](#4-phase-3--post-processing-improved-segmentation--llm-cleaning)
5. [Phase 4 — Ticker Summarization (9-LLM Comparison)](#5-phase-4--ticker-summarization-9-llm-comparison)
6. [Phase 5 — Validation (Ground Truth + Metrics)](#6-phase-5--validation-ground-truth--metrics)
7. [Phase 6 — Audio Pipeline (ASR + LLM Summarization)](#7-phase-6--audio-pipeline-asr--llm-summarization)
8. [Phase 7 — Multi-Engine OCR Comparison](#8-phase-7--multi-engine-ocr-comparison)
9. [Phase 8 — VLM-Based Ticker Extraction](#9-phase-8--vlm-based-ticker-extraction)
10. [Phase 9 — Web Surfaces (Defense Site + Interactive Workbench)](#10-phase-9--web-surfaces-defense-site--interactive-workbench)
11. [End-to-End Pipeline Diagram](#11-end-to-end-pipeline-diagram)
12. [Project Structure](#12-project-structure)
13. [Technical Glossary](#13-technical-glossary)

---

## 1. Project Overview

This thesis implements a complete pipeline for automatic extraction and summarization of TV news content from two complementary modalities:

- **Visual modality**: the scrolling news ticker at the bottom of the screen (OCR + VLM)
- **Auditory modality**: the spoken broadcast content (ASR — Automatic Speech Recognition)

The final output for each modality is a ranked comparison of 9 different LLMs summarizing the day's news, benchmarked against a human reference using ROUGE + BERTScore.

The visual modality is now investigated under **two parallel approaches**:
1. **OCR pipeline** — traditional 5-stage CV approach (frames → scroll → panorama → OCR → segment), validated and now baseline.
2. **VLM pipeline** — single-shot Vision-Language Model extraction (panorama tile → JSON headlines), added to compare against OCR head-to-head.

A separate **multi-engine OCR comparison** (Phase 7) benchmarks Tesseract, EasyOCR, PaddleOCR, docTR, and CRAFT+TrOCR on identical input, validating the engine choice in the production pipeline.

### Input videos

- **`AlJazeera_sample.mp4`** — 27-minute Al Jazeera sample (1280×720, 30fps) — used for initial pipeline development
- **`AlJazeera_14hrs_without_edits.mp4`** — 14.55-hour Al Jazeera broadcast (1,571,786 frames) — the main experimental dataset

### Pipeline summary (all modalities)

```
VIDEO
  │
  ├── VISUAL (OCR): frames → scroll detection → panorama → dual-engine OCR
  │                  → segmentation → improved segmentation → LLM cleaning
  │                  → 9-LLM summarization → evaluation
  │
  ├── VISUAL (VLM): v6 panoramas → tiling → Gemini 2.5 Flash per-tile JSON
  │                  → cross-tile dedup → (optional cleanup) → evaluation
  │
  └── AUDIO:        WAV extraction → Whisper ASR → time-based chunking
                    → 2-level LLM summarization → evaluation
```

---

## 2. Phase 1 — Vision-Based Ticker Extraction (v6 Pipeline)

Reference: `ticker_extraction_v6/` (frozen working code, do not modify).

The v6 pipeline processes a video in 5 sequential steps to extract individual news headlines from the scrolling ticker bar.

### Step 1 — Frame extraction (`step1_extract_ticker.py`)

Samples every Nth frame (N=5, effective 6 fps), crops the bottom 12% (y=88–100%, x=14–96% to skip the AlJazeera logo and right edge), skips frames darker than brightness 20 (black transitions). Produces a 1049×87 px strip per saved frame.

### Step 2 — Scroll detection (`step2_scroll_detection.py`)

Measures pixel-level horizontal displacement between consecutive ticker frames using `cv2.matchTemplate` with TM_CCOEFF_NORMED on the middle 50% of the image. Rejects matches with confidence < 0.3. Produces cumulative scroll offsets so later stitching can place each frame at the correct horizontal position.

### Step 3 — Panorama stitching (`step3_stitch_image.py`)

Builds a single long horizontal image by placing each frame at its cumulative offset, first-write-wins per column. Splits into ≤50,000-pixel chunks with 200 px overlap. For the 14h video, the total panorama is **3,723,253 pixels wide** (75 chunks).

### Step 4 — OCR (`step4_ocr.py`) — Dual-Engine

Runs OCR on each panorama chunk with a 3000 px sliding window (stride 1500), keeping only center-region words to prevent boundary duplicates.

**Dual-engine approach** (added in Phase 2):
- **EasyOCR** — primary text recognition (better word-level accuracy, 3× upscale)
- **Tesseract PSM 6** — secondary pass **only to detect `" - "` delimiters** between headlines (EasyOCR misses these because they're tiny relative to the text)

The dashes detected by Tesseract are inserted into EasyOCR's word stream at matching x-positions, producing text like `"ISRAELI FORCES AND SETTLERS HAVE KILLED 1,138 PALESTINIANS... - RESCUE CHARITIES SAY 71 PEOPLE REMAIN MISSING..."` that Step 5 can then split reliably.

### Step 5 — Segmentation + deduplication (`step5_segment.py`)

Splits the OCR text on the detected delimiter, quality-filters each candidate (uppercase ratio, real-word ratio, length), and deduplicates with rapidfuzz. Number-aware: items with different significant numbers are kept separate. Post-dedup merge-artifact removal identifies items whose halves match two different existing headlines.

**Results on 27-minute sample**: 18–19 unique news headlines (100% recall vs ground truth).

---

## 3. Phase 2 — Scaling to 14-Hour Video (Chunked Mode + Dual-Engine OCR)

Running the original v6 pipeline on the full 14-hour video produced **zero final news items** despite successful OCR. Two root causes were identified and fixed:

### Problem 1 — Panorama too wide for single-pass OCR

The 3.7-million-pixel panorama exceeded EasyOCR's reliable working range, producing significant text corruption at slice boundaries.

**Fix: `--chunk-minutes` flag in `main.py`**

Added chunked processing mode. Instead of one giant panorama, the pipeline runs the full 5 steps on **30-minute slices** of the video (30 chunks × 30 min). Each chunk's panorama is ~250,000 px wide — within the sweet spot for OCR. Per-chunk results are merged with cross-chunk deduplication at the end.

Chunks are saved to `output/chunks/chunk_NNN/` and the pipeline is resume-friendly — if killed mid-run, already-processed chunks are skipped.

### Problem 2 — No `" - "` delimiter detected in OCR output

EasyOCR treats the small dashes between ticker headlines as whitespace, so `step5`'s delimiter-based segmentation had nothing to split on. A regex-based fallback existed but was hardcoded to the original 27-min sample's specific headlines.

**Fix: dual-engine OCR** (described in Step 4 above).

Tesseract PSM 6 successfully detects dashes that EasyOCR misses. After this fix, Step 5's original delimiter path works for any video.

### Result on 14-hour video (vision extraction only)

- **2,137 raw items** across 30 chunks
- **~48 unique headlines** after cross-chunk dedup (before post-processing)
- Processing time: ~10 hours on CPU for the full chunked pipeline

---

## 4. Phase 3 — Post-Processing (Improved Segmentation + LLM Cleaning)

Two-stage refinement on top of the raw v6 output (both are in `pipeline/`, both leave v6 untouched).

### Stage A — `improve_segmentation.py`

Reprocesses the per-chunk OCR text (`output/chunks/*/ocr/full_ocr_text.txt`) with:
- All cycles from all chunks (not just one "best cycle" per chunk)
- Aggressive fuzzy dedup with digit-stripping (OCR corrupts numbers like `72,292 → 7,292 → 72,290`)
- Ideal-length tiebreaker (prefers 50–130 char single headlines over 150+ char merged items)
- Internal-dash re-splitting (catches any items still containing `" - "` after primary segmentation)
- Good-vs-merged classification — items ≤140 chars go in the primary pool; longer "merged" items are kept only if their content isn't already covered

**Output**: 48 cleaner headlines → written back to `ticker_extraction_v6/output/final/news_items.json`

### Stage B — `clean_news_items.py` (LLM post-processing)

Sends the 48 raw items to **Gemini 2.5 Flash** in batches of 15 with a prompt that asks the LLM to:
- Split any items containing two merged headlines
- Complete truncated items using context from other items in the list
- Fix OCR typos (`HORMU → HORMUZ`, `KilLed → KILLED`, etc.)
- Drop fragments < 30 chars

**Output**: 38 cleaned, complete, readable headlines → written to `news_items_cleaned.json` (separate file — raw v6 output preserved for evaluation).

Both files are kept so the thesis can report metrics for vision-only vs vision+LLM stages independently.

---

## 5. Phase 4 — Ticker Summarization (9-LLM Comparison)

Reference: `llm_summarization/` (frozen) + wrapper `pipeline/summarize_cleaned.py`.

The same 9-LLM summarization pipeline used for the original 27-min sample is re-run on the LLM-cleaned 14h headlines via a wrapper script that monkey-patches the config path at import time (does not modify the frozen module).

### Models evaluated

| Provider | Model | Tier |
|---|---|---|
| Groq | `llama-3.3-70b-versatile` | cloud, free |
| Groq | `llama-3.1-8b-instant` | cloud, free |
| Groq | `qwen/qwen3-32b` | cloud, free |
| Groq | `meta-llama/llama-4-scout-17b-16e-instruct` | cloud, free |
| Ollama (local) | `llama3.2` (3B) | local |
| Ollama (local) | `llama3.1:8b` | local |
| Google | `gemini-2.5-flash` | cloud, free tier |
| HuggingFace | `meta-llama/Meta-Llama-3-8B-Instruct` | cloud, free tier |
| Cohere | `command-r-08-2024` | cloud, free trial |

All models receive the same system prompt ("professional news editor, neutral journalistic style, one paragraph, no hallucinations") and user prompt (numbered list of the 38 cleaned headlines). Temperature=0.3, max_tokens=1024.

### Results (ticker summarization, 14-hour broadcast)

**8 of 9 models succeeded** — only Ollama `llama3.1:8b` timed out (300s CPU limit). Summaries range from 1,226 chars (terse) to 3,024 chars (verbose). Output stored in `llm_summarization/output_cleaned/latest.json`.

### Evaluation against the human visual reference (`reference_summary.txt`)

Scored via `pipeline/evaluate_cleaned_summaries.py` (monkey-patches the frozen
evaluator to write into `output_cleaned/` and mirrors to
`results/visual_14h_summary_evaluation.json`).

| Rank | Model | ROUGE-1 | ROUGE-L | BERT-F1 | Words |
|---|---|---|---|---|---|
| 1 | Gemini 2.5 Flash | 0.428 | 0.238 | **0.865** | 453 |
| 2 | Llama 4 Scout 17B (Groq) | 0.460 | 0.237 | 0.853 | 294 |
| 3 | Llama 3.3 70B (Groq) | 0.422 | 0.208 | 0.850 | 290 |
| 4 | Llama 3 8B (HuggingFace) | 0.341 | 0.186 | 0.850 | 215 |
| 5 | Command-R (Cohere) | 0.351 | 0.159 | 0.844 | 200 |
| 6 | Qwen3 32B (Groq) | 0.388 | **0.305** | 0.844 | 288 |
| 7 | Llama 3.2 3B (Ollama) | 0.330 | 0.184 | 0.839 | 213 |
| 8 | Llama 3.1 8B (Groq) | 0.290 | 0.170 | 0.839 | 194 |
| — | Llama 3.1 8B (Ollama) | — | — | — | failed L1 (CPU timeout) |

**Note on the legacy 27-min sample.** The earlier 9-LLM run in
`llm_summarization/output/` was performed against a different broadcast
(March 2026: Saudi / Sudan / Ukraine stories), but it was originally scored
against this same `reference_summary.txt`. That pairing was incorrect —
the reference describes the 14h Iran/Hormuz broadcast — so the 27-min run
is retained on disk for historical record only and is not surfaced on the
defense site. The canonical visual leaderboard is the table above.

---

## 6. Phase 5 — Validation (Ground Truth + Metrics)

Reference: `validation/` folder.

### Methodology

Manually-annotated ground truth against which the extraction pipeline is benchmarked. Standard precision / recall / F1 + OCR accuracy metrics — the evaluation methodology used in every information-extraction paper.

### Ground truth creation (`validation/regenerate_panorama.py`)

Regenerates the v6 panorama PNG for two selected 30-min slices of the 14h video (steps 1–3 only, no OCR). The resulting 15 MB wide PNG shows every scrolled ticker frame stitched end-to-end, readable at leisure.

- **Slice A** — chunk 17, hours 8:30–9:00
- **Slice B** — chunk 26, hours 13:00–13:30

The two slices come from different hours to test whether the pipeline works consistently over time and captures breaking news that may have rotated in/out.

Manual annotation produced:
- `slice_A_headlines.txt` — **27 unique headlines** (what appeared on screen)
- `slice_B_headlines.txt` — **28 unique headlines**
- **Combined (union)**: **39 unique headlines** (16 stories appeared in both slices)

### Metrics (`validation/validate_extraction.py`)

Computes, for each stage (raw v6, LLM-cleaned, VLM raw, VLM cleaned) × each slice (A, B, combined):

- **Precision** at 60% / 70% / 80% fuzzy-match thresholds
- **Recall** at same thresholds
- **F1** at same thresholds
- **Character Error Rate (CER)** on pairs matched at 70%
- **Word Error Rate (WER)** on pairs matched at 70%
- **Exact match count** (≥95% similarity)
- **Coverage** (% of GT with any ≥50% match)
- **Missed headlines** list (specific GT items not found)
- **False-positive** list (extracted items with no GT match — some may be legitimate headlines from unsampled parts of the 14h)

### Key results (OCR pipeline)

| Stage | GT | Ext | P@70 | **R@70** | **F1@70** | Exact | CER | WER |
|---|---|---|---|---|---|---|---|---|
| **Raw v6** (slice A) | 27 | 48 | 58.3% | 96.3% | 72.7% | 7 | 20.3% | 31.0% |
| **Raw v6** (slice B) | 28 | 48 | 60.4% | **100.0%** | 75.3% | 5 | 25.2% | 37.1% |
| **Raw v6** (combined) | 39 | 48 | 75.0% | **97.4%** | 84.8% | 7 | 23.4% | 35.9% |
| **LLM-cleaned** (slice A) | 27 | 38 | 63.2% | 88.9% | 73.9% | 12 | 11.2% | 16.2% |
| **LLM-cleaned** (slice B) | 28 | 38 | 65.8% | 89.3% | 75.8% | 11 | 13.6% | 18.0% |
| **LLM-cleaned** (combined) | 39 | 38 | **84.2%** | 92.3% | **88.1%** | **14** | **13.9%** | **19.6%** |

### Thesis takeaways

1. **Vision-based extraction (raw v6) achieves 97.4% recall on the combined GT** — the pipeline captures essentially all real headlines.
2. **Slice B hits 100% recall**: every single headline that appeared in the chunk 26 panorama was extracted.
3. **LLM post-processing trades ~5% recall for a +13% F1 and halved CER/WER.**  It rejects garbled fragments more aggressively (hence lower recall) but nearly doubles character-level accuracy.
4. The ~15% absolute precision gap at @70 threshold is largely explained by the **false-positive caveat**: some "FPs" are legitimate headlines from the 13 unsampled hours of video — the reported FP count is an upper bound.

Full machine-readable report: `results/validation_report.json`.

---

## 7. Phase 6 — Audio Pipeline (ASR + LLM Summarization)

Reference: `asr/` folder — completely independent from the ticker pipeline. Captures what the presenters actually said on-air (interviews, analysis, live reports) which is richer than ticker headlines.

### Step A — Audio extraction (`asr/extract_audio.py`)

Uses **PyAV** (bundled with faster-whisper, no ffmpeg CLI needed) to extract the audio track from the .mp4 into a 16 kHz mono WAV file (Whisper's preferred format). Supports optional time-window arguments for testing on slices.

- Input: 14.55 h .mp4
- Output: `asr/output/audio_full.wav` — **1.68 GB, 873.2 minutes**
- Time: ~10 minutes

### Step B — Transcription (`asr/transcribe.py`)

Uses **faster-whisper** (CTranslate2 backend, int8 quantization, `small` English-only model).

**Problem encountered**: faster-whisper loads the entire audio array into memory for feature extraction before transcribing, which failed with `MemoryError: 7.47 GiB` on the full 14 h audio.

**Solution**: chunked streaming transcription. The script reads the WAV file in **20-minute chunks** via Python's `wave` module, passes each chunk as a numpy array to the model, applies time-offset correction to each segment's timestamps, and saves per-chunk intermediate JSON files (for resume-friendly execution).

**Results on 14 h audio (CPU, Whisper `small` int8)**:
- 44 internal chunks × 20 min each
- **13,517 segments** transcribed
- **Processing time: 4h 44m**
- **Speed: 3.07× realtime**

Outputs:
- `asr/output/transcript_full.json` — machine-readable (segments with timestamps, logprobs, no-speech probabilities)
- `asr/output/transcript_full.txt` — human-readable one-segment-per-line with timestamps
- `asr/output/transcript_full.srt` — SubRip subtitles (can be played alongside the video)

### Step C — Time-based chunking (`asr/chunk_transcript.py`)

Splits the 13,517-segment transcript into **59 × 15-minute chunks**. Each chunk is ~2,000–2,600 words of plain text — an ideal size for LLM prompts.

Output: `asr/output/chunks/chunk_000.txt` … `chunk_058.txt`.

Why 15-minute chunks: fits easily within any LLM's context window, gives the summarizer enough material per chunk to produce a meaningful paragraph, and aligns with typical TV-news segment pacing.

### Step D — Two-level LLM summarization (`asr/summarize_transcript.py`)

Direct flat summarization of 59 × 2,500 words (~150 KB) won't fit in most LLMs' input windows, so a **two-level hierarchical summarization** is used:

**Level 1** — Each 15-min chunk × each of the 9 LLMs → one short paragraph summary. Total: 59 × 9 = **531 API calls**. Results stored in `llm_summarization/output_asr/level1_<timestamp>.json`.

**Level 2** — Each LLM's 59 Level-1 summaries are concatenated and fed back to that same LLM with a new prompt: *"Write a 5–7 paragraph final summary covering the day's major stories and themes in chronological order."*  Result: one final multi-paragraph summary per LLM.

Same 9 LLMs as ticker summarization. Same monkey-patch pattern (no modification of `llm_summarization/`). Output in a separate folder: `llm_summarization/output_asr/`.

### Level-1 coverage per model (out of 59 chunks)

| Model | Success |
|---|---|
| Llama 3.1 8B (Groq) | 59 / 59 ✅ |
| Qwen3 32B (Groq) | 59 / 59 ✅ |
| Llama 4 Scout 17B (Groq) | 59 / 59 ✅ |
| Llama 3 8B (HuggingFace) | 59 / 59 ✅ |
| Command-R (Cohere) | 59 / 59 ✅ |
| Llama 3.3 70B (Groq) | 36 / 59 (hit TPM limit) |
| Gemini 2.5 Flash | 21 / 59 (heavy rate limiting) |
| Llama 3.2 3B (Ollama local) | 2 / 59 (CPU timeouts) |
| Llama 3.1 8B (Ollama local) | 0 / 59 (CPU timeouts) |

### Level-2 final summary results (after full retry cascade)

After three rounds of recovery work (original → simple retry → chunked-L2
for TPM-capped cloud models → Ollama L1+L2 backfill), **8 of 9 models**
produced complete Level-2 summaries:

| Model | Level-1 coverage | Path to L2 success |
|---|---|---|
| Llama 3.3 70B (Groq) | 36/59 | original run with partial L1 |
| Llama 4 Scout 17B (Groq) | 59/59 | original run |
| Gemini 2.5 Flash | 21/59 | original run with partial L1 |
| Command-R (Cohere) | 59/59 | original run |
| Llama 3.1 8B (Groq) | 59/59 | **recovered via chunked-L2** (3-batch reduction avoids 6K TPM cap) |
| Qwen3 32B (Groq) | 59/59 | **recovered via chunked-L2** |
| Llama 3 8B (HuggingFace) | 59/59 | **recovered via chunked-L2** |
| Llama 3.2 3B (Ollama local) | **59/59** (filled in) | **L1+L2 backfilled** by `run_ollama_l1_l2.py` (~6.5 h CPU) |
| Llama 3.1 8B (Ollama local) | 4/59 | ❌ aborted — every chunk past #4 hit the 1200 s CPU timeout |

**Why the chunked-L2 recovery works.** The original Level-2 input is ~12 K
tokens (59 concatenated L1 paragraphs). Groq's free tier caps Llama 3.1 8B
and Qwen3 32B at **6 K TPM**, so a single L2 call rate-limits immediately
even with backoff. `asr/retry_level2_chunked.py` splits the 59 paragraphs
into 3 batches of 20, summarises each batch into one consolidated paragraph
(~3 K-token input, ~600-token output — comfortably inside the TPM budget),
then summarises the 3 batch-summaries into the final L2. Three calls per
model × backoff pacing keeps the running token rate under the limit.

**Why one Ollama model still fails.** Llama 3.1 8B (4.9 GB) on commodity
CPU takes ~14-17 min per 3 K-token chunk and creeps past the 1200 s read
timeout from chunk 5 onward — every retry produces the same `ReadTimeout`.
This is a hardware-level limit, not a software bug; documented as the
single failure in both leaderboards.

### Evaluation against the human audio reference (`reference_summary_audio.txt`)

Reference: 1,908 words, topic-grouped with a chronological backbone,
covering audio-only content (Pope's Easter homily, Kashmir / France /
Senegal voice-over reports, rescue-operation interview narrative) that
never appears in the ticker bar. Scored via `asr/merge_asr_runs.py` (which
calls `asr/evaluate_audio_summaries.py`). Report mirrored to
`results/asr_summary_evaluation.json`.

| Rank | Model | ROUGE-1 | ROUGE-L | BERT-F1 | Words |
|---|---|---|---|---|---|
| 1 | Command-R (Cohere) | 0.219 | 0.103 | **0.778** | 367 |
| 2 | Qwen3 32B (Groq) | 0.227 | 0.095 | 0.770 | 413 |
| 3 | Llama 3.2 3B (Ollama) | 0.225 | 0.095 | 0.768 | 444 |
| 4 | Llama 3.3 70B (Groq) | 0.307 | 0.132 | 0.768 | 540 |
| 5 | Llama 4 Scout 17B (Groq) | 0.373 | **0.144** | 0.767 | 742 |
| 6 | Llama 3.1 8B (Groq) | 0.271 | 0.115 | 0.767 | 449 |
| 7 | Gemini 2.5 Flash | 0.296 | 0.117 | 0.767 | 557 |
| 8 | Llama 3 8B (HuggingFace) | 0.261 | 0.111 | 0.758 | 466 |
| — | Llama 3.1 8B (Ollama) | — | — | — | failed L1 (CPU timeout) |

The two metrics disagree at the top — Command-R wins BERTScore F1 with the
shortest summary (semantically aligned), Llama 4 Scout wins ROUGE-L with the
longest (most n-gram overlap). BERT-F1 is uniformly lower than on the
visual path (0.76 vs 0.85) because the audio reference is 2.4× longer than
the median LLM summary; ROUGE recall is length-asymmetric.

### Whisper transcription quality (ASR vs human reference)

Separately, the Whisper output itself is evaluated on the two annotated
audio slices (`asr/eval/slice_{A,B}_groundtruth.txt`, 1,567 reference
words total) using `jiwer`. Run via
`evaluation/asr_evaluate.py`; output `results/asr_evaluation.json`.

| Slice | WER | CER |
|---|---|---|
| Slice A (08:30–09:00) | 5.4 % | 3.7 % |
| Slice B (13:00–13:30) | 7.8 % | 5.7 % |
| **Combined** | **6.6 %** | **4.8 %** |

faster-whisper `small` int8 on CPU is essentially production-grade on this
broadcast — 93–95 % word-level accuracy with no fine-tuning.

### Sample Level-2 output (Llama 3.3 70B)

> *"The day's major stories began with the ongoing tensions between the US and Iran, with President Trump's rhetoric being seen as potentially strengthening Iran's position. A US airman who was missing in Iran after his F-15E fighter jet was shot down was rescued in a daring operation, involving a firefight and the deployment of dozens of aircraft and hundreds of troops. Meanwhile, in the Gaza Strip, Hamas stated that it would not disarm, a key component of the second phase of the Gaza ceasefire plan, citing continued Israeli aggression and the killing of Palestinians. [...]"*

Full output: `llm_summarization/output_asr/latest.json` (merged from
`summaries_*.json` + `summaries_retry_*.json` +
`summaries_retry_chunked_*.json` + `summaries_retry_ollama_*.json` via
`asr/merge_asr_runs.py`).

---

## 8. Phase 7 — Multi-Engine OCR Comparison

Reference: `ocr_comparison/` folder.

A head-to-head benchmark of five OCR engines on the same two ground-truth panoramas used by Phase 5 validation, designed to **justify v6's choice of EasyOCR + Tesseract dash augmentation** as the production pipeline.

### Engines compared

| Engine | Type | Notes |
|---|---|---|
| **Tesseract** | classical | the only engine that natively detects the thin `" - "` delimiter |
| **EasyOCR** | scene-text DL | strong word-level recognition, blind to delimiters |
| **PaddleOCR (PP-OCRv5)** | DL | trained primarily on document text, weaker on TV-ticker fonts |
| **docTR** (db_resnet50 + crnn_vgg16_bn) | DL detector + recognizer | competent on news fonts |
| **CRAFT + TrOCR** (microsoft/trocr-base-printed) | hybrid | CRAFT detector + transformer recognizer |

A Gemini 2.5 Flash Vision adapter was also implemented but **excluded from reported results** — free-tier 20 RPM made wall-clock measurements non-comparable.

All engines were run **CPU-only** on the same two 30-min panoramas. The thesis notes this constraint disproportionately affects PaddleOCR and TrOCR (which benefit most from GPU).

### Two-table comparison architecture

The runner (`run_ocr_engine.py`) saves each engine's raw words and timing. Two distinct evaluations are emitted:

1. **Pure engines** — what each engine produces alone, with no help from other engines. Reveals which can detect the delimiter natively.
2. **Engines + dash augmentation** — Tesseract PSM 6 is run once across the panorama, dash positions are cached, then each engine's words are merged with synthetic `" - "` tokens at those x-positions before re-segmentation. This is the **v6 production architecture** applied uniformly across engines, so the comparison measures recognition quality only.

### Results (combined two slices = 39 GT headlines)

**Pure engines (engine alone):**

| Engine | Mean F1 | Mean CER |
|---|---|---|
| Tesseract | **85.4%** | 10.0% |
| docTR | 63.9% | 20.1% |
| PaddleOCR | 8.5% | 45.0% |
| EasyOCR | 0.0% | — (no delimiters detected) |
| CRAFT+TrOCR | 0.0% | — (no delimiters detected) |

**Engines + Tesseract dash augmentation (v6 production architecture):**

| Engine | Mean F1 | Mean CER | Time | Peak RAM |
|---|---|---|---|---|
| **EasyOCR** | **91.3%** | **8.8%** | 3.9 min | 1.8 GB |
| Tesseract | 85.4% | 10.2% | 0.6 min | 4 MB |
| docTR | 80.3% | 12.0% | 6.0 min | 2.1 GB |
| PaddleOCR | 23.0% | 27.8% | 21.9 min | 596 MB |
| CRAFT+TrOCR (hybrid) | 29.9% | 29.8% | 52.4 min | 2.8 GB |

### Thesis takeaways

1. **EasyOCR + Tesseract dash augmentation is the empirical winner** — 91.3 % F1, 8.8 % CER. The v6 production pipeline's engine choice is validated.
2. **Scene-text recognizers (EasyOCR, PaddleOCR, docTR, CRAFT+TrOCR) skip the thin `" - "` glyph entirely** when run alone. Without dash augmentation EasyOCR scores 0 % F1 — a striking demonstration that recognition quality is necessary but not sufficient when delimiter detection matters.
3. **Tesseract is uniquely valuable as a delimiter detector** even though it isn't the best general-purpose recognizer. The hybrid two-engine architecture exploits each engine's strength.
4. **Vision-Transformer recognizers (TrOCR, PaddleOCR) underperform on TV-ticker fonts** because their training distributions favor document text. Stylised broadcast fonts at low resolution are a documented weakness.

Output: `ocr_comparison/output/comparison_report.{json,txt}` (both tables, machine + human readable).

---

## 9. Phase 8 — VLM-Based Ticker Extraction

Reference: `vlm_extraction/` folder. Detailed standalone report: `vlm_extraction/PROGRESS_REPORT.md`.

Parallel alternative to the OCR pipeline. Instead of recognising glyphs and segmenting on delimiters, send a stitched panorama tile directly to a Vision-Language Model and have it emit structured JSON headlines in one shot.

### Strategy and architecture

- **Strategy A only**: feed v6-stitched panoramas to the VLM, one tile per call. Strategy B (per-frame) was explicitly out of scope.
- **Tile geometry**: 3000 × 87 px tiles, 1000 px overlap (stride 2000). Mirrors v6's slicing geometry. For the full 14 h video this is **1,845 tiles** across 75 chunk panoramas. Tiles are cached idempotently in `vlm_extraction/output/_tiles/`.
- **Output schema parity with OCR**: VLM produces `[{"id": int, "text": str}, ...]` identical to `news_items_cleaned.json`, so `llm_summarization/` consumes it unchanged via the same monkey-patch pattern as `pipeline/summarize_cleaned.py`.
- **Provider abstraction**: `vlm_extraction/adapters/{base,gemini,openai,anthropic,huggingface,groq}_adapter.py`. All adapters share the same prompt from `prompts.py`.

### Provider attempts — what worked, what didn't

| Provider | Status | Why |
|---|---|---|
| **Gemini 2.5 Flash** (paid tier 1) | ✅ Canonical paid run | $0.30/M input + $2.50/M output; thinking disabled via REST API for cost control |
| **Ministral 3 14B** (Mistral La Plateforme) | ✅ Canonical open-weights run | Free Experiment-plan tier (1 B tokens / month, no card), empirical ~40 RPM. Open-weights successor to the retired Pixtral 12B |
| Qwen2-VL 7B (HF Inference) | ❌ blocked | HF Inference Providers requires enabling a paid partner provider for vision models — free `HUGGINGFACE_API_KEY` cannot authorise |
| Groq Llama 4 Scout 17B vision | ⚠️ partial | Free-tier rate cap below 1,845-call requirement; aborted after 152/1,845 tiles |
| GPT-4o-mini (OpenAI) | ⏸ deferred | Key provided but billing not topped up; adapter ready and gated by `OPENAI_BILLING_ACTIVE=true` |
| Claude 3.5 Sonnet (Anthropic) | ⏸ deferred | No key provided; adapter ready |

### Major issues encountered and resolved

1. **Panorama too wide** (142,165 × 87 px → Gemini 400'd) — solved by introducing the tiler.
2. **Tile-edge fragments and mashed pairs** in the original prompt's output — Gemini sometimes concatenated two adjacent ticker headlines when " - " fell on a tile boundary. Required a strict-prompt rewrite (only fully-visible complete headlines, explicit anti-concatenation rule).
3. **Cost-tracking error** — discovered after the cumulative Google bill far exceeded the code's token-derived estimates. Two compounded bugs:
   - Wrong pricing constants in `config.py` ($0.075/M input vs actual $0.30/M; $0.30/M output vs actual $2.50/M) → 4× under-reporting.
   - Gemini 2.5 Flash thinking mode enabled by default → invisible reasoning tokens billed at $3.50/M, not captured in `candidates_token_count`.
   - **Fixed**: corrected pricing in config; rewrote `gemini_adapter.py` to use REST API with `thinkingConfig.thinkingBudget=0`. Cost-per-call is now *estimated* at ≈$0.000481, but this token-derived figure is **not reconciled against the actual invoice** — see "Cumulative VLM spend" below for the real billed total (**€8.28**).
4. **Cleanup over-merging** — an LLM cleanup pass (`pipeline/clean_vlm_headlines.py`) was implemented mirroring `clean_news_items.py`. With aggressive prompts the LLM dropped 72 → 9 items in one experiment. A 50 %-removal safety guard was added; even so, the cleanup hurts F1 on the strict-prompt raw output (drops recall 100 % → 62 %).

### Final canonical run (2026-05-04)

| | |
|---|---|
| Model | Gemini 2.5 Flash, paid tier 1, thinking disabled |
| Tiles | 1,845 (1 errored — recoverable) |
| Raw VLM emissions | 1,695 (~0.9 per tile, vs 6,397 with the old prompt) |
| After cross-tile fuzzy dedup | **100 items** (canonical output) |
| Wall time | 61 min (1.6 s/call, vs 9.4 s with thinking on) |
| Cost (token-derived estimate, **unverified**) | **≈$0.87** for this run — see real-spend note below |

### Results vs OCR (combined slice GT, same 39-headline annotation)

| Pipeline | Items | F1@70 | Mean CER | Recall | Cost | Wall time |
|---|---|---|---|---|---|---|
| OCR raw (v6) | 48 | 84.8 % | 23.4 % | 97.4 % | $0 | 12 min |
| **OCR LLM-cleaned** | **38** | **86.6 %** | **12.4 %** | **89.1 %** | **~$0.05** | **~12 min** |
| **VLM raw (Gemini, strict prompt)** | **100** | **60.5 %** | **18.8 %** | **100 %** | **≈$0.87**† | **~61 min** |
| VLM LLM-cleaned (over-merged) | 44 | 48.4 % | 11.5 % | 61.9 % | ≈$0.88† | ~61 min + cleanup |

†Cost column = code-computed token estimate for the canonical run only. It is **not** reconciled against the Google invoice and under-reports real spend. The **actual amount billed across all VLM experiments was €8.28** (see "Cumulative VLM spend"). Per-run accuracy is unverified.

### Thesis interpretation

Two valid readings of the data:

1. **F1-as-headline-metric reading**: OCR pipeline wins clearly (86.6 % vs 60.5 %). VLM is bottlenecked by per-tile recognition without global panorama context. OCR is also **17× cheaper and 5× faster**.
2. **Granularity-aware reading**: VLM achieves **100 % recall** vs OCR's 89 %. Lower precision is a granularity choice — VLM preserves per-cycle digit drift and minor rewordings as distinct items, OCR's segmentation collapses them. For downstream summarisation (where over-extraction is harmless and under-extraction loses information), VLM's recall advantage may matter more than precision.

The thesis can present this as a **methodology tradeoff finding** rather than a pure quality ranking.

### Cumulative VLM spend

**Actual amount billed by Google across all VLM experiments: €8.28** (the
user's real billing statement — this is the authoritative figure for the
thesis). This total covers the first full-video run with the old prompt
and thinking enabled, three cleanup-pass iterations, the strict-prompt
re-run, and the final cleanup attempt.

The per-call USD figures elsewhere in this document and in
`results/vlm_evaluation.{json,md}` are **code-computed token-cost
estimates only**. They were never reconciled against the invoice and
collectively under-report the real spend; the per-run accuracy (≈$0.87
for the canonical strict-prompt run) is **unverified**. Where a thesis
table needs a single cost number for the VLM approach, cite the real
billed total (**€8.28**), not the token estimate.

Output: `vlm_extraction/output/gemini/full_video/run_1/headlines_combined.json` (canonical 100 items).
Eval: `results/vlm_evaluation.json` and `results/vlm_evaluation.md`.

### Phase 8.5 — Open-weights VLM extension (Ministral 3 14B, free)

Added in 2026-05-12 to answer "does a free open-weights VLM compete?".
Entry point: `vlm_extraction/opensource_vlm.py`. The Mistral
La Plateforme free tier (1 B tokens / mo) serves Ministral 3 14B —
open-weights, vision-capable, and the official successor Mistral docs
point to after Pixtral 12B's 2025-12-31 retirement.

### 3-way head-to-head (same prompt, same tiles, 39-headline GT)

| Pipeline | F1@70 | Recall | Precision | Hallucination | Cost | Wall |
|---|---|---|---|---|---|---|
| OCR LLM-cleaned (v6 + Gemini cleaner) | **88.1 %** | 92.3 % | 84.2 % | n/a | ≈$0.05† | ~12 min |
| Gemini 2.5 Flash (paid) | 64.9 % | 100 % | 48.0 % | 38.0 % | ≈$0.87† | 61 min |
| **Ministral 3 14B (open-weights, free)** | **71.0 %** | 100 % | 55.3 % | 35.0 % | **$0.00** | 114 min |

†Code-computed token estimate for that single run. **Real money billed
by Google across all VLM experiments was €8.28** (see "Cumulative VLM
spend"); the per-run estimates under-report actual spend and their
accuracy is unverified.

The open-weights free VLM **beats paid Gemini on F1 with the same prompt
and tiles, at zero cost**; both VLMs still trail the OCR pipeline by
~17 pp on F1 because they emit many more candidate headlines (lower
precision, but better recall). Mistral emits 3,214 raw items across the
14 h broadcast versus Gemini's 100; the cross-tile fuzzy aggregator keeps
both within the same recall envelope.

Output: `vlm_extraction/output/mistral/full_video/run_1/headlines_combined.json`.
Report: `results/vlm_opensource_evaluation.{json,md}`.

---

## 10. Phase 9 — Web Surfaces (Defense Site + Interactive Workbench)

Reference: `site/`, `webapp/`, `design-system/`. Two parallel web deliverables sharing one design system.

### `site/` — Defense site (read-only, static)

Astro 4 static export with React islands for animation. **Six routes**:

- **Landing** — project overview, three-card path navigation (OCR / VLM / Audio), key-numbers strip including the open-weights VLM result.
- **`/visual`** — 8 animated stage panels walking through the OCR pipeline (frame extraction → scroll → panorama → OCR → segmentation → improved segmentation → LLM cleaning → 9-LLM summarization). Real EasyOCR bboxes and per-frame scroll deltas rendered from cached telemetry collected by `instrument_slice_A.py`. Stages with live data show a small "live data" badge; representative numbers are clearly labelled where used.
- **`/vlm`** — 3-way comparison (OCR / Gemini / Ministral) with method callout, ranked card row, per-run metrics table, and side-by-side first-20-headline samples from each VLM. Driven by `VLMComparison.jsx` + `VLMHeadlineSamples.jsx`.
- **`/audio`** — 5 animated stage panels for the ASR pipeline, plus the audio summarisation leaderboard (8 LLMs vs the audio reference) and the Whisper-quality WER/CER callout.
- **`/results`** — P/R/F1/CER tables, **parallel ROUGE/BERTScore leaderboards** for visual + audio summarisation (same EvalStage component, different accents), 3-way VLM comparison repeated for context, and the GT-vs-extracted overlay.
- **`/explore`** — tabbed data browser: headlines / transcript / summaries / raw OCR output, all searchable.

### `site/scripts/` — asset pipeline

- **`build_assets.py`** — idempotent. Reads source artefacts (eval JSONs, panoramas, transcript, summaries) from the project root and bundles ~2.8 MB into `site/public/data/` and `site/public/video/`. The bundled assets are committed so a fresh clone can `cd site && npm install && npm run dev` without the source video.
- **`instrument_slice_A.py`** — companion script that imports v6's modules read-only and persists artefacts v6 doesn't normally save (per-frame scroll deltas, real EasyOCR word-level bboxes on the first 3000-px panorama segment, real frame counts). The site's animated stages render these real numbers.

### Demo clip — `videos/extract_demo_clip.py`

Uses PyAV to extract `videos/demo_clip_30s.mp4` (480p H.264, ~1.3 MB) and `demo_clip_30s.wav` (16 kHz mono PCM, ~940 KB) from 08:30:00 → 08:30:30 of the source video. Aligns with Slice A so the OCR walkthrough matches the annotated panorama. Idempotent — skips if outputs exist. Both files are gitignored.

### `webapp/frontend/` — Interactive workbench (live, with backend)

Existing React + Vite UI redesigned to match the defense site's visual identity. Same FastAPI backend (`webapp/backend/`) — UI rewrite only, no backend changes. Adds Framer Motion animations, Lucide icons, dark mode, animated status pills, drag-drop upload zone. All API endpoints unchanged.

### `design-system/` — Shared visual identity (no build step)

- `tokens.css` — light + dark palettes, path-accent colors per pipeline (visual / audio / VLM), motion tokens, spacing, radii, shadows
- `components/visual/` — 8 React stage components reused by both `site/` and `webapp/frontend/`
- `components/audio/` — 4 React stage components, same pattern
- `components/shared/` — `StagePanel`, `ResultsHero`, `GTOverlay`, `Explorer`

Both projects' Tailwind configs glob the design-system folder so utility classes are extracted from the shared components. No copy-paste — single source of truth.

---

## 11. End-to-End Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                 INPUT: AlJazeera_14hrs.mp4 (14.55 h)            │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────────┐ ┌──────▼──────────────┐ ┌───▼──────────┐
│ VISUAL — OCR       │ │ VISUAL — VLM        │ │ AUDIO        │
│ (ticker_extraction │ │ (vlm_extraction/)   │ │ (asr/)       │
│  _v6 + pipeline)   │ │                     │ │              │
└───────┬────────────┘ └──────┬──────────────┘ └───┬──────────┘
        │                     │                    │
┌───────▼────────────┐ ┌──────▼──────────────┐ ┌───▼──────────┐
│ Step 1: frames     │ │ tiler.py:           │ │ extract_audio│
│ Step 2: scroll     │ │ panoramas → 3000x87 │ │ (PyAV→WAV)   │
│ Step 3: panorama   │ │ tiles, 1000 overlap │ │ → 1.68 GB    │
│ → 75 panoramas     │ │ → 1845 tiles        │ │   audio_full │
└───────┬────────────┘ └──────┬──────────────┘ └───┬──────────┘
        │                     │                    │
┌───────▼────────────┐ ┌──────▼──────────────┐ ┌───▼──────────┐
│ Step 4: dual-engine│ │ extract.py:         │ │ transcribe.py│
│ EasyOCR + Tesseract│ │ Gemini 2.5 Flash    │ │ faster-      │
│ (delimiter detect) │ │ (REST, thinking off)│ │ whisper      │
└───────┬────────────┘ │ → per-tile JSON     │ │ small int8   │
        │              └──────┬──────────────┘ │ 20-min chunks│
┌───────▼────────────┐        │                │ → 13,517 segs│
│ Step 5: segment    │ ┌──────▼──────────────┐ └───┬──────────┘
│ → 48 raw items     │ │ aggregate.py:       │     │
└───────┬────────────┘ │ rapidfuzz dedup     │ ┌───▼──────────┐
        │              │ → 100 items         │ │ chunk_       │
┌───────▼────────────┐ └──────┬──────────────┘ │ transcript   │
│ improve_segment    │        │                │ (15-min)     │
│ ation.py           │ ┌──────▼──────────────┐ │ → 59 chunks  │
│ → 48 items         │ │ headlines_combined  │ └───┬──────────┘
└───────┬────────────┘ │ .json               │     │
        │              │ (canonical VLM out) │ ┌───▼──────────┐
┌───────▼────────────┐ └──────────────┬──────┘ │ summarize_   │
│ clean_news_items.py│                │        │ transcript   │
│ (Gemini 2.5)       │                │        │ L1: 59x9 LLMs│
│ → 38 items         │                │        │ L2: 9 final  │
└───────┬────────────┘                │        └───┬──────────┘
        │                             │            │
┌───────▼────────────┐                │        ┌───▼──────────┐
│ summarize_cleaned  │                │        │ output_asr/  │
│ (9 LLMs)           │                │        │ latest.json  │
│ → output_cleaned/  │                │        └───┬──────────┘
└───────┬────────────┘                │            │
        │                             │            │
        └────────────┬────────────────┴────────────┘
                     │
        ┌────────────▼────────────────┐
        │ EVALUATION                  │
        │ validation/validate_extr.py │
        │ vlm_extraction/evaluate.py  │
        │ ocr_comparison/evaluate_all │
        │ → results/                  │
        │   • validation_report.json  │
        │   • vlm_evaluation.json/md  │
        │   • ocr_comparison_report   │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │ WEB SURFACES                │
        │ site/  (defense, static)    │
        │ webapp/  (interactive)      │
        │ design-system/  (shared)    │
        └─────────────────────────────┘
```

---

## 12. Project Structure

```
ticker_extraction/
├── ticker_extraction_v6/        ← FROZEN — 5-step OCR pipeline (DO NOT MODIFY)
│   ├── main.py, config.py, step1..step5_*.py
│   └── output/
│       ├── chunks/              (per-chunk outputs)
│       ├── panorama/            (75 chunk panorama PNGs, ~407 MB)
│       └── final/
│           ├── news_items.json             (improved raw v6 output, 48 items)
│           ├── news_items_cleaned.json     (LLM-cleaned, 38 items)
│           └── pipeline_stats.json
│
├── llm_summarization/           ← FROZEN — 9-LLM summarization + evaluation
│   ├── config.py, prompt.py, summarize.py, evaluate.py
│   ├── reference_summary.txt
│   ├── output/                  (27-min sample summaries)
│   ├── output_cleaned/          (LLM-cleaned 14h ticker summaries)
│   └── output_asr/              (ASR transcript summaries)
│
├── videos/                      ← raw input .mp4 files + extract_demo_clip.py
│
├── pipeline/                    ← Phase 3 + Phase 8 post-processing
│   ├── improve_segmentation.py  (re-segments raw OCR, better dedup)
│   ├── clean_news_items.py      (LLM correction for OCR output)
│   ├── clean_vlm_headlines.py   (LLM correction for VLM output)
│   ├── summarize_cleaned.py     (runs 9-LLM summ on cleaned items)
│   └── evaluate_cleaned_summaries.py  (scores output_cleaned/ vs reference,
│                                       canonical 14h visual eval)
│
├── validation/                  ← Phase 5 ground-truth validation
│   ├── regenerate_panorama.py   (rebuilds panorama PNG for a slice)
│   ├── validate_extraction.py   (computes P/R/F1, CER, WER)
│   └── ground_truth/
│       ├── slice_A_panorama.png (15 MB, hours 8:30–9:00)
│       ├── slice_A_headlines.txt (27 manually-annotated headlines)
│       ├── slice_B_panorama.png (13.6 MB, hours 13:00–13:30)
│       └── slice_B_headlines.txt (28 manually-annotated headlines)
│
├── ocr_comparison/              ← Phase 7 multi-engine OCR comparison
│   ├── config.py, run_ocr_engine.py
│   ├── augment_dashes.py, regenerate_pure.py, evaluate_all.py
│   ├── engines/  (Tesseract, EasyOCR, Paddle, docTR, Gemini, CRAFT+TrOCR)
│   └── output/
│       ├── slice_A/, slice_B/   (per-engine raw text, words, headlines, timing)
│       ├── comparison_report.json (both pure + augmented tables)
│       └── comparison_report.txt
│
├── vlm_extraction/              ← Phase 8 VLM extraction
│   ├── README.md, PROGRESS_REPORT.md
│   ├── config.py, prompts.py, tiler.py
│   ├── extract.py, aggregate.py, evaluate.py
│   ├── opensource_vlm.py        (open-source VLM runner + bespoke evaluator
│   │                             for the 3-way OCR vs Gemini vs Ministral
│   │                             comparison)
│   ├── adapters/                (base + gemini, openai, anthropic, hf, groq,
│   │                             mistral)
│   ├── output/
│   │   ├── _tiles/              (1845 cached 3000×87 tile PNGs)
│   │   ├── gemini/full_video/run_1/      (paid canonical run, 100 items)
│   │   ├── mistral/full_video/run_1/     (open-weights canonical run, 3214
│   │   │                                   raw items, 100% recall, $0.00)
│   │   └── groq_scout/                    (partial / abandoned)
│   ├── .env (gitignored), .env.example, .gitignore
│
├── asr/                         ← Phase 6 audio pipeline
│   ├── extract_audio.py, transcribe.py
│   ├── chunk_transcript.py, summarize_transcript.py
│   ├── retry_level2.py                   (single-call retry with backoff)
│   ├── retry_level2_chunked.py           (3-batch L2 for TPM-capped models)
│   ├── run_ollama_l1_l2.py               (resume-safe Ollama L1+L2,
│   │                                       1200s timeout + warmup,
│   │                                       supports --l2-only)
│   ├── merge_asr_runs.py                 (merges original + all retries +
│   │                                       re-runs evaluation)
│   ├── evaluate_audio_summaries.py       (wrapper: scores latest.json vs
│   │                                       reference_summary_audio.txt)
│   ├── extract_eval_slices.py, transcribe_eval_slices.py
│   │                                      (Whisper-quality ground truth flow)
│   ├── eval/                              (manually-annotated audio slices
│   │                                       for WER/CER scoring)
│   └── output/  (audio_full.wav, transcript_full.{json,txt,srt}, chunks/,
│                 level1_ollama_inflight.json)
│
├── evaluation/                  ← variance / determinism experiment
│   ├── asr_evaluate.py                    (Whisper-quality scorer)
│   ├── run_variance_summarization.py     (3-run repeat)
│   └── aggregate_variance.py
│
├── design-system/               ← Phase 9 shared visual identity (NEW)
│   ├── tokens.css
│   └── components/  (shared, visual, audio React components)
│
├── site/                        ← Phase 9 defense site, Astro 4 (NEW)
│   ├── astro.config.mjs, package.json
│   ├── src/{layouts,components,pages,styles}
│   ├── public/{data,video}      (BUNDLED, ~2.8 MB)
│   └── scripts/  (build_assets.py, instrument_slice_A.py)
│
├── webapp/                      ← Phase 9 interactive workbench (NEW)
│   ├── backend/                 (FastAPI: upload, jobs, pipeline orchestration)
│   ├── frontend/                (React + Vite, redesigned UI)
│   ├── jobs/, uploads/          (runtime state, gitignored)
│   └── README.md
│
├── results/                     ← all evaluation outputs (canonical thesis tables)
│   ├── validation_report.json   (Phase 5 OCR metrics, 88.1% F1 cleaned)
│   ├── asr_evaluation.json      (Phase 6 Whisper-quality WER/CER, 6.6%/4.8%)
│   ├── asr_summary_evaluation.json     (Phase 6 audio summarisation, 8 LLMs
│   │                                     vs reference_summary_audio.txt)
│   ├── visual_14h_summary_evaluation.json   (Phase 4 visual summarisation,
│   │                                          8 LLMs vs reference_summary.txt
│   │                                          on the 14h cleaned headlines)
│   ├── vlm_evaluation.json + .md       (Phase 8 paid Gemini run)
│   ├── vlm_opensource_evaluation.json + .md  (Phase 8.5 3-way comparison
│   │                                          OCR vs Gemini vs Ministral)
│   ├── ocr_comparison_report.json + .txt    (Phase 7 5-engine comparison)
│   ├── variance_report.json + .md           (3-run determinism experiment)
│
├── logs/                        (vlm_<provider>.log per provider)
│
├── CLAUDE.md                    ← project instructions for Claude
├── TODO.md
├── README.md
├── PROGRESS_REPORT.md           ← this file
├── requirements.txt
└── .gitignore
```

---

## 13. Technical Glossary

| Term | Meaning |
|---|---|
| **OCR** | Optical Character Recognition — reading text from images |
| **VLM** | Vision-Language Model — large language model whose input layer accepts images alongside text (e.g. Gemini 2.5 Flash, GPT-4o-mini, Claude 3.5 Sonnet) |
| **ASR** | Automatic Speech Recognition — converting spoken audio to text |
| **Whisper** | Open-source speech recognition model by OpenAI |
| **faster-whisper** | CTranslate2 port of Whisper that runs 4× faster on CPU with int8 quantization |
| **Panorama stitching** | Combining many overlapping images into one long image |
| **Tile / tiling** | In the VLM pipeline: cutting a wide panorama into VLM-digestible chunks (3000×87 px with 1000 px overlap) so the model can process it without exceeding image-size limits |
| **rapidfuzz** | Python library for fast fuzzy string matching (used for deduplication) |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation — n-gram overlap metric for summary evaluation |
| **BERTScore** | Semantic similarity metric using BERT embeddings, correlates with human judgment |
| **CER** | Character Error Rate — edit distance / reference length at character level |
| **WER** | Word Error Rate — same but at word level, standard ASR/OCR metric |
| **Precision** | Of extracted items, what fraction match a real headline |
| **Recall** | Of real headlines, what fraction were extracted |
| **F1** | Harmonic mean of precision and recall |
| **Fuzzy match threshold** | Minimum similarity (0–100) for two strings to be considered the same |
| **Chunked processing** | Splitting long inputs into manageable pieces to avoid memory/context-window issues |
| **Monkey-patching** | Modifying a module's attributes at runtime without changing its source code (used to reuse `llm_summarization/` without violating the "do not modify" rule) |
| **Thinking tokens** | (Gemini 2.5+ specific) Invisible reasoning tokens generated by "thinking" mode and billed separately at a higher rate ($3.50/M for Flash). Disabled in the VLM adapter via `thinkingConfig.thinkingBudget=0` |
| **TPM / RPM / RPD** | Tokens / Requests Per Minute / Day — rate limits measured by most LLM providers |
| **PSM** | Page Segmentation Mode — Tesseract's layout hint (6 = uniform block, 7 = single line) |
| **Strategy A / B** | (VLM extraction terminology) Strategy A = panorama-tile input, one VLM call per tile. Strategy B = per-frame input, multiple calls per ticker cycle. This thesis uses Strategy A only |
| **Hallucination rate** | (VLM evaluation) Fraction of model-emitted headlines that match no GT headline at the loosest 60 % fuzzy threshold — i.e. headlines the model invented or garbled beyond recognition |
| **Granularity** | The level at which an extraction pipeline keeps stories distinct. OCR's segmentation merges per-cycle digit drift into one item; VLM keeps each cycle separate. Both are valid; they trade precision for recall |
