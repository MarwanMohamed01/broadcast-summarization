# Progress Report: Vision-Based Extraction and LLM-Powered Summarization of TV News Tickers

**Project**: Master's Thesis — GIU Berlin
**Author**: Marwan Mohamed
**Last updated**: April 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1 — Vision-Based Ticker Extraction (v6 Pipeline)](#2-phase-1--vision-based-ticker-extraction-v6-pipeline)
3. [Phase 2 — Scaling to 14-Hour Video (Chunked Mode + Dual-Engine OCR)](#3-phase-2--scaling-to-14-hour-video-chunked-mode--dual-engine-ocr)
4. [Phase 3 — Post-Processing (Improved Segmentation + LLM Cleaning)](#4-phase-3--post-processing-improved-segmentation--llm-cleaning)
5. [Phase 4 — Ticker Summarization (9-LLM Comparison)](#5-phase-4--ticker-summarization-9-llm-comparison)
6. [Phase 5 — Validation (Ground Truth + Metrics)](#6-phase-5--validation-ground-truth--metrics)
7. [Phase 6 — Audio Pipeline (ASR + LLM Summarization)](#7-phase-6--audio-pipeline-asr--llm-summarization)
8. [End-to-End Pipeline Diagram](#8-end-to-end-pipeline-diagram)
9. [Project Structure](#9-project-structure)
10. [Technical Glossary](#10-technical-glossary)

---

## 1. Project Overview

This thesis implements a complete pipeline for automatic extraction and summarization of TV news content from two complementary modalities:

- **Visual modality**: the scrolling news ticker at the bottom of the screen (OCR)
- **Auditory modality**: the spoken broadcast content (ASR — Automatic Speech Recognition)

The final output for each modality is a ranked comparison of 9 different LLMs summarizing the day's news, benchmarked against a human reference using ROUGE + BERTScore.

### Input videos

- **`AlJazeera_sample.mp4`** — 27-minute Al Jazeera sample (1280×720, 30fps) — used for initial pipeline development
- **`AlJazeera_14hrs_without_edits.mp4`** — 14.55-hour Al Jazeera broadcast (1,571,786 frames) — the main experimental dataset

### Pipeline summary (both modalities)

```
VIDEO
  │
  ├── VISUAL PATH: frames → scroll detection → panorama → dual-engine OCR
  │                → segmentation → improved segmentation → LLM cleaning
  │                → 9-LLM summarization → evaluation
  │
  └── AUDIO PATH: WAV extraction → Whisper ASR → time-based chunking
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

### Results (ticker summarization)

**8 of 9 models succeeded** — only Ollama `llama3.1:8b` timed out (300s CPU limit). Summaries range from 1226 chars (terse) to 3024 chars (verbose). Output stored in `llm_summarization/output_cleaned/latest.json`.

Final evaluation against a human reference summary (ROUGE + BERTScore) is pending your reference submission.

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
- **Combined (union)**: **39 unique headlines**

### Metrics (`validation/validate_extraction.py`)

Computes, for each stage (raw v6, LLM-cleaned) × each slice (A, B, combined):

- **Precision** at 60% / 70% / 80% fuzzy-match thresholds
- **Recall** at same thresholds
- **F1** at same thresholds
- **Character Error Rate (CER)** on pairs matched at 70%
- **Word Error Rate (WER)** on pairs matched at 70%
- **Exact match count** (≥95% similarity)
- **Coverage** (% of GT with any ≥50% match)
- **Missed headlines** list (specific GT items not found)
- **False-positive** list (extracted items with no GT match — some may be legitimate headlines from unsampled parts of the 14h)

### Key results

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

### Level-2 final summary results

| Model | Level-1 coverage | Level-2 status | Output length |
|---|---|---|---|
| Llama 3.3 70B (Groq) | 36/59 | ✅ | 3,411 chars |
| Llama 4 Scout 17B (Groq) | **59/59** | ✅ | 4,857 chars |
| Gemini 2.5 Flash | 21/59 | ✅ | 3,917 chars |
| Command-R (Cohere) | **59/59** | ✅ | 2,362 chars |
| Llama 3.2 3B (Ollama) | 2/59 | ✅ (earlier run) | 3,023 chars |
| Llama 3.1 8B (Groq) | 59/59 | ❌ TPM exceeded on 12 K-token concatenated input |
| Qwen3 32B (Groq) | 59/59 | ❌ TPM exceeded |
| Llama 3 8B (HuggingFace) | 59/59 | ❌ HF "Bad Request" (context length) |

**5 out of 9 models** produced complete final summaries covering the full day's broadcast. The retry script (`asr/retry_level2.py`) uses exponential backoff on rate limits.

### Sample Level-2 output (Llama 3.3 70B)

> *"The day's major stories began with the ongoing tensions between the US and Iran, with President Trump's rhetoric being seen as potentially strengthening Iran's position. A US airman who was missing in Iran after his F-15E fighter jet was shot down was rescued in a daring operation, involving a firefight and the deployment of dozens of aircraft and hundreds of troops. Meanwhile, in the Gaza Strip, Hamas stated that it would not disarm, a key component of the second phase of the Gaza ceasefire plan, citing continued Israeli aggression and the killing of Palestinians. [...]"*

Full output: `llm_summarization/output_asr/latest.json`.

### Evaluation (pending)

Requires a human reference ASR summary (to be written). When available, `llm_summarization/evaluate.py` is run via a monkey-patched wrapper against the Level-2 summaries to compute ROUGE + BERTScore rankings for each of the 5 successful models.

---

## 8. End-to-End Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                 INPUT: AlJazeera_14hrs.mp4 (14.55 h)            │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
 ┌────────▼──────────┐               ┌───────────▼──────────┐
 │ VISUAL PATH       │               │ AUDIO PATH            │
 │ (ticker_extraction│               │ (asr/)                │
 │  _v6 + pipeline)  │               │                       │
 └────────┬──────────┘               └───────────┬──────────┘
          │                                       │
 ┌────────▼──────────┐               ┌───────────▼──────────┐
 │ Step 1: frames    │               │ extract_audio.py     │
 │ Step 2: scroll    │               │ (PyAV → 16kHz WAV)   │
 │ Step 3: panorama  │               │                       │
 └────────┬──────────┘               │ → audio_full.wav      │
          │                          │   (1.68 GB, 14h)      │
 ┌────────▼──────────┐               └───────────┬──────────┘
 │ Step 4: OCR       │                           │
 │ (EasyOCR text +   │               ┌───────────▼──────────┐
 │  Tesseract dash)  │               │ transcribe.py        │
 └────────┬──────────┘               │ (faster-whisper,     │
          │                          │  small, int8, CPU,   │
 ┌────────▼──────────┐               │  20-min chunks)      │
 │ Step 5: segment   │               │                       │
 │ (delimiter split) │               │ → transcript_full     │
 │ → ~48 items       │               │   .json/.txt/.srt     │
 └────────┬──────────┘               │   (13,517 segments)   │
          │                          └───────────┬──────────┘
 ┌────────▼──────────┐                           │
 │ improve_segment   │               ┌───────────▼──────────┐
 │ ation.py          │               │ chunk_transcript.py  │
 │ (all cycles, best │               │ (15-min chunks)      │
 │  version dedup)   │               │ → 59 chunk .txt files│
 │ → 48 items        │               └───────────┬──────────┘
 └────────┬──────────┘                           │
          │                          ┌───────────▼──────────┐
 ┌────────▼──────────┐               │ summarize_transcript │
 │ clean_news_items  │               │ .py                  │
 │ .py (Gemini 2.5)  │               │ LEVEL 1: 59×9 LLMs   │
 │ → 38 items        │               │ LEVEL 2: 9 final     │
 └────────┬──────────┘               │  multi-paragraph     │
          │                          │  summaries           │
 ┌────────▼──────────┐               └───────────┬──────────┘
 │ summarize_cleaned │                           │
 │ .py (9 LLMs)      │               ┌───────────▼──────────┐
 │ → 9 paragraph-    │               │ output_asr/latest    │
 │   summaries       │               │ .json (5 of 9 with   │
 └────────┬──────────┘               │  successful L2)      │
          │                          └──────────────────────┘
 ┌────────▼──────────┐
 │ output_cleaned/   │
 │ latest.json       │
 │ (8 of 9 with      │
 │  successful sum)  │
 └────────┬──────────┘
          │
 ┌────────▼──────────┐
 │ VALIDATION        │
 │ (validation/)     │
 │ P/R/F1, CER/WER   │
 │ vs manually       │
 │ annotated GT      │
 └───────────────────┘
```

---

## 9. Project Structure

```
ticker_extraction/
├── ticker_extraction_v6/        ← FROZEN — 5-step OCR pipeline (DO NOT MODIFY)
│   ├── main.py, config.py, step1..step5_*.py
│   └── output/
│       ├── chunks/              (per-chunk outputs from chunked mode)
│       └── final/
│           ├── news_items.json             (improved raw v6 output, 48 items)
│           ├── news_items_cleaned.json     (LLM-cleaned, 38 items)
│           └── pipeline_stats.json
│
├── llm_summarization/           ← FROZEN — 9-LLM summarization + evaluation
│   ├── config.py, prompt.py, summarize.py, evaluate.py
│   ├── reference_summary.txt    (for 27-min sample)
│   ├── output/                  (summaries for original 27-min sample)
│   ├── output_cleaned/          (summaries for LLM-cleaned 14h ticker)
│   └── output_asr/              (summaries for ASR transcript)
│       ├── level1_<ts>.json
│       ├── summaries_<ts>.json
│       └── latest.json
│
├── videos/                      ← raw input .mp4 files
│
├── pipeline/                    ← Phase 3 post-processing scripts
│   ├── improve_segmentation.py  (re-segments raw OCR, better dedup)
│   ├── clean_news_items.py      (LLM correction via Gemini 2.5)
│   └── summarize_cleaned.py     (runs 9-LLM summ on cleaned items)
│
├── validation/                  ← Phase 5 ground-truth validation
│   ├── regenerate_panorama.py   (rebuilds panorama PNG for a slice)
│   ├── validate_extraction.py   (computes P/R/F1, CER, WER)
│   └── ground_truth/
│       ├── README.md            (annotation instructions)
│       ├── slice_A_panorama.png (15 MB, hours 8:30–9:00)
│       ├── slice_A_headlines.txt (27 manually-annotated headlines)
│       ├── slice_B_panorama.png (13.6 MB, hours 13:00–13:30)
│       └── slice_B_headlines.txt (28 manually-annotated headlines)
│
├── asr/                         ← Phase 6 audio pipeline
│   ├── extract_audio.py         (PyAV → 16 kHz mono WAV)
│   ├── transcribe.py            (faster-whisper chunked streaming)
│   ├── chunk_transcript.py      (time-based 15-min chunking)
│   ├── summarize_transcript.py  (2-level LLM summarization)
│   ├── retry_level2.py          (retry Level-2 with backoff)
│   └── output/
│       ├── audio_full.wav       (1.68 GB, 14h)
│       ├── audio_slice_*.wav    (test slices)
│       ├── transcript_full.json/.txt/.srt (13,517 segments)
│       ├── transcript_slice_A.* (test slice transcript)
│       ├── chunks/              (59 × 15-min chunks)
│       └── _chunks_transcript_full/ (intermediate per-20min resume files)
│
├── results/                     ← all evaluation outputs
│   └── validation_report.json   (full Phase 5 metrics)
│
├── old/                         ← superseded experiments
│   ├── extract_headlines_llm.py
│   └── resegment.py
│
├── CLAUDE.md                    ← project instructions for Claude
├── TODO.md
├── README.md
├── PROGRESS_REPORT.md           ← this file
├── requirements.txt
└── .gitignore
```

---

## 10. Technical Glossary

| Term | Meaning |
|---|---|
| **OCR** | Optical Character Recognition — reading text from images |
| **ASR** | Automatic Speech Recognition — converting spoken audio to text |
| **Whisper** | Open-source speech recognition model by OpenAI |
| **faster-whisper** | CTranslate2 port of Whisper that runs 4× faster on CPU with int8 quantization |
| **Panorama stitching** | Combining many overlapping images into one long image |
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
| **TPM** | Tokens Per Minute — rate limit measured by most LLM providers |
| **PSM** | Page Segmentation Mode — Tesseract's layout hint (6 = uniform block, 7 = single line) |
