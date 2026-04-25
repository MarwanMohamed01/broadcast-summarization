# Multimodal Vision- and Audio-Based Extraction and LLM-Powered Summarization of TV News

Master's Thesis — German International University (GIU), Berlin

---

## Overview

An end-to-end pipeline that automatically extracts and summarizes news content from TV broadcast videos through two independent modalities:

| Path | Input | Method | Output |
|---|---|---|---|
| **Visual** | Scrolling ticker bar | OCR (EasyOCR + Tesseract) | Extracted headlines + 9-LLM summaries |
| **Audio** | Spoken broadcast | ASR (Whisper) | Full transcript + 9-LLM multi-paragraph summaries |

Both paths are compared against human reference summaries using ROUGE and BERTScore.

The repository ships **two web surfaces** that share a single design system:

- **`site/`** — a polished, read-only **defense site** (Astro static export) that animates every pipeline stage and exposes the precomputed results for committee review.
- **`webapp/frontend/`** — an **interactive workbench** (React + FastAPI) where a user uploads a video, picks a time range, and runs the live OCR / ASR / LLM pipeline.

A separate **OCR engine comparison** (`ocr_comparison/`) benchmarks five OCR engines (Tesseract, EasyOCR, PaddleOCR, docTR, CRAFT+TrOCR) on the same two ground-truth panoramas, in both pure and Tesseract-dash-augmented configurations.

### Key Results

| Metric | Raw Vision Pipeline | + LLM Cleaning |
|---|---|---|
| **Recall @70%** | 97.4% | 92.3% |
| **Precision @70%** | 75.0% | 84.2% |
| **F1 @70%** | 84.8% | **88.1%** |
| **CER** | 23.4% | **13.9%** |
| **WER** | 35.9% | **19.6%** |

Validated against 39 manually-annotated ground-truth headlines from two 30-minute slices of a 14-hour Al Jazeera broadcast.

### OCR Engine Comparison (head-to-head, CPU-only)

Five engines on the **same two ground-truth panoramas**, segmentation pipeline held constant, two configurations reported.

**Pure engines (no Tesseract dash augmentation)**

| Engine | Mean F1 | Notes |
|---|---|---|
| Tesseract | **85.4%** | Only engine to natively detect the `" - "` delimiter |
| docTR | 63.9% | Detects some dashes |
| PaddleOCR | 8.5% | Noisy recognition, few dashes |
| EasyOCR | 0.0% | Skips `" - "` glyph entirely → segmentation collapses |
| CRAFT + TrOCR | 0.0% | Same dash problem |

**Engines + Tesseract dash augmentation (matches v6 production architecture)**

| Engine | Mean F1 | CER | Time | Peak RAM |
|---|---|---|---|---|
| **EasyOCR** | **91.3%** | **8.8%** | 3.9 min | 1.8 GB |
| Tesseract | 85.4% | 10.2% | 0.6 min | 4 MB |
| docTR | 80.3% | 12.0% | 6.0 min | 2.1 GB |
| PaddleOCR | 23.0% | 27.8% | 21.9 min | 596 MB |
| CRAFT + TrOCR (hybrid) | 29.9% | 29.8% | 52.4 min | 2.8 GB |

EasyOCR's win in the augmented configuration validates v6's choice of EasyOCR as the primary recognizer.

---

## Architecture

```
                        INPUT: TV News Video (.mp4)
                                  |
                 +----------------+----------------+
                 |                                 |
          VISUAL PATH                        AUDIO PATH
                 |                                 |
    Frame Extraction (OpenCV)           Audio Extraction (PyAV)
                 |                                 |
    Scroll Detection (Template Match)   Whisper ASR (faster-whisper)
                 |                           small model, int8, CPU
    Panorama Stitching                         |
                 |                      Time-based Chunking
    Dual-Engine OCR                      (59 x 15-min chunks)
    (EasyOCR text + Tesseract dashes)          |
                 |                      2-Level LLM Summarization
    Headline Segmentation               L1: per-chunk paragraph
    + Deduplication                     L2: final multi-paragraph
                 |                             |
    LLM Cleaning (Gemini 2.5 Flash)            |
                 |                             |
    9-LLM Summarization  <----same 9 models---->
                 |                             |
    ROUGE + BERTScore Evaluation               |
```

## LLMs Used

| Provider | Model | Type |
|---|---|---|
| Groq | Llama 3.3 70B | Cloud (free tier) |
| Groq | Llama 3.1 8B | Cloud (free tier) |
| Groq | Qwen3 32B | Cloud (free tier) |
| Groq | Llama 4 Scout 17B | Cloud (free tier) |
| Ollama | Llama 3.2 3B | Local |
| Ollama | Llama 3.1 8B | Local |
| Google | Gemini 2.5 Flash | Cloud (free tier) |
| HuggingFace | Llama 3 8B Instruct | Cloud (free tier) |
| Cohere | Command-R | Cloud (free trial) |

---

## Quick Start

### Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on PATH
- (Optional) [Ollama](https://ollama.com/) for local LLM inference
- Node.js 18+ (for the web app frontend)

### Installation

```bash
git clone https://github.com/MarwanMohamed01/broadcast-summarization.git
cd broadcast-summarization

pip install -r requirements.txt
```

### API Keys

Copy the example env file and fill in your keys (all providers offer free tiers):

```bash
cp .env.example llm_summarization/.env
# Edit llm_summarization/.env with your API keys
```

### Run the OCR Pipeline (CLI)

```bash
# Short video (no chunking needed)
cd ticker_extraction_v6
python main.py --video ../videos/your_video.mp4

# Long video (14+ hours) — process in 30-minute chunks
python main.py --video ../videos/your_video.mp4 --chunk-minutes 30
```

### Run Post-Processing + Summarization

```bash
# Improve segmentation (better dedup, all cycles)
python pipeline/improve_segmentation.py

# LLM cleaning (fixes OCR typos, splits merged headlines)
python pipeline/clean_news_items.py

# Run 9-LLM summarization on cleaned headlines
python pipeline/summarize_cleaned.py
```

### Run the ASR Pipeline (CLI)

```bash
# Extract audio
python asr/extract_audio.py

# Transcribe with Whisper (small model, ~3x realtime on CPU)
python asr/transcribe.py

# Chunk into 15-minute segments
python asr/chunk_transcript.py

# 2-level LLM summarization (59 chunks x 9 models)
python asr/summarize_transcript.py
```

### Run the Interactive Workbench (`webapp/`)

```bash
# Terminal 1: Backend (FastAPI)
uvicorn webapp.backend.main:app --reload --port 8000

# Terminal 2: Frontend (React + Vite, redesigned with shared design system)
cd webapp/frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

### Run the Defense Site (`site/`)

A polished, read-only Astro site that animates every pipeline stage using the
already-saved outputs. Built for the supervisor and committee — no live runs.

```bash
# 1. Extract a 30-second demo MP4 + WAV from the source video (idempotent, ~1 min)
python videos/extract_demo_clip.py

# 2. (Optional but recommended) Capture real EasyOCR bboxes + scroll deltas
#    on Slice A so the OCR + scroll-detection stages render real data
python site/scripts/instrument_slice_A.py

# 3. Build the asset bundle (panoramas → WebP, waveform peaks, eval scores)
cd site
python scripts/build_assets.py

# 4. Run the dev server
npm install
npm run dev      # http://localhost:4321
```

Routes: `/`, `/visual`, `/audio`, `/results`, `/explore`. Static export (`npm run build`) deploys to any static host.

### Run the OCR Engine Comparison (`ocr_comparison/`)

```bash
pip install paddlepaddle paddleocr "python-doctr[torch]" transformers torch psutil

# Run each engine on both ground-truth slices (skips already-done pairs)
python -m ocr_comparison.run_ocr_engine --engine tesseract   --slice all
python -m ocr_comparison.run_ocr_engine --engine easyocr     --slice all
python -m ocr_comparison.run_ocr_engine --engine paddle      --slice all
python -m ocr_comparison.run_ocr_engine --engine doctr       --slice all
python -m ocr_comparison.run_ocr_engine --engine craft_trocr --slice all

# Two segmentation passes: pure engines + Tesseract-dash-augmented
python -m ocr_comparison.regenerate_pure
python -m ocr_comparison.augment_dashes

# Aggregate both tables into ocr_comparison/output/comparison_report.{json,txt}
python -m ocr_comparison.evaluate_all
```

### Run Validation

```bash
# Regenerate panorama for ground-truth annotation
python validation/regenerate_panorama.py --chunk 17 --slice A

# Compute P/R/F1/CER/WER against annotated ground truth
python validation/validate_extraction.py
```

---

## Project Structure

```
.
├── ticker_extraction_v6/       # 5-step OCR pipeline (frame → scroll → panorama → OCR → segment)
│   ├── main.py                 # Entry point: --video, --chunk-minutes, --engine
│   ├── step1_extract_ticker.py # Frame extraction + ticker crop
│   ├── step2_scroll_detection.py
│   ├── step3_stitch_image.py   # Panorama stitching
│   ├── step4_ocr.py            # Dual-engine: EasyOCR text + Tesseract dash detection
│   ├── step5_segment.py        # Delimiter split + dedup
│   └── output/final/           # news_items.json, news_items_cleaned.json
│
├── llm_summarization/          # 9-LLM summarization + ROUGE/BERTScore evaluation
│   ├── summarize.py            # Run all 9 models
│   ├── evaluate.py             # Score against human reference
│   ├── output_cleaned/         # Ticker summaries (14h video)
│   └── output_asr/             # ASR summaries
│
├── pipeline/                   # Post-processing (outside frozen modules)
│   ├── improve_segmentation.py # Better dedup across all chunks/cycles
│   ├── clean_news_items.py     # LLM-based error correction (Gemini 2.5 Flash)
│   └── summarize_cleaned.py    # Wrapper to run summarization on cleaned items
│
├── validation/                 # Ground-truth validation
│   ├── regenerate_panorama.py  # Build panorama PNG for manual annotation
│   ├── validate_extraction.py  # Compute P/R/F1/CER/WER
│   └── ground_truth/           # Annotated headlines + instructions
│
├── asr/                        # Audio pipeline
│   ├── extract_audio.py        # Video → 16kHz mono WAV (PyAV)
│   ├── transcribe.py           # Whisper ASR with chunked streaming
│   ├── chunk_transcript.py     # Split into 15-min text chunks
│   ├── summarize_transcript.py # 2-level hierarchical LLM summarization
│   └── output/                 # transcript.json/.txt/.srt + chunks/
│
├── webapp/                     # Interactive workbench
│   ├── backend/                # FastAPI (upload, job queue, pipeline orchestration)
│   └── frontend/               # React + Vite + Tailwind (redesigned, uses design-system/)
│
├── site/                       # Defense site (Astro static export)
│   ├── src/                    # layouts, pages (visual, audio, results, explore), components
│   ├── public/data/            # built — gitignored, regenerated by scripts/build_assets.py
│   └── scripts/                # build_assets.py + instrument_slice_A.py (real OCR bboxes)
│
├── design-system/              # Shared visual identity (no build step)
│   ├── tokens.css              # CSS custom properties (light/dark, motion, type)
│   └── components/             # React stage components used by site/ AND webapp/frontend/
│
├── ocr_comparison/             # Multi-OCR head-to-head experiment
│   ├── engines/                # Tesseract, EasyOCR, Paddle, docTR, CRAFT+TrOCR adapters
│   ├── augment_dashes.py       # Tesseract PSM 6 dash augmentation
│   ├── regenerate_pure.py      # Pure-engine segmentation rebuild
│   └── evaluate_all.py         # Both comparison tables: pure + augmented
│
├── results/                    # Evaluation reports (validation_report.json)
├── requirements.txt
├── .env.example                # Template for API keys
├── PROGRESS_REPORT.md          # Detailed technical report
└── CLAUDE.md                   # Project conventions + module documentation
```

---

## Web App

The web interface wraps both pipelines into a user-friendly demo:

1. **Upload** any TV news video (`.mp4`)
2. **Preview** the video and drag a range slider to select a segment (max 30 min)
3. **Choose** the task: Ticker extraction, Audio transcription, or Both
4. **Select** which LLMs to run (default: all 9)
5. **View results**: each LLM's summary displayed in its own card with latency and token count
6. **Download** the full results as JSON

The backend runs pipelines in background threads with progress polling. Retry-with-backoff handles transient LLM rate limits automatically.

---

## Evaluation Methodology

### Ticker Extraction Validation

- Two 30-minute slices manually annotated (27 + 28 headlines)
- Combined union: 39 unique ground-truth headlines
- Fuzzy matching (rapidfuzz) at 60/70/80% thresholds
- Metrics: Precision, Recall, F1, Character Error Rate, Word Error Rate
- Per-stage comparison: raw OCR vs. LLM-cleaned

### LLM Summarization Evaluation

- ROUGE-1, ROUGE-2, ROUGE-L (n-gram overlap)
- BERTScore (semantic similarity via contextual embeddings)
- Ranked by BERTScore F1 descending
- Same human reference for all 9 models (fair comparison)

---

## Tech Stack

| Component | Technology |
|---|---|
| Video processing | OpenCV, NumPy |
| OCR | EasyOCR (text), Tesseract (delimiter detection) |
| Speech-to-text | faster-whisper (CTranslate2, int8, CPU) |
| Audio I/O | PyAV |
| Text matching | rapidfuzz, python-Levenshtein |
| LLM inference | Groq, Gemini, HuggingFace, Cohere, Ollama, OpenAI SDKs |
| Evaluation | rouge-score, bert-score |
| Backend | FastAPI, uvicorn |
| Frontend | React, Vite, Tailwind CSS, rc-slider |

---

## License

This project is part of a Master's thesis at the German International University (GIU), Berlin.

---

## Author

**Marwan Mohamed**
M.Sc. student — GIU Berlin
