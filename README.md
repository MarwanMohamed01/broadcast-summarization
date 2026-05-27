# Multimodal Vision- and Audio-Based Extraction and LLM-Powered Summarization of TV News

Master's Thesis — German International University (GIU), Berlin

---

## Overview

An end-to-end pipeline that automatically extracts and summarizes news content from a 14-hour Al Jazeera TV broadcast through three independent modalities, then evaluates every stage against manually-annotated ground truth.

| Path | Input | Method | Output |
|---|---|---|---|
| **Visual — OCR** | Scrolling ticker bar | 5-step CV pipeline (EasyOCR + Tesseract) | Extracted headlines + 9-LLM summaries |
| **Visual — VLM** | Same ticker panoramas | Vision-Language Model per tile (Gemini / Ministral) | JSON headlines + 9-LLM summaries |
| **Audio** | Spoken broadcast | Whisper ASR → 2-level LLM summarization | Full transcript + 9-LLM multi-paragraph summaries |

All three extraction outputs feed a shared 9-LLM summarization roster and are scored with ROUGE + BERTScore against separate human reference summaries. Extraction quality is validated against 39 manually-annotated ground-truth headlines. A separate **OCR engine comparison** (`ocr_comparison/`) benchmarks five engines under identical conditions. A **variance experiment** (`evaluation/`) confirms summarization determinism across three independent runs.

The repository also ships two web surfaces sharing a single design system:

- **`site/`** — polished, read-only **defense site** (Astro) that animates every pipeline stage with real data and displays all evaluation leaderboards.
- **`webapp/`** — **interactive workbench** (React + FastAPI) for live video upload and pipeline execution.

---

## Key Results

### Ticker Extraction (OCR pipeline vs 39 GT headlines)

| Stage | Recall @70% | Precision @70% | F1 @70% | CER |
|---|---|---|---|---|
| Raw v6 OCR | 97.4% | 75.0% | 84.8% | 23.4% |
| + LLM cleaning (Gemini 2.5 Flash) | 92.3% | 84.2% | **88.1%** | **13.9%** |

### VLM Extraction (same 39 GT headlines, full-video run)

| Pipeline | Items | F1 @70% | Recall | Precision | Cost |
|---|---|---|---|---|---|
| OCR + LLM cleaning (above) | 38 | **88.1%** | 92.3% | 84.2% | ≈ $0.05 |
| Gemini 2.5 Flash (paid VLM) | 100 | 64.9% | **100%** | 48.0% | **€8.28** actual billed |
| Ministral 3 14B (free, open-weights) | 3,214 | 71.0% | **100%** | 55.3% | **$0.00** |

Both VLMs achieve perfect recall at the cost of lower precision. The OCR pipeline leads on F1 by ~17 pp.

### OCR Engine Comparison (CPU-only, same two GT panoramas)

**Pure engines (each engine alone):**

| Engine | Mean F1 | Notes |
|---|---|---|
| Tesseract | **85.4%** | Only engine to natively detect `" - "` delimiters |
| docTR | 63.9% | — |
| PaddleOCR | 8.5% | — |
| EasyOCR | 0.0% | Skips `" - "` glyph → segmentation collapses |
| CRAFT + TrOCR | 0.0% | Same delimiter problem |

**Engines + Tesseract dash augmentation (v6 production architecture):**

| Engine | Mean F1 | CER | Time | Peak RAM |
|---|---|---|---|---|
| **EasyOCR** | **91.3%** | **8.8%** | 3.9 min | 1.8 GB |
| Tesseract | 85.4% | 10.2% | 0.6 min | 4 MB |
| docTR | 80.3% | 12.0% | 6.0 min | 2.1 GB |
| CRAFT + TrOCR | 29.9% | 29.8% | 52.4 min | 2.8 GB |
| PaddleOCR | 23.0% | 27.8% | 21.9 min | 596 MB |

EasyOCR's win in the augmented configuration validates v6's production choice.

### Whisper Transcription Quality (vs manually-annotated ground truth)

| Metric | Combined (1,567-word reference) |
|---|---|
| WER | **6.6%** |
| CER | **4.8%** |

### LLM Summarization — Visual Path (vs 687-word human reference)

| Model | BERT-F1 | ROUGE-L |
|---|---|---|
| **Gemini 2.5 Flash** | **0.865** | 0.238 |
| Llama 4 Scout 17B (Groq) | 0.853 | 0.237 |
| Llama 3.3 70B (Groq) | 0.850 | 0.208 |
| **Qwen3 32B (Groq)** | 0.844 | **0.305** |
| Llama 3 8B (HuggingFace) | 0.850 | 0.186 |
| Command-R (Cohere) | 0.844 | 0.159 |
| Llama 3.2 3B (Ollama) | 0.839 | 0.184 |
| Llama 3.1 8B (Groq) | 0.839 | 0.170 |
| Llama 3.1 8B (Ollama) | — | — (CPU timeout) |

### LLM Summarization — Audio Path (vs 1,908-word human reference)

| Model | BERT-F1 | ROUGE-L |
|---|---|---|
| **Command-R (Cohere)** | **0.778** | 0.104 |
| Qwen3 32B (Groq) | 0.770 | 0.095 |
| Llama 3.2 3B (Ollama) | 0.768 | 0.095 |
| Llama 3.3 70B (Groq) | 0.768 | 0.132 |
| **Llama 4 Scout 17B (Groq)** | 0.767 | **0.144** |
| Llama 3.1 8B (Groq) | 0.767 | 0.115 |
| Gemini 2.5 Flash | 0.767 | 0.117 |
| Llama 3 8B (HuggingFace) | 0.758 | 0.111 |
| Llama 3.1 8B (Ollama) | — | — (CPU timeout at L1) |

---

## Architecture

```
                       INPUT: TV News Video (.mp4)
                                 |
              +------------------+------------------+
              |                                     |
       VISUAL PATH                            AUDIO PATH
              |                                     |
  Frame Extraction (OpenCV)            Audio Extraction (PyAV)
              |                                     |
  Scroll Detection                    Whisper ASR (faster-whisper)
  (template matching)                 small model, int8, CPU
              |                                     |
  Panorama Stitching                  Time-based Chunking
    |              |                  (59 x 15-min chunks)
    |           3000px tiles                        |
    |              |                  2-Level LLM Summarization
    |         VLM call per tile       L1: per-chunk paragraph (x 9)
    |         (Gemini / Ministral)    L2: final synthesis (x 9)
    |              |                              |
  Dual-Engine OCR     JSON headlines             |
  EasyOCR + Tesseract                            |
    |              |                             |
  Segmentation    Aggregation                    |
    + Dedup        + Dedup                       |
    |              |                             |
  LLM Cleaning (Gemini 2.5 Flash)               |
    |              |                             |
    +------+-------+                             |
           |                                     |
    9-LLM Summarization  <----- same 9 models ----+
           |
    ROUGE + BERTScore vs human reference
```

---

## LLM Roster (shared across all paths)

| Provider | Model | Type |
|---|---|---|
| Groq | `llama-3.3-70b-versatile` | Cloud (free tier) |
| Groq | `llama-3.1-8b-instant` | Cloud (free tier) |
| Groq | `qwen/qwen3-32b` | Cloud (free tier) |
| Groq | `meta-llama/llama-4-scout-17b-16e-instruct` | Cloud (free tier) |
| Ollama | `llama3.2` (3B) | Local |
| Ollama | `llama3.1:8b` | Local |
| Google | `gemini-2.5-flash` | Cloud (free tier) |
| HuggingFace | `meta-llama/Meta-Llama-3-8B-Instruct` | Cloud (free tier) |
| Cohere | `command-r-08-2024` | Cloud (free trial) |

---

## Quick Start

### Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on PATH
- (Optional) [Ollama](https://ollama.com/) for local LLM inference
- Node.js 18+ (for the web surfaces)

### Installation

```bash
git clone https://github.com/MarwanMohamed01/broadcast-summarization.git
cd broadcast-summarization
pip install -r requirements.txt
```

### API Keys

```bash
cp llm_summarization/.env.example llm_summarization/.env
# Edit with your Groq / Gemini / HuggingFace / Cohere keys

cp vlm_extraction/.env.example vlm_extraction/.env
# Edit with your Gemini / Mistral keys for VLM extraction
```

### Run the OCR Pipeline

```bash
# Short video
cd ticker_extraction_v6
python main.py --video ../videos/your_video.mp4

# Long video (14+ hours) — 30-minute chunks
python main.py --video ../videos/your_video.mp4 --chunk-minutes 30
```

### Run Post-Processing + Summarization (Visual / OCR)

```bash
# Better dedup across all chunks
python pipeline/improve_segmentation.py

# LLM error correction (Gemini 2.5 Flash)
python pipeline/clean_news_items.py

# 9-LLM summarization on cleaned headlines
python pipeline/summarize_cleaned.py

# Evaluate vs human reference
python pipeline/evaluate_cleaned_summaries.py
```

### Run VLM Extraction

```bash
# Gemini paid run (all panorama tiles)
python -m vlm_extraction.extract --vlm gemini --slice all --runs 1

# Ministral open-weights run (free tier)
python -m vlm_extraction.opensource_vlm run --slice full_video

# Evaluate extraction quality
python -m vlm_extraction.evaluate

# 3-way comparison report (OCR vs Gemini vs Ministral)
python -m vlm_extraction.opensource_vlm evaluate
```

### Run the Audio Pipeline

```bash
# Extract audio track
python asr/extract_audio.py

# Transcribe (~3× realtime on CPU, chunked streaming)
python asr/transcribe.py

# Split into 15-minute chunks
python asr/chunk_transcript.py

# 2-level hierarchical LLM summarization
python asr/summarize_transcript.py

# Evaluate vs human reference
python asr/evaluate_audio_summaries.py
```

### Run the OCR Engine Comparison

```bash
pip install paddlepaddle paddleocr "python-doctr[torch]" transformers torch psutil

# Run each engine on both ground-truth slices
python -m ocr_comparison.run_ocr_engine --engine tesseract   --slice all
python -m ocr_comparison.run_ocr_engine --engine easyocr     --slice all
python -m ocr_comparison.run_ocr_engine --engine paddle      --slice all
python -m ocr_comparison.run_ocr_engine --engine doctr       --slice all
python -m ocr_comparison.run_ocr_engine --engine craft_trocr --slice all

# Segmentation passes
python -m ocr_comparison.regenerate_pure
python -m ocr_comparison.augment_dashes

# Both comparison tables → ocr_comparison/output/comparison_report.{json,txt}
python -m ocr_comparison.evaluate_all
```

### Run Extraction Validation

```bash
# Compute P/R/F1/CER/WER vs annotated ground truth (39 headlines)
python validation/validate_extraction.py
```

### Run the Defense Site (`site/`)

The asset bundle (`site/public/data/` + `site/public/video/`, ~2.8 MB) is committed so the site works immediately after `npm install` with no source video needed.

```bash
cd site
npm install
npm run dev      # http://localhost:4321
```

Routes: `/` (landing), `/visual` (OCR pipeline), `/vlm` (VLM comparison), `/audio` (ASR pipeline), `/results` (leaderboards), `/explore` (data browser).

To regenerate assets from a new source video:
```bash
python videos/extract_demo_clip.py          # 30-second demo clip
python site/scripts/instrument_slice_A.py   # real EasyOCR bboxes + scroll deltas
python site/scripts/build_assets.py         # panoramas, waveform, eval scores
```

### Run the Interactive Workbench (`webapp/`)

```bash
# Terminal 1: Backend
uvicorn webapp.backend.main:app --reload --port 8000

# Terminal 2: Frontend
cd webapp/frontend && npm install && npm run dev
```

Open http://localhost:5173. Upload any `.mp4`, select a time range, choose OCR / ASR / both, and watch the 9 LLMs compete in real time.

---

## Project Structure

```
.
├── ticker_extraction_v6/       # FROZEN — 5-step OCR pipeline
│   ├── main.py                 # --video, --chunk-minutes
│   ├── step1..step5_*.py       # frame extract → scroll → stitch → OCR → segment
│   └── output/final/           # news_items.json (raw), news_items_cleaned.json
│
├── vlm_extraction/             # VLM-based ticker extraction (parallel to OCR)
│   ├── extract.py              # top-level runner (--vlm, --slice, --runs)
│   ├── aggregate.py            # cross-tile dedup → headlines_combined.json
│   ├── evaluate.py             # P/R/F1/CER vs GT; results/vlm_evaluation.json
│   ├── opensource_vlm.py       # Ministral runner + 3-way comparison report
│   ├── tiler.py                # 3000×87 px tiles with 1000 px overlap
│   ├── adapters/               # gemini, mistral, openai, anthropic, groq, hf
│   └── output/                 # gemini/, mistral/, groq_scout/ canonical runs
│
├── llm_summarization/          # FROZEN — 9-LLM summarization + evaluation
│   ├── summarize.py / evaluate.py
│   ├── reference_summary.txt        # human reference — visual path
│   ├── reference_summary_audio.txt  # human reference — audio path
│   ├── output_cleaned/         # canonical visual summaries + evaluation_latest.json
│   └── output_asr/             # canonical audio summaries + evaluation_latest.json
│
├── pipeline/                   # post-processing (outside frozen modules)
│   ├── improve_segmentation.py
│   ├── clean_news_items.py     # LLM OCR error correction
│   ├── clean_vlm_headlines.py  # LLM VLM output correction
│   ├── summarize_cleaned.py
│   └── evaluate_cleaned_summaries.py
│
├── asr/                        # Audio pipeline
│   ├── extract_audio.py        # video → 16 kHz mono WAV (PyAV)
│   ├── transcribe.py           # faster-whisper, chunked streaming
│   ├── chunk_transcript.py     # 59 × 15-min text chunks
│   ├── summarize_transcript.py # 2-level hierarchical LLM summarization
│   ├── merge_asr_runs.py       # combines all retry runs → latest.json
│   ├── evaluate_audio_summaries.py
│   ├── eval/                   # ASR ground truth + Whisper output (text files)
│   └── output/                 # transcript_full.{json,txt,srt}, chunks/
│
├── ocr_comparison/             # Multi-OCR head-to-head experiment
│   ├── engines/                # tesseract, easyocr, paddle, doctr, craft_trocr
│   ├── run_ocr_engine.py       # top-level runner
│   ├── augment_dashes.py       # Tesseract PSM 6 dash augmentation pass
│   ├── regenerate_pure.py      # pure-engine segmentation rebuild
│   └── evaluate_all.py         # both comparison tables
│
├── evaluation/                 # Determinism / variance experiment (3-run repeat)
│   ├── run_variance_summarization.py
│   └── aggregate_variance.py   # → results/variance_report.{json,md}
│
├── validation/                 # Extraction validation vs annotated GT
│   ├── validate_extraction.py  # P/R/F1/CER/WER; → results/validation_report.json
│   └── ground_truth/           # slice_A/B_headlines.txt (39 total GT headlines)
│
├── results/                    # Canonical evaluation outputs (thesis tables)
│   ├── validation_report.json
│   ├── asr_evaluation.json
│   ├── asr_summary_evaluation.json
│   ├── visual_14h_summary_evaluation.json
│   ├── vlm_evaluation.json / .md
│   ├── vlm_opensource_evaluation.json / .md
│   └── variance_report.json / .md
│
├── site/                       # Defense site (Astro static export)
│   ├── src/pages/              # index, visual, vlm, audio, results, explore
│   ├── public/data/            # prebuilt asset bundle (~2.8 MB, committed)
│   └── scripts/                # build_assets.py, instrument_slice_A.py
│
├── webapp/                     # Interactive workbench
│   ├── backend/                # FastAPI — upload, job queue, pipeline orchestration
│   └── frontend/               # React + Vite + Tailwind (uses design-system/)
│
├── design-system/              # Shared visual identity (no build step)
│   ├── tokens.css              # CSS custom properties (colors, motion, spacing)
│   └── components/             # React stage panels used by site/ and webapp/
│
├── scripts/                    # Thesis figure generation
│   ├── generate_thesis_figures_ch3.py   # F3.1–F3.4 methodology flowcharts
│   ├── generate_thesis_figures_ch3_f3_0.py  # F3.0 frame-sampling diagram
│   └── generate_thesis_figures_ch5.py   # Ch5 results charts
│
├── videos/
│   └── extract_demo_clip.py    # extracts 30-second demo MP4 + WAV for site/
│
├── requirements.txt
├── .env.example
├── PROGRESS_REPORT.md          # detailed technical report
└── CLAUDE.md                   # project conventions + module documentation
```

---

## Evaluation Methodology

### Extraction Validation
- Two 30-minute slices manually annotated → 27 + 28 = 39 unique combined GT headlines
- Fuzzy matching (rapidfuzz) at 60 / 70 / 80% thresholds
- Metrics: Precision, Recall, F1 @ threshold, CER, WER, exact-match count
- Evaluated separately for: raw v6 OCR, LLM-cleaned OCR, Gemini VLM, Ministral VLM

### Transcription Quality
- Two short WAV slices manually transcribed (1,567-word reference total)
- Scored with jiwer: WER, CER, MER, WIL; both normalized to lowercase

### Summarization Evaluation
- ROUGE-1, ROUGE-2, ROUGE-L (n-gram overlap)
- BERTScore F1 (semantic similarity via contextual embeddings)
- Visual path scored against `reference_summary.txt` (687 words)
- Audio path scored against `reference_summary_audio.txt` (1,908 words)
- 8 of 9 models succeeded on both paths; Llama 3.1 8B (Ollama) timed out on CPU

---

## Tech Stack

| Component | Technology |
|---|---|
| Video / audio I/O | OpenCV, PyAV |
| OCR | EasyOCR (text recognition), Tesseract (delimiter detection) |
| VLM extraction | Google Gemini 2.5 Flash, Mistral Ministral 3 14B |
| Speech-to-text | faster-whisper (CTranslate2, int8, CPU) |
| Text dedup | rapidfuzz, python-Levenshtein |
| LLM providers | Groq, Google Gemini, HuggingFace, Cohere, Ollama, OpenAI SDKs |
| Evaluation | rouge-score, bert-score, jiwer |
| Defense site | Astro 4, React, Tailwind CSS |
| Workbench backend | FastAPI, uvicorn |
| Workbench frontend | React, Vite, Tailwind CSS, Framer Motion |

---

## License

This project is part of a Master's thesis at the German International University (GIU), Berlin.

## Author

**Marwan Mohamed**
M.Sc. student — GIU Berlin
