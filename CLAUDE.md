# CLAUDE.md

## Project
**Name:** ticker_extraction
**Thesis:** Vision-Based Extraction and LLM-Powered Summarization of TV News Tickers
**Author:** Masters student, GIU Berlin

This project implements two parallel pipelines that extract and summarize TV news content:
  1. **Visual path** — OCR of the scrolling ticker bar → LLM cleaning → 9-LLM summarization
  2. **Audio path** — Whisper ASR of spoken broadcast → 2-level LLM summarization

Both paths produce independently-evaluated LLM summary rankings (ROUGE + BERTScore)
and share the same 9-model LLM roster for fair comparison.

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
│   ├── reference_summary.txt    (for 27-min sample)
│   ├── output/                  (summaries for original 27-min sample)
│   ├── output_cleaned/          (summaries for LLM-cleaned 14-hour ticker)
│   └── output_asr/              (summaries for ASR transcript)
│
├── videos/                      ← raw input videos (.mp4)
│
├── pipeline/                    ← ticker post-processing scripts (NEW CODE)
│   ├── improve_segmentation.py  (re-segments v6 OCR text with better dedup)
│   ├── clean_news_items.py      (LLM error correction on raw v6 output)
│   └── summarize_cleaned.py     (runs llm_summarization on cleaned items)
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
│   ├── retry_level2.py          (retry Level-2 with backoff for rate limits)
│   └── output/
│       ├── audio_full.wav       (1.68 GB, 14.55 h mono 16 kHz)
│       ├── transcript_full.json/.txt/.srt (13,517 segments)
│       ├── chunks/              (59 × 15-min chunk .txt files)
│       └── _chunks_transcript_full/  (intermediate 20-min resume files)
│
├── design-system/               ← shared visual identity (NEW)
│   ├── tokens.css               (CSS custom properties used by both sites)
│   ├── components/              (shared React components)
│   │   ├── shared/              (StagePanel, ResultsHero, GTOverlay, Explorer)
│   │   ├── visual/              (8 visual-pipeline stage components)
│   │   └── audio/               (4 audio-pipeline stage components)
│   └── README.md
│
├── site/                        ← defense site (NEW — Astro static export)
│   ├── astro.config.mjs, tailwind.config.mjs, package.json
│   ├── src/{layouts,components,pages,styles}
│   ├── public/{data,video}      (built — gitignored, regenerated by build_assets.py)
│   ├── scripts/build_assets.py  (idempotent asset pipeline)
│   └── README.md
│
├── results/                     ← evaluation outputs
│   └── validation_report.json   (full Part 1 metrics)
│
├── old/                         ← superseded experiments (for reference)
│   ├── extract_headlines_llm.py
│   └── resegment.py
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

## Web Surfaces (in `site/`, `webapp/frontend/`, `design-system/`)

Two parallel web deliverables share one design system:

### `site/` — Defense site (read-only, static)
Astro 4 static export with React islands for animation. Five routes:
landing, `/visual` (8 animated stage panels), `/audio` (5 animated stage
panels), `/results` (P/R/F1/CER + ROUGE/BERTScore tables + GT-vs-extracted
overlay), `/explore` (tabbed data browser: headlines / transcript /
summaries / raw OCR with search). Zero runtime API calls — everything is
precomputed by `scripts/build_assets.py`. Uses 30-sec demo MP4 + WAV
extracted from the source video at 08:30:00 (matches Slice A so the OCR
walkthrough lines up with the annotated panorama).

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

## Rules Claude MUST Follow
1. **Never modify** `ticker_extraction_v6/` or `llm_summarization/` — they are frozen working code.
   (Writing to their `output/` folders is allowed — that's data, not code.)
2. **Always use relative paths** from the project root. New scripts in `pipeline/`, `validation/`,
   `asr/`, `ocr_comparison/` should use `PROJECT_DIR = Path(__file__).parent.parent.resolve()`.
3. **Always wrap** file I/O and API calls in `try/except` with meaningful error messages.
4. **Always log errors** to a `logs/` folder at project root (create if missing).
5. **Always skip already-processed files** — check for existing output before processing.
6. **Use the existing venv** at `news_summarization/venv/` for all new scripts.
7. Before running any destructive command (delete, overwrite, force-push), confirm with the user.
8. When adding new code, explain the plan first before writing it.

## Folders To Ignore
`old/` (superseded experiments), anything inside `ticker_extraction_v6/output/chunks/*/panorama/` (cleaned up to save disk),
`asr/output/_chunks_transcript_full/` (intermediate resume files), any `ticker_extraction_v2/..v5/` or legacy
`news_summarization/` folders if they reappear.
