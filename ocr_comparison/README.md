# OCR Comparison

Compare multiple OCR engines on the same two ground-truth panoramas
(Slice A and Slice B) to produce a thesis-grade head-to-head table.

## Engines

| Name | Category | Install |
|---|---|---|
| `tesseract` | Classic LSTM | already installed |
| `easyocr` | Deep (CRAFT + CRNN) | already installed |
| `paddle` | Deep (PP-OCRv4) | `pip install paddlepaddle paddleocr` |
| `doctr` | Deep (DB + CRNN, Mindee) | `pip install "python-doctr[torch]"` |
| `gemini` | LLM vision | already installed (`google-generativeai`) |
| `craft_trocr` | Hybrid (EasyOCR detector + TrOCR recognizer) | `pip install transformers torch` |

## One-time install (CPU-only)

```bash
pip install paddlepaddle paddleocr "python-doctr[torch]" transformers torch psutil
```

(Tesseract binary + EasyOCR + Gemini SDK are already present in the project env.)

## Run

```bash
# 1. Run OCR for each engine on both slices
python -m ocr_comparison.run_ocr_engine --engine tesseract --slice all
python -m ocr_comparison.run_ocr_engine --engine easyocr  --slice all
python -m ocr_comparison.run_ocr_engine --engine paddle   --slice all
python -m ocr_comparison.run_ocr_engine --engine doctr    --slice all
python -m ocr_comparison.run_ocr_engine --engine craft_trocr --slice all

# 2a. (Re)build pure-engine segmentation from cached words.json
#     — this is what the engine produces with NO Tesseract help.
python -m ocr_comparison.regenerate_pure

# 2b. Apply Tesseract dash augmentation + re-segment every engine
#     — matches v6 production architecture.
python -m ocr_comparison.augment_dashes

# 3. Aggregate results into BOTH comparison tables
python -m ocr_comparison.evaluate_all
```

Re-runs of an (engine, slice) pair are skipped automatically — pass
`--force` to redo. Add `--skip-segment` to run OCR only.

## Outputs

```
ocr_comparison/output/
├── slice_A/
│   ├── _tesseract_dashes.json                 (cached dash positions)
│   ├── tesseract_words.json                   (raw OCR tokens with positions)
│   ├── tesseract_timing.json                  (wall-clock + memory)
│   ├── tesseract_raw_text.txt                 (PURE — engine alone)
│   ├── tesseract_headlines.json               (PURE — segmented)
│   ├── tesseract_segmentation_stats.json      (PURE)
│   ├── tesseract_raw_text_augmented.txt       (AUGMENTED — engine + dashes)
│   ├── tesseract_headlines_augmented.json     (AUGMENTED — segmented)
│   ├── tesseract_segmentation_stats_augmented.json
│   └── (same set for each other engine)
├── slice_B/ (same)
├── comparison_report.json    (both tables, machine-readable)
└── comparison_report.txt     (both tables, human-readable)
```

Logs: `logs/ocr_comparison_<engine>_<slice>.log`.

## What is measured

Per (engine, slice):
- **P/R/F1** at fuzzy thresholds 60 / 70 / 80%
- **CER, WER** on pairs matched at 70%
- **Wall-clock seconds**, **pixels/sec throughput**
- **RSS memory delta (MB)**
- **Exact-match count, missed-headlines list, false-positive list**

The hybrid `craft_trocr` is reported in a separate table since it
shares its detector with EasyOCR — it is a recognizer-only experiment,
not a full stand-alone engine.

## Fairness notes (for the thesis)

- Every engine receives the **same panorama PNGs** — no differences in
  detection preprocessing beyond what each engine does internally.
- Every engine's OCR text flows through the **same v6 segmentation
  pipeline** (`step5_segment.py`). This isolates the OCR engine as the
  only variable.
- The thesis reports **two side-by-side tables** so the reader can see
  both perspectives:
    - **Pure engines** — recognition only. Shows that scene-text
      recognizers (EasyOCR, PaddleOCR, docTR, TrOCR) skip the thin " - "
      delimiter glyph entirely, which causes their segmentation to
      collapse. This is an honest "what does engine X produce alone"
      finding.
    - **Engines + Tesseract dash augmentation** — recognition with the
      same dash-detection pass v6 production uses. Isolates
      *recognition quality* from delimiter detection. This matches v6's
      hybrid EasyOCR+Tesseract architecture and is the apples-to-apples
      number for ranking engines on text-recognition quality.
- No LLM cleaning pass is applied — raw OCR → segmentation →
  evaluation. All engines would likely gain ~5-8 F1 points with
  Gemini-based cleaning (analogous to the 84.8 → 88.1 F1 jump v6
  already reported after its cleaning pass).
- CPU-only execution. Report makes this explicit because TrOCR and
  PaddleOCR would be substantially faster on GPU.
- Gemini 2.5 Flash Vision was excluded from reported results: free-tier
  rate limits produced inconsistent timings (tiles stalling on 429
  retries rather than real inference cost), which would make speed
  comparisons misleading. The engine code remains under `engines/` for
  future use with a paid key.
