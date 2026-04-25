# site/ — Defense site

A polished, read-only thesis-presentation site that animates every stage of
both pipelines (visual + audio) using the precomputed outputs already saved
on disk. Built for the supervisor and committee — not for live processing.

> **Looking for the interactive workbench?** That's [`webapp/frontend/`](../webapp/frontend/).
> Both share visual identity via [`design-system/`](../design-system/).

## Stack

- **Astro 4** — static export (HTML by default, hydrates only interactive islands)
- **React 18** — animation components, charts, transcript scrubber
- **Framer Motion 11** — every scroll-triggered + spring animation
- **Tailwind 3** — wired to shared CSS variables in `../design-system/tokens.css`
- **Recharts** — bar charts on `/results`
- **Lucide React** — icons
- **`@fontsource/{inter, jetbrains-mono}`** — fonts ship offline

## Routes

| Path | What it shows |
|---|---|
| `/` | Landing — pitch + key numbers |
| `/visual` | 8 stage panels: source video → frame extraction → scroll detection → panorama stitch → OCR → segmentation → LLM clean → 9-LLM fan-out → eval |
| `/audio` | 5 stage panels: source video → audio extract → Whisper ASR → 15-min chunking → L1→L2 hierarchical summary, then shared 9-LLM fan-out + eval |
| `/results` | Headline P/R/F1/CER plus 9-LLM ROUGE/BERTScore plus GT-vs-extracted overlay |
| `/explore` | Tabbed data browser: Headlines / Transcript / Summaries / Raw OCR with search |

## Develop

```bash
# 1. Install Node deps
cd site
npm install

# 2. Build the asset bundle (one-shot — pulls JSON + crops panoramas + decimates WAV)
npm run build:assets         # runs python scripts/build_assets.py

# 3. Run the dev server
npm run dev                  # http://localhost:4321
```

Hot-reload works on every `.astro` page and `.jsx` component, including the
shared design-system components imported from `../design-system/components/`.

## Production build & deploy

```bash
npm run build:assets         # idempotent — skips up-to-date files
npm run build                # writes site/dist/
```

Deploy `site/dist/` to any static host (Vercel, GitHub Pages, Netlify, S3).
The site makes zero network requests at runtime apart from the data files
under `/data/` which are bundled with the deploy.

## Refreshing assets

The asset script re-runs are idempotent. Re-run after:

- New OCR results in `ticker_extraction_v6/output/`
- New summaries in `llm_summarization/output*/`
- New ASR transcripts in `asr/output/`
- New ground-truth panoramas in `validation/ground_truth/`
- A new demo clip extracted by `videos/extract_demo_clip.py`

```bash
python scripts/build_assets.py
```

The script logs what it copied vs. skipped to `logs/site_build.log` and
enforces a 2 MB-per-file / 20 MB-total cap.

## Optional: instrumented Slice A run (real OCR bboxes + scroll deltas)

The frame-extraction, scroll-detection, and OCR stages on `/visual` will use
*real* per-frame counts, real median Δx, and real EasyOCR bounding boxes
when an extra one-shot run has been completed:

```bash
python site/scripts/instrument_slice_A.py        # ~5–10 min on CPU
python site/scripts/build_assets.py              # picks up the new files
```

The instrumenter re-runs v6's steps 1+2+4 on the same frame range as
ground-truth Slice A and persists what v6 itself doesn't save. Outputs
land under `site/public/data/instrumented/slice_A/`. Without this run the
stages still animate — they just use placeholder counts and the boxes
don't trace the visible characters.

## What's NOT here (by design)

- ❌ No live OCR / ASR / LLM API calls — everything is precomputed
- ❌ No upload form — that lives in the [interactive workbench](../webapp/frontend/)
- ❌ No reference to the `ocr_comparison/` engine experiment — out of scope
- ❌ No hidden backend — pure static export

## Data gaps + simplifications

After running both `python site/scripts/instrument_slice_A.py` and
`python site/scripts/build_assets.py`, the only purely-synthetic items
left are decorative.

- **Frame-extraction stage** — Real source/kept frame counts come from
  the instrumented run; without it, falls back to `pipeline_stats.json`
  (real total + computed kept). The 30-tile strip on screen is always
  illustrative; sample rate (`every 5th frame`) is real.
- **Scroll-detection stage** — Real median Δx pulled from the
  instrumented run's per-frame deltas. Without it, falls back to a
  representative 12 px label.
- **OCR stage** — When the instrumented bboxes JSON is present, the
  on-screen rectangles ARE EasyOCR's actual detections on the first
  3000-px segment of Slice A's panorama and they wrap the visible
  characters. Without it, falls back to evenly-spaced synthetic boxes.
- **ASR-stage waveform** — Real peaks decimated from
  `videos/demo_clip_30s.wav` (2000 buckets). Audio does NOT play in the
  browser by design (no licensing entanglement with the source broadcast).
- **`/visual` and `/audio` eval bars** — Real per-LLM ROUGE-L + BERTScore
  F1 from `llm_summarization/output*/evaluation_latest.json`, aggregated
  by `build_assets.py` into `eval_scores.json`. Live-data badge appears
  when the chart is using real numbers.
- **`/results` headline metrics** — Real P/R/F1/CER from
  `validation_report.json`.

## Folder layout

```
site/
├── astro.config.mjs
├── tailwind.config.mjs
├── package.json
├── public/
│   ├── data/         (built — gitignored, regenerated by build_assets.py)
│   └── video/        (built — gitignored, the 30-sec preview clip)
├── scripts/
│   └── build_assets.py
└── src/
    ├── layouts/BaseLayout.astro
    ├── components/{Nav,Footer}.astro
    ├── styles/global.css
    └── pages/
        ├── index.astro       (landing)
        ├── visual.astro
        ├── audio.astro
        ├── results.astro
        └── explore.astro
```
