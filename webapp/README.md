# webapp — News Ticker / Audio Summarization

A web app wrapper around the existing Python pipelines (ticker extraction,
LLM cleaning, audio ASR, two-level summarization).

## How to run (two terminals)

### Terminal 1 — backend (FastAPI on port 8000)

From the project root:

```bash
pip install -r webapp/backend/requirements.txt    # one time
uvicorn webapp.backend.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for the auto-generated Swagger API.

### Terminal 2 — frontend (Vite + React on port 5173)

```bash
cd webapp/frontend
npm install                                       # one time
npm run dev
```

Then open <http://localhost:5173/> in a browser.

## User flow

1. Upload a `.mp4` (any TV news video).
2. Pick a time range (≤30 minutes — to keep processing under ~20 min wall time).
3. Pick the task: **ticker** (OCR the scrolling bar), **audio** (transcribe and summarize voice), or **both**.
4. Pick which LLMs to run (default: all 9).
5. Click **Summarize**. Page polls the backend every 2 seconds for status.
6. When done, the page renders one card per LLM with its summary, latency, and token count.

## Limits (MVP)

- Max segment length per job: **30 min** (`MAX_SEGMENT_SECONDS` in [`backend/main.py`](backend/main.py))
- Max upload size: **5 GB**
- Single-user, no auth
- Job state held in memory + persisted to `webapp/jobs/<id>/`

## Storage

| Path | Contents |
|---|---|
| `webapp/uploads/<video_id>.mp4` | uploaded videos (gitignored) |
| `webapp/jobs/<job_id>/status.json` | live job status |
| `webapp/jobs/<job_id>/result.json` | final 9-LLM summary |
| `webapp/jobs/<job_id>/ticker/` | per-job OCR intermediate files |
| `webapp/jobs/<job_id>/audio/` | per-job WAV + transcript + chunks |

All `webapp/uploads/`, `webapp/jobs/`, and `webapp/frontend/node_modules/`
are gitignored.

## Architecture

```
React (5173) ──HTTP/JSON──▶ FastAPI (8000) ──direct imports──▶
    │                            │
    │                            ├──▶ ticker_extraction_v6/  (frozen)
    │                            ├──▶ llm_summarization/     (frozen)
    │                            ├──▶ pipeline/              (LLM cleaning, summ wrapper)
    │                            ├──▶ asr/                   (Whisper + chunks + summ)
    │                            └──▶ webapp/jobs/<id>/      (per-job outputs)
    │
    └──polls every 2s──▶ /api/jobs/<id> for status
                         /api/jobs/<id>/result for the final 9-summary payload
```

No subprocesses — all heavy lifting runs in-process. Whisper / EasyOCR
models load once at the first request and stay in memory for subsequent
requests (lazy-loaded by their respective pipelines).

## Phase status

- ✅ **Phase A** — backend (upload, video info, job submit/status/result, models list)
- ✅ **Phase B** — minimal frontend (upload → configure → submit → poll → display)
- ⏳ **Phase C** — Server-Sent Events for fine-grained live progress
- ⏳ **Phase D** — video player + draggable time-range slider
- ⏳ **Phase E** — Tailwind UI polish, results download, history page
