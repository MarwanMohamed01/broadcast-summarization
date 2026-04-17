# webapp/backend — FastAPI service

Runs on `http://localhost:8000`. React frontend (Phase B) hits it from
`http://localhost:5173`.

## Run

From the project root:

```bash
# install backend-specific deps (the rest come from the main project)
pip install -r webapp/backend/requirements.txt

# start the server with auto-reload on code changes
uvicorn webapp.backend.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for the auto-generated Swagger UI.

## Endpoints (MVP)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | health probe |
| POST | `/api/upload` | upload a .mp4 (multipart/form-data, field name `file`) |
| GET  | `/api/videos/{id}/info` | get duration / fps / resolution |
| GET  | `/api/videos/{id}` | stream video for the preview player |
| GET  | `/api/models` | list the 9 configured LLMs |
| POST | `/api/jobs` | submit a summarization job |
| GET  | `/api/jobs` | list all jobs |
| GET  | `/api/jobs/{id}` | job status |
| GET  | `/api/jobs/{id}/result` | final result (9 LLM summaries) |

## JobRequest body

```json
{
  "video_id": "abc123",
  "task": "ticker",         // or "audio" or "both"
  "start_sec": 0,
  "end_sec": 1800,           // 30 min max
  "models": ["Gemini 2.5 Flash", "Command-R (Cohere)"]  // optional; omit = all 9
}
```

## Storage

- Uploaded videos → `webapp/uploads/<video_id>.mp4` (gitignored)
- Job outputs → `webapp/jobs/<job_id>/` (gitignored)
  - `status.json` — current status
  - `result.json` — final result
  - `ticker/` — per-step OCR outputs (frames, panorama, ocr text, final)
  - `audio/` — extracted audio wav, transcript, chunks

Job state is held in memory for the process lifetime, and also
persisted to the filesystem so a restarted server can still serve
completed job results.

## Limits

- Max segment length per job: **30 min** (`MAX_SEGMENT_SECONDS` in main.py)
- Max upload size: **5 GB**
- Single-user MVP — no auth
