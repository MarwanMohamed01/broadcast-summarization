"""
FastAPI backend for the ticker_extraction web app.

Endpoints:
  POST /api/upload                — upload a video, returns VideoInfo
  GET  /api/videos/{id}/info      — get video metadata
  GET  /api/videos/{id}           — stream video for preview
  POST /api/jobs                  — submit a summarization job
  GET  /api/jobs                  — list all jobs
  GET  /api/jobs/{id}             — get job status
  GET  /api/jobs/{id}/result      — get full job result
  GET  /api/models                — list configured LLM model display names
  GET  /health                    — simple health check

Run:
    uvicorn webapp.backend.main:app --reload --port 8000
"""

import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()
UPLOADS_DIR = PROJECT_DIR / "webapp" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Backend-local imports
sys.path.insert(0, str(Path(__file__).parent))
from models import VideoInfo, JobRequest  # noqa: E402
from jobs import create_job, get_job, list_jobs  # noqa: E402
from pipelines import run_ticker_pipeline, run_audio_pipeline  # noqa: E402

# Also allow importing llm_summarization's config for the /api/models endpoint
sys.path.insert(0, str(PROJECT_DIR / "llm_summarization"))


# ── FastAPI app ──

app = FastAPI(
    title="News Ticker Summarization API",
    description="Upload a video, extract the ticker / audio, summarize with 9 LLMs.",
    version="0.1.0",
)

# Allow any localhost / 127.0.0.1 port for dev (Vite picks 5173, 5174, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Upload + video metadata ──

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


def _get_video_info(video_path: Path, video_id: str) -> VideoInfo:
    """Read video metadata with OpenCV and return a VideoInfo."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(400, "Could not open uploaded video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / fps if fps > 0 else 0
    return VideoInfo(
        video_id=video_id,
        filename=video_path.name,
        size_mb=round(video_path.stat().st_size / 1e6, 1),
        duration_seconds=round(duration, 1),
        width=width,
        height=height,
        fps=round(fps, 2),
        total_frames=total_frames,
    )


@app.post("/api/upload", response_model=VideoInfo)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video. Returns a video_id and metadata."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Extension {ext} not allowed. Use: {ALLOWED_EXTENSIONS}")

    video_id = uuid.uuid4().hex[:12]
    dest = UPLOADS_DIR / f"{video_id}{ext}"

    total = 0
    with open(dest, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large (>{MAX_UPLOAD_BYTES // 1e9:.0f} GB)")
            out.write(chunk)

    return _get_video_info(dest, video_id)


def _resolve_video(video_id: str) -> Path:
    for ext in ALLOWED_EXTENSIONS:
        p = UPLOADS_DIR / f"{video_id}{ext}"
        if p.exists():
            return p
    raise HTTPException(404, f"Video {video_id} not found")


@app.get("/api/videos/{video_id}/info", response_model=VideoInfo)
def video_info(video_id: str):
    path = _resolve_video(video_id)
    return _get_video_info(path, video_id)


@app.get("/api/videos/{video_id}")
def stream_video(video_id: str):
    """Stream the uploaded video for preview in the frontend."""
    path = _resolve_video(video_id)
    return FileResponse(path, media_type="video/mp4", filename=path.name)


# ── LLM models catalog ──

@app.get("/api/models")
def list_models():
    """Return the 9 configured LLM models."""
    import config as llm_config  # noqa
    return [
        {"provider": p, "model_id": m, "display_name": d}
        for p, m, d in llm_config.MODELS
    ]


# ── Jobs ──

# Maximum segment length the user can ask for. 30 minutes matches the
# chunked OCR pipeline's natural unit and keeps requests bounded.
MAX_SEGMENT_SECONDS = 30 * 60


@app.post("/api/jobs")
def submit_job(request: JobRequest):
    """Submit a summarization job. Returns job_id + initial status."""
    if request.end_sec <= request.start_sec:
        raise HTTPException(400, "end_sec must be greater than start_sec")
    segment = request.end_sec - request.start_sec
    if segment > MAX_SEGMENT_SECONDS:
        raise HTTPException(
            400,
            f"Segment length {segment/60:.1f} min exceeds maximum "
            f"{MAX_SEGMENT_SECONDS/60:.0f} min",
        )

    # Verify video exists
    _resolve_video(request.video_id)

    job = create_job(
        request.video_id, request.task,
        request.start_sec, request.end_sec,
        request.models,
    )

    # Pick pipeline fn
    if request.task == "ticker":
        pipeline_fn = run_ticker_pipeline
    elif request.task == "audio":
        pipeline_fn = run_audio_pipeline
    elif request.task == "both":
        def pipeline_fn(job):
            ticker = run_ticker_pipeline(job)
            audio = run_audio_pipeline(job)
            return {"task": "both", "ticker": ticker, "audio": audio}
    else:
        raise HTTPException(400, f"Unknown task: {request.task}")

    job.run(pipeline_fn)
    return job.to_status_dict()


@app.get("/api/jobs")
def jobs_list():
    """List all known jobs (in-memory for this process run)."""
    return [j.to_status_dict() for j in list_jobs()]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return job.to_status_dict()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != "done":
        return JSONResponse(
            status_code=409,
            content={"detail": f"Job status is {job.status}, not done",
                     "status": job.status, "error": job.error},
        )
    return job.result
