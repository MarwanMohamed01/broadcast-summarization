"""
In-memory job queue with threading.

For MVP: one process-wide dict + a thread per running job. Results are
also persisted to disk (webapp/jobs/<job_id>/status.json, result.json)
so state survives server restarts if the job was already complete.

Upgrade path: replace with Celery + Redis if we ever need multi-user
parallelism or distributed workers.
"""

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()
JOBS_DIR = PROJECT_DIR / "webapp" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


class Job:
    """Represents a single background job with progress tracking."""

    def __init__(self, video_id: str, task: str, start_sec: float, end_sec: float,
                 models: Optional[list[str]] = None):
        self.job_id = uuid.uuid4().hex[:12]
        self.video_id = video_id
        self.task = task
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.models = models
        self.status = "queued"
        self.progress = 0.0
        self.stage = "queued"
        self.message = ""
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.job_dir = JOBS_DIR / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def to_status_dict(self) -> dict:
        """Return a serializable status dict."""
        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "task": self.task,
            "status": self.status,
            "progress": round(self.progress, 3),
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
        }

    def update(self, stage: str = None, progress: float = None,
               message: str = None, status: str = None) -> None:
        """Thread-safe update of job progress fields."""
        with self._lock:
            if stage is not None:
                self.stage = stage
            if progress is not None:
                self.progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                self.message = message
            if status is not None:
                self.status = status
            self.updated_at = datetime.now().isoformat()
        self._persist_status()

    def _persist_status(self) -> None:
        """Write status.json to disk so state survives server restart."""
        try:
            with open(self.job_dir / "status.json", "w", encoding="utf-8") as f:
                json.dump(self.to_status_dict(), f, indent=2)
        except Exception:
            pass  # non-critical

    def _persist_result(self) -> None:
        """Write result.json to disk."""
        if self.result is None:
            return
        try:
            with open(self.job_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(self.result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.error = f"Failed to persist result: {e}"

    def run(self, pipeline_fn: Callable[["Job"], dict]) -> None:
        """Start the job in a background thread."""
        def _target():
            try:
                self.update(status="running", stage="starting", progress=0.0)
                self.result = pipeline_fn(self)
                self._persist_result()
                self.update(status="done", stage="completed", progress=1.0,
                            message="Pipeline finished successfully")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.error = str(e)
                self.update(status="failed", message=f"{type(e).__name__}: {e}")
                # Write the full traceback to the job directory for debugging
                try:
                    with open(self.job_dir / "error.log", "w", encoding="utf-8") as f:
                        f.write(tb)
                except Exception:
                    pass

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()


# Process-wide job registry
_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def create_job(video_id: str, task: str, start_sec: float, end_sec: float,
               models: Optional[list[str]] = None) -> Job:
    job = Job(video_id, task, start_sec, end_sec, models)
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            return _JOBS[job_id]
    # Maybe it's persisted from a prior server run
    job_dir = JOBS_DIR / job_id
    status_path = job_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            # Synthesize a Job from persisted status (read-only)
            job = Job.__new__(Job)
            job.__dict__.update({
                "job_id": status["job_id"],
                "video_id": status["video_id"],
                "task": status["task"],
                "status": status["status"],
                "progress": status.get("progress", 0),
                "stage": status.get("stage", ""),
                "message": status.get("message", ""),
                "result": None,
                "error": None,
                "created_at": status["created_at"],
                "updated_at": status.get("updated_at", status["created_at"]),
                "start_sec": status.get("start_sec", 0),
                "end_sec": status.get("end_sec", 0),
                "_lock": threading.Lock(),
                "_thread": None,
                "job_dir": job_dir,
                "models": None,
            })
            # Load result if present
            result_path = job_dir / "result.json"
            if result_path.exists():
                job.result = json.loads(result_path.read_text(encoding="utf-8"))
            with _JOBS_LOCK:
                _JOBS[job_id] = job
            return job
        except Exception:
            return None
    return None


def list_jobs() -> list[Job]:
    with _JOBS_LOCK:
        return list(_JOBS.values())
