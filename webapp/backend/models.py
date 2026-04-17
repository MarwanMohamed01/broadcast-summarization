"""Pydantic request/response schemas for the web API."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── Video metadata returned after upload ──

class VideoInfo(BaseModel):
    video_id: str
    filename: str
    size_mb: float
    duration_seconds: float
    width: int
    height: int
    fps: float
    total_frames: int


# ── Job submission request body ──

class JobRequest(BaseModel):
    video_id: str
    task: Literal["ticker", "audio", "both"]
    start_sec: float = Field(0, ge=0, description="Start of time range in seconds")
    end_sec: float = Field(..., gt=0, description="End of time range in seconds")
    models: Optional[list[str]] = Field(
        default=None,
        description="LLM model keys to run. None = all 9 configured models."
    )


# ── Job status + result ──

class JobStatus(BaseModel):
    job_id: str
    video_id: str
    task: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    progress: float = Field(0, ge=0, le=1, description="Overall progress 0-1")
    stage: str = ""
    message: str = ""
    created_at: str
    updated_at: str
    start_sec: float
    end_sec: float


class LLMSummary(BaseModel):
    display_name: str
    provider: str
    model_id: str
    summary: Optional[str]
    status: str
    error: Optional[str] = None
    latency_seconds: float
    input_tokens: int
    output_tokens: int


class JobResult(BaseModel):
    job_id: str
    task: str
    status: str
    ticker_items: Optional[list[str]] = None
    ticker_cleaned_items: Optional[list[str]] = None
    ticker_summaries: Optional[list[LLMSummary]] = None
    audio_transcript_preview: Optional[str] = None
    audio_summaries: Optional[list[LLMSummary]] = None
    error: Optional[str] = None
