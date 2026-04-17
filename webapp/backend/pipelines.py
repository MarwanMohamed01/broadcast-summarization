"""
Pipeline wrappers for the web app.

Both `ticker_extraction_v6/` and `llm_summarization/` ship a top-level
`config.py`. Python's import system can only have one module named
`config`, so we explicitly load each of them by file path with
importlib and route them via `sys.modules['config']` at the right
moment so each downstream module sees the config it expects.

Exposes:
  - run_ticker_pipeline(job)
  - run_audio_pipeline(job)
"""

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()
V6_DIR = PROJECT_DIR / "ticker_extraction_v6"
LLM_DIR = PROJECT_DIR / "llm_summarization"
ASR_DIR = PROJECT_DIR / "asr"
PIPELINE_DIR = PROJECT_DIR / "pipeline"


def _load_from_file(name: str, file_path: Path):
    """Load a Python module from an explicit file path under a chosen name."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── 1) Load both `config` modules with explicit names ──
# Stored under unique names so we can reference each from our code.
V6_CONFIG = _load_from_file("v6_config", V6_DIR / "config.py")
LLM_CONFIG = _load_from_file("llm_config", LLM_DIR / "config.py")

# ── 2) Set sys.modules['config'] = v6's, then import v6 step modules ──
# Each v6 step module does `import config` at the top, which is then
# captured as a reference. Once captured, switching sys.modules['config']
# later does not affect them.
sys.modules["config"] = V6_CONFIG
sys.path.insert(0, str(V6_DIR))

from step1_extract_ticker import extract_ticker_frames  # noqa: E402
from step2_scroll_detection import detect_all_scrolls  # noqa: E402
from step3_stitch_image import stitch_panorama  # noqa: E402
from step4_ocr import ocr_panorama_chunks  # noqa: E402
from step5_segment import segment_news  # noqa: E402

# ── 3) Switch sys.modules['config'] to llm's, then import llm summarize ──
sys.modules["config"] = LLM_CONFIG
sys.path.insert(0, str(LLM_DIR))

import summarize as LLM_SUMMARIZE  # noqa: E402
from prompt import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

# ── 4) Make pipeline/ and asr/ importable for our helpers ──
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(ASR_DIR))


# ── 5) Replace LLM_SUMMARIZE.run_model with a retry-with-backoff version ──
# Every downstream caller (summarize_cleaned, summarize_transcript, our
# own _summarize_items) gets automatic retries on transient errors
# without having to change their code.

import time  # noqa: E402

_original_run_model = LLM_SUMMARIZE.run_model


def _is_retryable_error(err_str: str) -> bool:
    """Return True if an error message looks like a transient failure worth retrying."""
    s = (err_str or "").lower()
    signals = (
        "429", "rate limit", "rate_limit", "too many", "tpm limit",
        "503", "service unavailable", "502", "bad gateway", "504", "gateway timeout",
        "timed out", "timeout", "connection reset", "read timed out",
        "temporarily unavailable", "overloaded",
    )
    return any(s2 in s for s2 in signals)


def run_model_with_retry(provider, model_id, display, system, user,
                         max_retries=2, base_delay=20):
    """Wrap llm_summarization.summarize.run_model with exponential backoff
    on transient errors (429 rate limits, 503 service unavailable, timeouts).

    max_retries=2 means up to 3 total attempts.
    Non-retryable errors (e.g. 400 bad request, context length) fail immediately.
    """
    delay = base_delay
    last = None
    for attempt in range(max_retries + 1):
        result = _original_run_model(provider, model_id, display, system, user)
        last = result
        if result["status"] == "success":
            if attempt > 0:
                print(f"    [retry OK] {display} succeeded on attempt {attempt + 1}")
            return result

        err = result.get("error") or ""
        if not _is_retryable_error(err) or attempt == max_retries:
            return result

        print(
            f"    [retry] {display} transient error on attempt {attempt + 1}: "
            f"{err[:120]}  — waiting {delay}s"
        )
        time.sleep(delay)
        delay *= 2  # exponential
    return last


# Install the retry wrapper in place of the original
LLM_SUMMARIZE.run_model = run_model_with_retry


def _retry_call(fn, max_retries=2, base_delay=20):
    """Generic retry wrapper for functions that raise on transient errors.
    Used for clean_news_items.call_gemini / call_groq which raise rather
    than returning a structured error dict."""
    def wrapped(*args, **kwargs):
        delay = base_delay
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if not _is_retryable_error(str(e)) or attempt == max_retries:
                    raise
                print(
                    f"    [retry] {fn.__name__} transient error: "
                    f"{str(e)[:120]} — waiting {delay}s"
                )
                time.sleep(delay)
                delay *= 2
        if last_exc is not None:
            raise last_exc
    return wrapped


# Patch clean_news_items's API callers with retry.
# clean_news_items lives in pipeline/ (our code, not frozen), so we
# intercept its module-level functions after import.
import clean_news_items as _cn_module  # noqa: E402

if not getattr(_cn_module, "_retry_installed", False):
    _cn_module.call_gemini = _retry_call(_cn_module.call_gemini)
    _cn_module.call_groq = _retry_call(_cn_module.call_groq)
    _cn_module._retry_installed = True


# ── Helpers ──────────────────────────────────────────

def _stage(job, stage: str, progress: float, message: str = "") -> None:
    job.update(stage=stage, progress=progress, message=message)


def _frames_from_seconds(video_path: Path, start_sec: float, end_sec: float):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, int(start_sec * fps)), min(total_frames, int(end_sec * fps))


def _load_video_path(video_id: str) -> Path:
    uploads = PROJECT_DIR / "webapp" / "uploads"
    for ext in (".mp4", ".mov", ".mkv", ".avi"):
        p = uploads / f"{video_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No video found for id {video_id}")


# ── TICKER PIPELINE ──────────────────────────────────

def run_ticker_pipeline(job) -> dict:
    video_path = _load_video_path(job.video_id)
    start_frame, end_frame = _frames_from_seconds(
        video_path, job.start_sec, job.end_sec,
    )

    out_dir = job.job_dir / "ticker"
    frames_dir = out_dir / "ticker_frames"
    panorama_dir = out_dir / "panorama"
    ocr_dir = out_dir / "ocr"
    final_dir = out_dir / "final"
    for d in (frames_dir, panorama_dir, ocr_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Per-job override of v6 config dirs
    orig = (V6_CONFIG.PANORAMA_DIR, V6_CONFIG.OCR_DIR, V6_CONFIG.FINAL_DIR)
    V6_CONFIG.PANORAMA_DIR = panorama_dir
    V6_CONFIG.OCR_DIR = ocr_dir
    V6_CONFIG.FINAL_DIR = final_dir

    try:
        _stage(job, "frames", 0.05, f"Extracting frames {start_frame}-{end_frame}")
        frame_paths, _ = extract_ticker_frames(
            video_path, start_frame=start_frame, end_frame=end_frame,
            output_dir=frames_dir,
        )
        if not frame_paths:
            raise RuntimeError("No ticker frames extracted")

        _stage(job, "scroll", 0.20, f"Detecting scroll across {len(frame_paths)} frames")
        scroll_data, _ = detect_all_scrolls(frame_paths)
        if not scroll_data:
            raise RuntimeError("Scroll detection failed")

        _stage(job, "panorama", 0.35, "Stitching panorama")
        panorama_paths, _ = stitch_panorama(scroll_data)
        if not panorama_paths:
            raise RuntimeError("Panorama stitching failed")

        _stage(job, "ocr", 0.50, f"OCR on {len(panorama_paths)} panorama chunks")
        full_text, _ = ocr_panorama_chunks(panorama_paths)
        if not full_text:
            raise RuntimeError("OCR produced no text")

        _stage(job, "segment", 0.70, "Segmenting headlines")
        news_items, _ = segment_news(full_text)
        raw_items = [it["text"] for it in news_items]

        _stage(job, "clean_llm", 0.80, "Cleaning headlines with Gemini")
        cleaned_items = _clean_items_with_llm(raw_items)

        # Fallback: if LLM cleaning returned nothing (e.g. all providers
        # rate-limited/failed) but we had raw items, summarize the raw
        # items instead of sending an empty list to the LLMs.
        if not cleaned_items and raw_items:
            print(
                f"    [fallback] LLM cleaning returned 0 items — "
                f"using {len(raw_items)} raw items for summarization"
            )
            cleaned_items = raw_items

        if not cleaned_items:
            raise RuntimeError(
                "No headlines to summarize — both OCR segmentation "
                "and LLM cleaning produced zero items"
            )

        n_models = len(job.models) if job.models else len(LLM_CONFIG.MODELS)
        _stage(job, "summarize", 0.90, f"Summarizing with {n_models} LLMs ({len(cleaned_items)} items)")
        summaries = _summarize_items(cleaned_items, job.models)
    finally:
        V6_CONFIG.PANORAMA_DIR, V6_CONFIG.OCR_DIR, V6_CONFIG.FINAL_DIR = orig

    return {
        "task": "ticker",
        "start_sec": job.start_sec,
        "end_sec": job.end_sec,
        "raw_items": raw_items,
        "cleaned_items": cleaned_items,
        "summaries": summaries,
    }


# ── AUDIO PIPELINE ──────────────────────────────────

def run_audio_pipeline(job) -> dict:
    # Lazy imports — these don't conflict with `config`
    from extract_audio import extract_audio
    import transcribe as tr
    from chunk_transcript import chunk_transcript

    video_path = _load_video_path(job.video_id)
    out_dir = job.job_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "audio.wav"
    duration_sec = job.end_sec - job.start_sec

    _stage(job, "extract_audio", 0.05, f"Extracting {duration_sec/60:.0f}-min audio")
    extract_audio(
        video_path, audio_path,
        start_sec=job.start_sec, duration_sec=duration_sec,
    )

    _stage(job, "transcribe", 0.25, "Transcribing with Whisper (small)")
    original_output_dir = tr.OUTPUT_DIR
    tr.OUTPUT_DIR = out_dir
    try:
        tr.transcribe_chunked(
            audio_path, model_size="small", language="en",
            out_prefix="transcript", chunk_minutes=20,
        )
    finally:
        tr.OUTPUT_DIR = original_output_dir

    transcript_json = out_dir / "transcript.json"
    transcript_data = json.loads(transcript_json.read_text(encoding="utf-8"))
    transcript_text = " ".join(s["text"] for s in transcript_data["segments"])

    _stage(job, "chunk", 0.60, "Chunking transcript into 15-min blocks")
    chunks_dir = out_dir / "chunks"
    chunk_transcript(transcript_json, chunk_minutes=15, out_dir=chunks_dir)

    _stage(job, "summarize", 0.70, "Two-level LLM summarization")
    summaries = _summarize_transcript_chunks(chunks_dir, job.models)

    return {
        "task": "audio",
        "start_sec": job.start_sec,
        "end_sec": job.end_sec,
        "transcript_preview": transcript_text[:2000],
        "transcript_length_chars": len(transcript_text),
        "transcript_segments": len(transcript_data["segments"]),
        "summaries": summaries,
    }


# ── Internal: LLM cleaning + summarization ──

def _clean_items_with_llm(items: list[str]) -> list[str]:
    if not items:
        return []
    import clean_news_items as cn
    BATCH = 15
    cleaned = []
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    for bi, batch in enumerate(batches):
        cleaned.extend(cn.clean_batch(batch, bi, len(batches)))
    cleaned = cn.deduplicate(cleaned)
    return [c for c in cleaned if len(c) >= 30]


def _summarize_items(items: list[str], selected_models=None) -> list[dict]:
    news_items = [{"id": i + 1, "text": t} for i, t in enumerate(items)]
    user = build_user_prompt(news_items)
    models = LLM_CONFIG.MODELS
    if selected_models:
        models = [m for m in LLM_CONFIG.MODELS if m[2] in selected_models]
    return [
        LLM_SUMMARIZE.run_model(p, mid, d, SYSTEM_PROMPT, user)
        for p, mid, d in models
    ]


def _summarize_transcript_chunks(chunks_dir: Path, selected_models=None) -> list[dict]:
    import summarize_transcript as st

    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
    if not chunk_files:
        raise RuntimeError("No transcript chunks found")

    models = LLM_CONFIG.MODELS
    if selected_models:
        models = [m for m in LLM_CONFIG.MODELS if m[2] in selected_models]

    # Level 1
    level1 = {m[2]: [] for m in models}
    for chunk_path in chunk_files:
        text = st.load_chunk(chunk_path)
        for p, mid, d in models:
            level1[d].append(st.level1_summarize(text, p, mid, d))

    # Level 2
    out = []
    for p, mid, d in models:
        good = [r["summary"] for r in level1[d]
                if r["status"] == "success" and r.get("summary")]
        if not good:
            out.append({
                "display_name": d, "provider": p, "model_id": mid,
                "summary": None, "status": "error",
                "error": "no successful level-1 summaries",
                "latency_seconds": 0, "input_tokens": 0, "output_tokens": 0,
            })
            continue
        out.append(st.level2_summarize(good, p, mid, d))
    return out
