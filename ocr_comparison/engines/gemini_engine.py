"""
Gemini 2.5 Flash Vision adapter.

Strategy:
  - Tile the wide panorama into 3000px-wide strips, overlapping by 300px
    (fewer but larger tiles keeps us under the free-tier 20 RPM limit).
  - Upscale each tile 3x before sending so Gemini does not have to
    recognize 87px-tall text after its internal 3072px longest-side cap.
  - Retry on 429 using the server-provided retry_delay when available,
    else exponential backoff.
  - Print per-tile progress so long runs are visibly progressing.
"""
import io
import os
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image

from .base import EngineResult, Word
from ..config import (
    GEMINI_MODEL,
    GEMINI_TILE_W,
    GEMINI_TILE_OVERLAP,
    GEMINI_UPSCALE,
    GEMINI_THROTTLE_S,
)

name = "gemini"

_model = None


def _get_model():
    global _model
    if _model is None:
        import google.generativeai as genai
        from dotenv import load_dotenv
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "llm_summarization", ".env",
        )
        load_dotenv(env_path)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in llm_summarization/.env")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(GEMINI_MODEL)
    return _model


PROMPT = (
    "You are reading a TV news ticker. Extract every piece of visible text "
    "from this image EXACTLY as it appears, in strict left-to-right reading "
    "order. Preserve dashes ( - ) between headlines. Do NOT summarize, "
    "correct, or reorder anything. Output ONLY the extracted text, nothing else."
)


def _image_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    if GEMINI_UPSCALE and GEMINI_UPSCALE > 1:
        h, w = img_bgr.shape[:2]
        img_bgr = cv2.resize(
            img_bgr, (w * GEMINI_UPSCALE, h * GEMINI_UPSCALE),
            interpolation=cv2.INTER_CUBIC,
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _retry_delay_from_exception(e: Exception) -> float:
    """Extract the server-suggested retry delay (in seconds) from a 429."""
    msg = str(e)
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
    if match:
        return float(match.group(1))
    return 0.0


def _call_gemini(img_bgr: np.ndarray) -> str:
    import google.generativeai as genai  # noqa: F401
    model = _get_model()
    png_bytes = _image_to_png_bytes(img_bgr)
    for attempt in range(4):
        try:
            resp = model.generate_content([
                PROMPT,
                {"mime_type": "image/png", "data": png_bytes},
            ])
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                # Prefer server's hint, else exponential
                wait = _retry_delay_from_exception(e) or (10 * (attempt + 1))
                print(f"    [Gemini 429] sleeping {wait:.0f}s (attempt {attempt+1}/4)", flush=True)
                time.sleep(wait + 1)
                continue
            # Non-rate-limit error — surface it
            raise
    print("    [Gemini] giving up on this tile after 4 retries", flush=True)
    return ""


def run(panorama: np.ndarray) -> EngineResult:
    _get_model()
    h, w = panorama.shape[:2]
    step = GEMINI_TILE_W - GEMINI_TILE_OVERLAP
    tile_count = max(1, (w + step - 1) // step)
    print(f"  Gemini: panorama {w}x{h}, will send ~{tile_count} tiles "
          f"(tile={GEMINI_TILE_W}px, step={step}px, upscale={GEMINI_UPSCALE}x)",
          flush=True)

    texts: list[str] = []
    tiles_done = 0
    tile_start_time = time.time()

    for x in range(0, w, step):
        x_end = min(x + GEMINI_TILE_W, w)
        if x_end - x < 200:
            break
        tile = panorama[:, x:x_end]

        t0 = time.time()
        txt = _call_gemini(tile)
        call_s = time.time() - t0
        if txt:
            texts.append(txt)
        tiles_done += 1

        chars = len(txt) if txt else 0
        elapsed = time.time() - tile_start_time
        print(f"    tile {tiles_done}/{tile_count}: x={x} "
              f"call={call_s:.1f}s chars={chars} elapsed={elapsed/60:.1f}min",
              flush=True)

        # Throttle to stay under free-tier RPM
        time.sleep(GEMINI_THROTTLE_S)

    full_text = " ".join(texts)

    return EngineResult(
        full_text=full_text,
        words=[],
        meta={
            "engine": name,
            "model": GEMINI_MODEL,
            "tile_w": GEMINI_TILE_W,
            "tile_overlap": GEMINI_TILE_OVERLAP,
            "tiles_processed": tiles_done,
            "upscale": GEMINI_UPSCALE,
            "throttle_s": GEMINI_THROTTLE_S,
        },
    )
