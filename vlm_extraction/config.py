"""Central config for the VLM extraction branch.

Loads keys from vlm_extraction/.env first, then falls back to
llm_summarization/.env for any key not set locally. Defines the model
registry, panorama input paths, and per-provider rate-limit / pricing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.parent.resolve()
VLM_DIR = PROJECT_DIR / "vlm_extraction"
OUTPUT_DIR = VLM_DIR / "output"
LOGS_DIR = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Load .env files in order: local overrides, then llm_summarization fallback.
load_dotenv(VLM_DIR / ".env", override=False)
load_dotenv(PROJECT_DIR / "llm_summarization" / ".env", override=False)


def _env(name: str, *aliases: str) -> Optional[str]:
    for key in (name, *aliases):
        val = os.environ.get(key)
        if val:
            return val
    return None


# ----- API keys -------------------------------------------------------------
GOOGLE_API_KEY = _env("GOOGLE_API_KEY", "GEMINI_API_KEY")
OPENAI_API_KEY = _env("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
HF_TOKEN = _env("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = _env("GROQ_API_KEY")
MISTRAL_API_KEY = _env("MISTRAL_API_KEY")

# Hard gate: OpenAI key is provided but billing is not active.
OPENAI_BILLING_ACTIVE = (
    os.environ.get("OPENAI_BILLING_ACTIVE", "false").lower() == "true"
)

# ----- Panorama input -------------------------------------------------------
GROUND_TRUTH_DIR = PROJECT_DIR / "validation" / "ground_truth"

SLICE_BOUNDARIES = {
    "slice_A": {"start_h": 8.5, "end_h": 9.0, "label": "08:30-09:00"},
    "slice_B": {"start_h": 13.0, "end_h": 13.5, "label": "13:00-13:30"},
    "full_video": {"start_h": 0.0, "end_h": 14.55, "label": "full 14.55 h"},
}

# Slice-level eval re-uses the single combined panorama produced for each
# 30-min ground-truth slice (same image OCR validation sees).
# Full-video mode iterates ALL 75 v6 chunk panoramas, mirroring how the
# OCR pipeline runs on the full 14h source.
V6_PANORAMA_DIR = PROJECT_DIR / "ticker_extraction_v6" / "output" / "panorama"
_FULL_VIDEO_PANOS = sorted(V6_PANORAMA_DIR.glob("panorama_chunk_*.png"))

PANORAMA_SOURCES = {
    "slice_A": [GROUND_TRUTH_DIR / "slice_A_panorama.png"],
    "slice_B": [GROUND_TRUTH_DIR / "slice_B_panorama.png"],
    "full_video": _FULL_VIDEO_PANOS,
}

GROUND_TRUTH_HEADLINES = {
    "slice_A": GROUND_TRUTH_DIR / "slice_A_headlines.txt",
    "slice_B": GROUND_TRUTH_DIR / "slice_B_headlines.txt",
}

# ----- Model registry ------------------------------------------------------
# `available` is computed from key presence + billing flag, NOT hard-coded.
MODEL_REGISTRY = {
    "gemini": {
        "provider": "google",
        "model_id": "gemini-2.5-flash",
        # Paid tier 1: 1000 RPM. Free tier was 10 RPM — set GEMINI_FREE_TIER=true
        # to revert if the user hasn't enabled billing yet.
        "rpm": 10 if os.environ.get("GEMINI_FREE_TIER", "").lower() == "true" else 1000,
        "is_paid": True,
        "available": GOOGLE_API_KEY is not None,
        "skip_reason": None if GOOGLE_API_KEY else "GOOGLE_API_KEY not set",
        # Gemini 2.5 Flash paid tier 1 pricing (USD per 1M tokens).
        # CORRECTED 2026-05-02: previously had wrong values (0.075 / 0.30),
        # which under-reported cost ~4×. Real paid-tier prices below.
        "input_per_1m_tokens_usd": 0.30,
        "output_per_1m_tokens_usd": 2.50,
        # Thinking tokens (billed separately) cost more. We disable thinking
        # in the adapter so this should always be 0 in practice.
        "thinking_per_1m_tokens_usd": 3.50,
    },
    "gpt4o_mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "rpm": 500,
        "is_paid": True,
        "available": OPENAI_API_KEY is not None and OPENAI_BILLING_ACTIVE,
        "skip_reason": (
            None
            if (OPENAI_API_KEY and OPENAI_BILLING_ACTIVE)
            else (
                "OPENAI_API_KEY not set"
                if not OPENAI_API_KEY
                else "OPENAI_BILLING_ACTIVE=false (deferred until top-up)"
            )
        ),
        "input_per_1m_tokens_usd": 0.15,
        "output_per_1m_tokens_usd": 0.60,
    },
    "claude": {
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-latest",
        "rpm": 50,
        "is_paid": True,
        "available": ANTHROPIC_API_KEY is not None,
        "skip_reason": None if ANTHROPIC_API_KEY else "ANTHROPIC_API_KEY not set",
        "input_per_1m_tokens_usd": 3.00,
        "output_per_1m_tokens_usd": 15.00,
    },
    "qwen": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "rpm": 20,
        "is_paid": False,
        "available": False,  # HF Inference Providers requires paid partner provider; disabled.
        "skip_reason": "HF Inference Providers won't serve Qwen2-VL on free tier (model_not_supported)",
        "input_per_1m_tokens_usd": 0.0,
        "output_per_1m_tokens_usd": 0.0,
    },
    "groq_scout": {
        "provider": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "rpm": 30,
        "is_paid": False,
        "available": GROQ_API_KEY is not None,
        "skip_reason": None if GROQ_API_KEY else "GROQ_API_KEY not set",
        # Groq free tier — cost reported as 0.0 USD.
        "input_per_1m_tokens_usd": 0.0,
        "output_per_1m_tokens_usd": 0.0,
    },
    "mistral": {
        # Open-weights Ministral 3 14B (model id `ministral-14b-2512`) served
        # via Mistral La Plateforme. Free Experiment plan: empirical rate
        # ≥ 40 RPM observed on this account (docs say 2 RPM, but the live
        # cap is much higher), 1 B tokens / month, no card required.
        # Replaces the retired Pixtral 12B (deprecated 2025-12-31); Ministral
        # 3 14B is the official successor Mistral docs point to.
        # Throttle conservatively at 30 RPM to leave headroom for TPM cap.
        "provider": "mistral",
        "model_id": "ministral-14b-2512",
        "rpm": 30,
        "is_paid": False,
        "available": MISTRAL_API_KEY is not None,
        "skip_reason": None if MISTRAL_API_KEY else "MISTRAL_API_KEY not set",
        "input_per_1m_tokens_usd": 0.0,
        "output_per_1m_tokens_usd": 0.0,
    },
}

DEFAULT_RUNS = 3
TEMPERATURE = 0.0
REQUEST_TIMEOUT_S = 120

# Tiling — slice panoramas are ~142k px wide; no VLM accepts that as one
# image. We tile each panorama into TILE_WIDTH-px chunks with TILE_OVERLAP
# overlap so headlines straddling a tile boundary are captured intact in
# at least one tile, then dedup across tiles at aggregation time.
TILE_WIDTH = 3000
TILE_OVERLAP = 1000
TILE_CACHE_DIR = VLM_DIR / "output" / "_tiles"


def available_vlms() -> list[str]:
    return [k for k, v in MODEL_REGISTRY.items() if v["available"]]


def deferred_vlms() -> dict[str, str]:
    return {
        k: v["skip_reason"]
        for k, v in MODEL_REGISTRY.items()
        if not v["available"]
    }
