"""VLMAdapter ABC + shared response container."""

from __future__ import annotations

import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    raw_text: str
    headlines: list[dict[str, str]]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    wall_seconds: float = 0.0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "headlines": self.headlines,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_seconds": self.wall_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }


class VLMAdapter(ABC):
    """Abstract VLM client.

    Subclasses implement `_call(image_bytes, mime, prompt)` and return
    a `VLMResponse`. The base class handles file reading, prompt
    plumbing, and JSON parsing of the model output.
    """

    name: str = "base"
    model_id: str = ""

    def __init__(self, api_key: str, *, temperature: float = 0.0,
                 timeout_s: int = 120):
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s

    # ---- public entry point ------------------------------------------------
    def extract_headlines(self, panorama_path: Path, prompt: str) -> VLMResponse:
        if not panorama_path.exists():
            return VLMResponse(raw_text="", headlines=[],
                               error=f"Panorama not found: {panorama_path}")
        try:
            image_bytes = panorama_path.read_bytes()
        except Exception as exc:
            return VLMResponse(raw_text="", headlines=[],
                               error=f"Failed to read panorama: {exc}")
        mime = "image/png" if panorama_path.suffix.lower() == ".png" else "image/jpeg"

        try:
            resp = self._call(image_bytes, mime, prompt)
        except Exception as exc:
            logger.exception("[%s] adapter call failed", self.name)
            return VLMResponse(raw_text="", headlines=[],
                               error=f"{type(exc).__name__}: {exc}")

        if resp.error:
            return resp
        resp.headlines = self._parse_headlines(resp.raw_text)
        return resp

    # ---- subclass hook -----------------------------------------------------
    @abstractmethod
    def _call(self, image_bytes: bytes, mime: str, prompt: str) -> VLMResponse:
        ...

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _b64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("ascii")

    @staticmethod
    def _parse_headlines(raw: str) -> list[dict[str, str]]:
        """Extract the headlines list from a model's raw response.

        Tolerant: strips ```json fences, falls back to first JSON object
        in the text, normalises text fields to upper-case strings.
        """
        if not raw:
            return []
        cleaned = raw.strip()
        # Strip code fences if present.
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        # Try direct JSON load first.
        obj = None
        try:
            obj = json.loads(cleaned)
        except Exception:
            # Fall back to the first {...} block in the text.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    obj = json.loads(match.group(0))
                except Exception:
                    obj = None
        if not isinstance(obj, dict):
            return []
        items = obj.get("headlines", []) or []
        out: list[dict[str, str]] = []
        for it in items:
            if isinstance(it, str):
                text = it.strip()
                conf = "medium"
            elif isinstance(it, dict):
                text = str(it.get("text", "")).strip()
                # New prompt drops the `confidence` field — default to "high"
                # since the model only emits headlines it is confident in.
                conf = str(it.get("confidence", "high")).strip().lower()
                if conf not in {"high", "medium", "low"}:
                    conf = "high"
            else:
                continue
            if not text:
                continue
            out.append({"text": text.upper(), "confidence": conf})
        return out
