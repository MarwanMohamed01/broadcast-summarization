"""Gemini 2.5 Flash adapter — REST API for full thinking-budget control.

We use the REST endpoint instead of `google.generativeai` because the
SDK doesn't reliably expose `thinking_config` / `thinking_budget`
across versions. Thinking is DISABLED here (`thinking_budget = 0`)
to keep cost predictable. Thinking tokens are billed at $3.50/M
(non-thinking output is $2.50/M); leaving thinking on can 1.5-2× the
real cost without showing up in `usage_metadata.candidates_token_count`.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .base import VLMAdapter, VLMResponse


class GeminiAdapter(VLMAdapter):
    name = "gemini"
    model_id = "gemini-2.5-flash"
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
    )

    def _call(self, image_bytes: bytes, mime: str, prompt: str) -> VLMResponse:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": mime,
                                          "data": self._b64(image_bytes)}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                # Disable thinking — saves real billing dollars even though
                # thinking tokens don't appear in candidates_token_count.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        url = f"{self.endpoint}?key={self.api_key}"
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        wall = time.time() - t0
        if r.status_code != 200:
            return VLMResponse(raw_text="", headlines=[], wall_seconds=wall,
                               error=f"HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        try:
            cand = body["candidates"][0]
            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts
                            if not p.get("thought"))
        except Exception:
            text = ""
        usage = body.get("usageMetadata", {}) or {}
        return VLMResponse(
            raw_text=text,
            headlines=[],
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            wall_seconds=wall,
            metadata={
                "model_id": self.model_id,
                "thoughts_token_count": usage.get("thoughtsTokenCount", 0),
                "total_token_count": usage.get("totalTokenCount"),
            },
        )
