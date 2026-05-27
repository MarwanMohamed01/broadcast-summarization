"""Groq vision adapter (Llama 4 Scout 17B, OpenAI-compatible API).

Free tier — same key the llm_summarization branch already uses for
text-only inference. Llama 4 Scout supports image inputs natively.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .base import VLMAdapter, VLMResponse


class GroqAdapter(VLMAdapter):
    name = "groq_scout"
    model_id = "meta-llama/llama-4-scout-17b-16e-instruct"
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def _call(self, image_bytes: bytes, mime: str, prompt: str) -> VLMResponse:
        data_url = f"data:{mime};base64,{self._b64(image_bytes)}"
        payload = {
            "model": self.model_id,
            "temperature": self.temperature,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        t0 = time.time()
        r = requests.post(self.endpoint, headers=headers, json=payload,
                          timeout=self.timeout_s)
        wall = time.time() - t0
        if r.status_code == 429:
            return VLMResponse(raw_text="", headlines=[], wall_seconds=wall,
                               error=f"Groq rate limited: {r.text[:300]}")
        if r.status_code != 200:
            return VLMResponse(raw_text="", headlines=[], wall_seconds=wall,
                               error=f"HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except Exception:
            text = ""
        usage = body.get("usage") or {}
        return VLMResponse(
            raw_text=text,
            headlines=[],
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            wall_seconds=wall,
            metadata={"model_id": self.model_id},
        )
