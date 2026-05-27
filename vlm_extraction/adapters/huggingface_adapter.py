"""HuggingFace Inference adapter for Qwen2-VL 7B."""

from __future__ import annotations

import time
from typing import Any

import requests

from .base import VLMAdapter, VLMResponse


class HuggingFaceAdapter(VLMAdapter):
    """Calls HF Inference Providers (router) for a vision-language model.

    The HF provider router accepts an OpenAI-compatible chat completion
    payload, so we use the same data-URL image format as the OpenAI
    adapter. If the model is cold-loading the call returns 503 with a
    `estimated_time` field and we surface that as an error rather than
    blocking — the runner skips the model gracefully.
    """

    name = "qwen"
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    endpoint = "https://router.huggingface.co/v1/chat/completions"

    def _call(self, image_bytes: bytes, mime: str, prompt: str) -> VLMResponse:
        data_url = f"data:{mime};base64,{self._b64(image_bytes)}"
        payload = {
            "model": self.model_id,
            "temperature": self.temperature,
            "max_tokens": 4096,
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
        if r.status_code == 503:
            return VLMResponse(raw_text="", headlines=[], wall_seconds=wall,
                               error=f"HF model cold-loading: {r.text[:200]}")
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
