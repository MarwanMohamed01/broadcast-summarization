"""Mistral La Plateforme adapter — Ministral 3 14B (open-weights, free tier).

Replaces the retired Pixtral 12B; Ministral 3 14B is the official
successor Mistral docs point to and shares the same vision API surface.
Open-weights, vision-capable, served free on the Experiment plan
(2 RPM, 1 B tokens / month, no card).

OpenAI-compatible chat-completions schema, with images sent inline as
base64 data URLs in the `image_url` part — same content layout as the
existing Groq and HuggingFace adapters, so prompts are byte-identical.
"""

from __future__ import annotations

import time

import requests

from .base import VLMAdapter, VLMResponse


class MistralAdapter(VLMAdapter):
    name = "mistral"
    model_id = "ministral-14b-2512"
    endpoint = "https://api.mistral.ai/v1/chat/completions"

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
                        {"type": "image_url", "image_url": data_url},
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        t0 = time.time()
        try:
            r = requests.post(self.endpoint, headers=headers, json=payload,
                              timeout=self.timeout_s)
        except requests.exceptions.RequestException as exc:
            wall = time.time() - t0
            return VLMResponse(raw_text="", headlines=[], wall_seconds=wall,
                               error=f"network: {type(exc).__name__}: {exc}")
        wall = time.time() - t0
        if r.status_code == 429:
            return VLMResponse(raw_text="", headlines=[], wall_seconds=wall,
                               error=f"Mistral rate limited: {r.text[:300]}")
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
