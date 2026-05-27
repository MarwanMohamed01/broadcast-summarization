"""OpenAI GPT-4o-mini adapter (deferred until user tops up billing)."""

from __future__ import annotations

import time
from typing import Any

from .base import VLMAdapter, VLMResponse


class OpenAIAdapter(VLMAdapter):
    name = "gpt4o_mini"
    model_id = "gpt-4o-mini"

    def __init__(self, api_key: str, **kw: Any) -> None:
        super().__init__(api_key, **kw)
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, timeout=self.timeout_s)

    def _call(self, image_bytes: bytes, mime: str, prompt: str) -> VLMResponse:
        data_url = f"data:{mime};base64,{self._b64(image_bytes)}"
        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.model_id,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        wall = time.time() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return VLMResponse(
            raw_text=text,
            headlines=[],
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            wall_seconds=wall,
            metadata={"model_id": self.model_id},
        )
