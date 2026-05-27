"""Anthropic Claude 3.5 Sonnet adapter (deferred until user provides key)."""

from __future__ import annotations

import time
from typing import Any

from .base import VLMAdapter, VLMResponse


class AnthropicAdapter(VLMAdapter):
    name = "claude"
    model_id = "claude-3-5-sonnet-latest"

    def __init__(self, api_key: str, **kw: Any) -> None:
        super().__init__(api_key, **kw)
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout_s)

    def _call(self, image_bytes: bytes, mime: str, prompt: str) -> VLMResponse:
        t0 = time.time()
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=4096,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": self._b64(image_bytes),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        wall = time.time() - t0
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        return VLMResponse(
            raw_text=text,
            headlines=[],
            input_tokens=getattr(resp.usage, "input_tokens", None),
            output_tokens=getattr(resp.usage, "output_tokens", None),
            wall_seconds=wall,
            metadata={"model_id": self.model_id, "stop_reason": resp.stop_reason},
        )
