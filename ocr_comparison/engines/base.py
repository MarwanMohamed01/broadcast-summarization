"""
Base OCR engine interface.

Every engine adapter must implement a single function:

    run(panorama_img: np.ndarray) -> EngineResult

The runner in run_ocr_engine.py wraps this, measures timing + memory,
and persists the output. Keeping engines minimal makes it easy to add
new ones later.
"""
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np


@dataclass
class Word:
    """One recognized token with its left-x position."""
    x: int
    text: str
    confidence: float  # 0-1 normalized


@dataclass
class EngineResult:
    """What every engine must return."""
    full_text: str
    words: list[Word] = field(default_factory=list)
    # Free-form per-engine metadata (model names, settings used, etc.)
    meta: dict = field(default_factory=dict)


class OCREngine(Protocol):
    name: str

    def run(self, panorama: np.ndarray) -> EngineResult: ...


def words_to_text(words: list[Word]) -> str:
    """Concatenate words (in the order provided) into a single text string."""
    return " ".join(w.text for w in words if w.text)
