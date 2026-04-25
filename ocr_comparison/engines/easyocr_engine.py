"""
EasyOCR adapter (pure — no Tesseract dash augmentation).

This gives the honest head-to-head number for EasyOCR by itself.
v6's production pipeline adds Tesseract dash-detection on top; that is
a v6 engineering choice, not part of the raw OCR comparison.
"""
import cv2
import numpy as np

from .base import EngineResult, Word
from ..config import SLICE_W, STRIDE, CONF_THRESHOLDS

name = "easyocr"

_reader = None  # lazy-loaded EasyOCR Reader


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _preprocess(img: np.ndarray, scale: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def _ocr_slice(img: np.ndarray) -> list[Word]:
    reader = _get_reader()
    processed = _preprocess(img)
    results = reader.readtext(processed)
    thr = CONF_THRESHOLDS["easyocr"]
    words: list[Word] = []
    for (bbox, txt, conf) in results:
        if conf > thr and txt.strip():
            # bbox is 4 corners; use left x; divide by scale to match original
            x = int(bbox[0][0] / 3)
            words.append(Word(x=x, text=txt.strip(), confidence=float(conf)))
    return words


def run(panorama: np.ndarray) -> EngineResult:
    _get_reader()  # load once up front
    h, w = panorama.shape[:2]
    margin = STRIDE // 2
    all_words: list[Word] = []
    slices_done = 0

    for x in range(0, w, STRIDE):
        x_end = min(x + SLICE_W, w)
        if x_end - x < 500:
            break
        slice_img = panorama[:, x:x_end]
        words = _ocr_slice(slice_img)

        slice_w = x_end - x
        if x == 0:
            keep = [wd for wd in words if wd.x < slice_w - margin]
        elif x + SLICE_W >= w:
            keep = [wd for wd in words if wd.x >= margin]
        else:
            keep = [wd for wd in words if margin <= wd.x < slice_w - margin]

        for wd in keep:
            wd.x += x
        all_words.extend(keep)
        slices_done += 1

    all_words.sort(key=lambda wd: wd.x)
    full_text = " ".join(wd.text for wd in all_words)

    return EngineResult(
        full_text=full_text,
        words=all_words,
        meta={
            "engine": name,
            "slice_w": SLICE_W,
            "stride": STRIDE,
            "slices_processed": slices_done,
            "upscale": 3,
            "confidence_threshold": CONF_THRESHOLDS["easyocr"],
        },
    )
