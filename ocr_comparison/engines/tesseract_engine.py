"""
Tesseract OCR adapter.

Mirrors v6's slicing approach: 3000px slices, 1500px stride, center-region
word filtering. No dash augmentation — this is a pure-Tesseract run.
"""
import cv2
import numpy as np
import pytesseract

from .base import EngineResult, Word
from ..config import SLICE_W, STRIDE, TESSERACT_CMD, CONF_THRESHOLDS

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

name = "tesseract"


def _preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _ocr_slice(img: np.ndarray) -> list[Word]:
    processed = _preprocess(img)
    data = pytesseract.image_to_data(
        processed,
        config="--psm 7",
        output_type=pytesseract.Output.DICT,
    )
    thr = CONF_THRESHOLDS["tesseract"]
    words: list[Word] = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        conf = int(data["conf"][i])
        x = int(data["left"][i])
        if txt and conf > thr:
            words.append(Word(x=x, text=txt, confidence=conf / 100.0))
    return words


def run(panorama: np.ndarray) -> EngineResult:
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

        # Shift x positions into panorama coords
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
            "psm": 7,
            "confidence_threshold": CONF_THRESHOLDS["tesseract"],
        },
    )
