"""
PaddleOCR adapter.

PaddleOCR has its own detector (DB++) and recognizer (SVTR/PP-OCRv4).
It can handle wider images than Tesseract, but we still slice with the
same 3000px window as the others for apples-to-apples comparison.
"""
import cv2
import numpy as np

from .base import EngineResult, Word
from ..config import SLICE_W, STRIDE, CONF_THRESHOLDS

name = "paddle"

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        # PaddleOCR v3+: use_textline_orientation replaces use_angle_cls.
        # Ticker text is horizontal so we disable orientation detection.
        # enable_mkldnn=False avoids a known PaddlePaddle 3.x CPU runtime bug:
        # "ConvertPirAttribute2RuntimeAttribute not support ArrayAttribute<DoubleAttribute>"
        # which triggers when OneDNN tries to execute the PP-OCRv5 detector graph.
        _ocr = PaddleOCR(
            use_textline_orientation=False,
            lang="en",
            enable_mkldnn=False,
        )
    return _ocr


def _ocr_slice(img: np.ndarray) -> list[Word]:
    ocr = _get_ocr()
    result = ocr.predict(img)
    thr = CONF_THRESHOLDS["paddle"]
    words: list[Word] = []
    # PaddleOCR v3 returns a list of result dicts; each has
    # "rec_texts", "rec_scores", "rec_boxes" aligned by index.
    if not result:
        return words
    page = result[0]
    texts = page.get("rec_texts")
    scores = page.get("rec_scores")
    boxes = page.get("rec_boxes")
    if texts is None or scores is None or boxes is None:
        return words
    for txt, conf, box in zip(texts, scores, boxes):
        if conf > thr and txt and txt.strip():
            # box is a numpy array [x1,y1,x2,y2] (axis-aligned)
            x = int(box[0])
            words.append(Word(x=x, text=txt.strip(), confidence=float(conf)))
    return words


def run(panorama: np.ndarray) -> EngineResult:
    _get_ocr()  # warm up
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
            "model": "PP-OCRv4 (en)",
            "confidence_threshold": CONF_THRESHOLDS["paddle"],
        },
    )
