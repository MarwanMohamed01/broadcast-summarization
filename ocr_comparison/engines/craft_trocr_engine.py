"""
CRAFT (EasyOCR detector) + TrOCR (Microsoft recognizer) hybrid.

NOTE: This is a recognizer-only comparison. TrOCR cannot detect text by
itself, so we borrow EasyOCR's CRAFT detector and feed the cropped word
boxes to TrOCR. Reported separately from the 5 end-to-end engines.

Model: microsoft/trocr-base-printed (printed text, English).
"""
import cv2
import numpy as np

from .base import EngineResult, Word
from ..config import SLICE_W, STRIDE

name = "craft_trocr"

_easyocr_reader = None
_trocr_processor = None
_trocr_model = None


def _get_detector():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def _get_recognizer():
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        model_id = "microsoft/trocr-base-printed"
        _trocr_processor = TrOCRProcessor.from_pretrained(model_id)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_id)
        _trocr_model.eval()
    return _trocr_processor, _trocr_model


def _recognize_crop(crop_bgr: np.ndarray) -> str:
    import torch
    processor, model = _get_recognizer()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pixel_values = processor(images=rgb, return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixel_values, max_length=64)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _ocr_slice(slice_bgr: np.ndarray) -> list[Word]:
    detector = _get_detector()
    # Upscale like EasyOCR engine does — CRAFT is more reliable on larger text
    h, w = slice_bgr.shape[:2]
    up = cv2.resize(slice_bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    # detail=1 returns boxes only (no recognition from EasyOCR — we only use CRAFT)
    # We still need the boxes; recognize via TrOCR.
    detections = detector.detect(up)
    # detections = (horizontal_boxes, free_boxes); we only need horizontal.
    boxes = detections[0][0] if detections and detections[0] else []

    words: list[Word] = []
    for box in boxes:
        # box is [x_min, x_max, y_min, y_max] in upscaled coords
        x_min, x_max, y_min, y_max = [int(v) for v in box]
        x_min = max(0, x_min)
        x_max = min(up.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(up.shape[0], y_max)
        if x_max - x_min < 10 or y_max - y_min < 5:
            continue
        crop = up[y_min:y_max, x_min:x_max]
        try:
            txt = _recognize_crop(crop)
        except Exception:
            continue
        if not txt:
            continue
        words.append(Word(
            x=int(x_min / 3),  # back to original slice coords
            text=txt,
            confidence=1.0,   # TrOCR generate() doesn't give a per-word score
        ))
    return words


def run(panorama: np.ndarray) -> EngineResult:
    _get_detector()
    _get_recognizer()
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
            "detector": "EasyOCR CRAFT",
            "recognizer": "microsoft/trocr-base-printed",
            "slice_w": SLICE_W,
            "stride": STRIDE,
            "slices_processed": slices_done,
            "note": "Hybrid recognizer-only pipeline, reported separately.",
        },
    )
