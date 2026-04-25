"""
docTR (Mindee) adapter — PyTorch backend, DB detector + CRNN recognizer.

docTR expects RGB images (OpenCV gives BGR). We convert before calling.
"""
import cv2
import numpy as np

from .base import EngineResult, Word
from ..config import SLICE_W, STRIDE, CONF_THRESHOLDS

name = "doctr"

_model = None


def _get_model():
    global _model
    if _model is None:
        from doctr.models import ocr_predictor
        # pretrained=True downloads on first use.
        _model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
    return _model


def _ocr_slice(img_bgr: np.ndarray) -> list[Word]:
    model = _get_model()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    result = model([img_rgb])
    thr = CONF_THRESHOLDS["doctr"]
    words: list[Word] = []
    # result.pages[0].blocks[*].lines[*].words[*]
    page = result.pages[0]
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                conf = float(word.confidence)
                if conf <= thr:
                    continue
                txt = (word.value or "").strip()
                if not txt:
                    continue
                # geometry is ((x0,y0),(x1,y1)) in relative coords 0-1
                (x0, _), (_, _) = word.geometry
                x_px = int(x0 * w)
                words.append(Word(x=x_px, text=txt, confidence=conf))
    return words


def run(panorama: np.ndarray) -> EngineResult:
    _get_model()
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
            "det_arch": "db_resnet50",
            "reco_arch": "crnn_vgg16_bn",
            "confidence_threshold": CONF_THRESHOLDS["doctr"],
        },
    )
