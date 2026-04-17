"""
Step 4: OCR the panoramic ticker strips.
"""

import cv2
import numpy as np
from pathlib import Path
import config


def _preprocess_for_tesseract(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _preprocess_for_easyocr(img: np.ndarray, scale: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def _ocr_slice_with_positions(img: np.ndarray, engine: str, reader=None) -> list[tuple]:
    if engine == "tesseract":
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

        processed = _preprocess_for_tesseract(img)
        data = pytesseract.image_to_data(
            processed,
            config="--psm 7",
            output_type=pytesseract.Output.DICT
        )

        words = []
        for i in range(len(data["text"])):
            txt = data["text"][i].strip()
            conf = int(data["conf"][i])
            word_x = int(data["left"][i])
            word_w = int(data["width"][i])
            if txt and conf > 20:
                words.append((word_x, word_w, txt, conf))
        return words

    elif engine == "easyocr":
        processed = _preprocess_for_easyocr(img)
        results = reader.readtext(processed)
        words = []
        for (bbox, txt, conf) in results:
            if conf > 0.3 and txt.strip():
                x_pos = int(bbox[0][0] / 3)
                w = int((bbox[1][0] - bbox[0][0]) / 3)
                words.append((x_pos, w, txt.strip(), conf))
        return words

    else:
        raise ValueError(f"Unknown OCR engine: {engine}")


def _detect_dashes_tesseract(img: np.ndarray) -> list[int]:
    """
    Use Tesseract PSM 6 to find x-positions of " - " delimiters in a slice.
    Returns list of x-pixel positions where dashes were detected.
    """
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

        processed = _preprocess_for_tesseract(img)
        data = pytesseract.image_to_data(
            processed,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )

        dash_positions = []
        for i in range(len(data["text"])):
            txt = data["text"][i].strip()
            if txt in ("-", "—", "–", "~"):
                x_pos = int(data["left"][i])
                dash_positions.append(x_pos)
        return dash_positions
    except Exception:
        return []


def ocr_panorama_chunks(panorama_paths: list[Path]) -> tuple[str, dict]:
    """
    Run center-only OCR on panorama chunks.

    Uses EasyOCR for text recognition (better accuracy) and Tesseract
    as a secondary pass to detect " - " delimiters between headlines
    (which EasyOCR misses due to their small size).

    Returns (full OCR text, stats dict).
    """
    config.OCR_DIR.mkdir(parents=True, exist_ok=True)

    engine = config.OCR_ENGINE
    print(f"OCR engine: {engine}")

    reader = None
    if engine == "easyocr":
        import easyocr
        print("  Loading EasyOCR model...")
        reader = easyocr.Reader(["en"], gpu=False)

    SLICE_W = 3000
    STRIDE = SLICE_W // 2

    all_text_parts = []
    total_slices = 0
    total_words = 0
    total_dashes_detected = 0
    all_word_confidences = []

    for pano_path in panorama_paths:
        print(f"  OCR on {pano_path.name}...")
        img = cv2.imread(str(pano_path))
        if img is None:
            print(f"    Cannot read {pano_path}")
            continue

        h, w = img.shape[:2]
        num_slices = max(1, (w - STRIDE) // STRIDE + 1)
        print(f"    Size: {w}x{h}, ~{num_slices} slices")

        margin = STRIDE // 2
        chunk_texts = []
        slice_idx = 0

        for x in range(0, w, STRIDE):
            x_end = min(x + SLICE_W, w)
            if x_end - x < 500:
                break

            slice_img = img[:, x:x_end]

            # Primary OCR: get words with positions
            words = _ocr_slice_with_positions(slice_img, engine, reader)

            # Secondary: detect dash positions with Tesseract
            dash_positions = _detect_dashes_tesseract(slice_img)

            # Scale dash positions if easyocr was used (it scales 3x)
            if engine == "easyocr":
                # EasyOCR word positions are already divided by 3 in
                # _ocr_slice_with_positions, so dash positions need same
                dash_positions = [d for d in dash_positions]
                # Tesseract runs on original scale, EasyOCR positions are /3
                # No adjustment needed — both are in original pixel coords

            # Filter to center region only
            slice_actual_w = x_end - x
            if x == 0:
                center_words = [wd for wd in words if wd[0] < slice_actual_w - margin]
                center_dashes = [d for d in dash_positions if d < slice_actual_w - margin]
            elif x + SLICE_W >= w:
                center_words = [wd for wd in words if wd[0] >= margin]
                center_dashes = [d for d in dash_positions if d >= margin]
            else:
                center_words = [wd for wd in words
                                if margin <= wd[0] < slice_actual_w - margin]
                center_dashes = [d for d in dash_positions
                                 if margin <= d < slice_actual_w - margin]

            total_dashes_detected += len(center_dashes)

            # Collect confidence values
            for wd in center_words:
                all_word_confidences.append(wd[3])

            total_words += len(center_words)

            # Insert " - " delimiters into the word list at the right positions
            # Each word has (x_pos, width, text, confidence)
            # Insert a synthetic dash word at each detected dash position
            all_elements = list(center_words)
            for d_pos in center_dashes:
                all_elements.append((d_pos, 10, "-", 1.0))

            # Sort all elements by x position
            all_elements.sort(key=lambda e: e[0])

            text = " ".join(e[2] for e in all_elements)
            if text:
                chunk_texts.append(text)

            slice_idx += 1
            if slice_idx % 10 == 0:
                print(f"    {slice_idx}/{num_slices} slices...")

        total_slices += slice_idx
        chunk_text = " ".join(chunk_texts)
        all_text_parts.append(chunk_text)

        txt_path = config.OCR_DIR / f"{pano_path.stem}_ocr.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(chunk_text)

    full_text = " ".join(all_text_parts)

    combined_path = config.OCR_DIR / "full_ocr_text.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"  Total OCR text length: {len(full_text)} chars")
    print(f"  Dashes detected (Tesseract): {total_dashes_detected}")

    # Compute OCR quality metrics
    alpha_chars = sum(1 for c in full_text if c.isalpha())
    upper_chars = sum(1 for c in full_text if c.isupper())
    digit_chars = sum(1 for c in full_text if c.isdigit())
    space_chars = sum(1 for c in full_text if c.isspace())
    other_chars = len(full_text) - alpha_chars - digit_chars - space_chars

    stats = {
        "ocr": {
            "engine": engine,
            "slice_width_px": SLICE_W,
            "slice_stride_px": STRIDE,
            "total_slices_processed": total_slices,
            "total_words_extracted": total_words,
            "total_dashes_detected": total_dashes_detected,
            "total_text_length_chars": len(full_text),
            "avg_chars_per_slice": round(len(full_text) / max(total_slices, 1), 1),
            "character_breakdown": {
                "alphabetic": alpha_chars,
                "uppercase": upper_chars,
                "digits": digit_chars,
                "spaces": space_chars,
                "other_symbols": other_chars,
                "uppercase_ratio": round(upper_chars / max(alpha_chars, 1) * 100, 1),
            },
            "word_confidence": {
                "avg": round(float(np.mean(all_word_confidences)) * 100, 1) if all_word_confidences else 0,
                "min": round(float(np.min(all_word_confidences)) * 100, 1) if all_word_confidences else 0,
                "max": round(float(np.max(all_word_confidences)) * 100, 1) if all_word_confidences else 0,
                "median": round(float(np.median(all_word_confidences)) * 100, 1) if all_word_confidences else 0,
                "total_words_scored": len(all_word_confidences),
            },
        },
    }

    return full_text, stats


if __name__ == "__main__":
    chunks = sorted(config.PANORAMA_DIR.glob("panorama_chunk_*.png"))
    if chunks:
        text, stats = ocr_panorama_chunks(chunks)
        import json
        print(json.dumps(stats, indent=2))
        print(f"\nFirst 500 chars:\n{text[:500]}")
    else:
        print("No panorama chunks found. Run step3 first.")
