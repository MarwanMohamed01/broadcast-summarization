"""
Step 2: Detect pixel-level scroll offset between consecutive ticker frames.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config


def _to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def measure_scroll(frame_a: np.ndarray, frame_b: np.ndarray, max_shift: int) -> tuple[int, float]:
    """
    Measure how many pixels frame_b is shifted left relative to frame_a.
    Returns (shift_pixels, confidence).
    """
    gray_a = _to_gray(frame_a)
    gray_b = _to_gray(frame_b)

    h, w = gray_a.shape

    template_w = int(w * config.SCROLL_TEMPLATE_WIDTH_PCT)
    t_x1 = w - template_w - max_shift
    t_x2 = w - max_shift
    if t_x1 < 0:
        t_x1 = 0
        t_x2 = min(template_w, w - max_shift)

    template = gray_a[:, t_x1:t_x2]

    if template.shape[1] < 10:
        return 0, 0.0

    search_x1 = max(0, t_x1 - max_shift)
    search_x2 = min(w, t_x2)
    search_region = gray_b[:, search_x1:search_x2]

    if search_region.shape[1] <= template.shape[1]:
        return 0, 0.0

    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    expected_x = t_x1 - search_x1
    actual_x = max_loc[0]
    shift = expected_x - actual_x

    return shift, max_val


def detect_all_scrolls(frame_paths: list[Path]) -> tuple[list[dict], dict]:
    """
    Measure scroll offset between all consecutive frame pairs.
    Returns (list of scroll data dicts, stats dict).
    """
    if len(frame_paths) < 2:
        print("Need at least 2 frames for scroll detection.")
        return [], {}

    results = []
    results.append({
        "path": frame_paths[0],
        "scroll_offset": 0,
        "cumulative_offset": 0,
        "confidence": 1.0,
    })

    cumulative = 0
    prev_img = cv2.imread(str(frame_paths[0]))
    low_conf_count = 0
    zero_shift_count = 0
    all_confidences = []

    print(f"Measuring scroll across {len(frame_paths)} frames...")

    for i in tqdm(range(1, len(frame_paths)), desc="Scroll detection"):
        curr_img = cv2.imread(str(frame_paths[i]))
        if curr_img is None:
            continue

        shift, conf = measure_scroll(prev_img, curr_img, config.SCROLL_MAX_SHIFT)
        all_confidences.append(conf)

        if conf < 0.3:
            shift = 0
            low_conf_count += 1

        if shift < config.SCROLL_MIN_SHIFT:
            shift = 0

        if shift == 0:
            zero_shift_count += 1

        cumulative += shift
        results.append({
            "path": frame_paths[i],
            "scroll_offset": shift,
            "cumulative_offset": cumulative,
            "confidence": conf,
        })

        prev_img = curr_img

    shifts = [r["scroll_offset"] for r in results if r["scroll_offset"] > 0]
    if shifts:
        avg_shift = np.mean(shifts)
        print(f"  Avg scroll per sample: {avg_shift:.1f}px")
        print(f"  Total scroll: {cumulative}px")
        print(f"  Low confidence frames (skipped): {low_conf_count}")
    else:
        avg_shift = 0
        print("  WARNING: No scroll detected!")

    stats = {
        "scroll_detection": {
            "frames_processed": len(results),
            "total_scroll_px": cumulative,
            "avg_scroll_per_frame_px": round(float(avg_shift), 1) if shifts else 0,
            "max_scroll_per_frame_px": max(shifts) if shifts else 0,
            "min_scroll_per_frame_px": min(shifts) if shifts else 0,
            "frames_with_movement": len(shifts),
            "frames_with_zero_shift": zero_shift_count,
            "low_confidence_frames": low_conf_count,
            "low_confidence_pct": round(low_conf_count / max(len(results) - 1, 1) * 100, 1),
            "avg_confidence": round(float(np.mean(all_confidences)), 3) if all_confidences else 0,
            "min_confidence": round(float(np.min(all_confidences)), 3) if all_confidences else 0,
        },
    }

    return results, stats


if __name__ == "__main__":
    frames = sorted(config.FRAMES_DIR.glob("ticker_*.png"))
    if frames:
        results, stats = detect_all_scrolls(frames)
        import json
        print(json.dumps(stats, indent=2))
    else:
        print("No ticker frames found. Run step1 first.")
