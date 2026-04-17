"""
Step 3: Stitch ticker frames into panoramic image strips.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config


def stitch_panorama(scroll_data: list[dict]) -> tuple[list[Path], dict]:
    """
    Stitch ticker frames into panoramic strips using scroll offsets.
    Returns (list of panorama image paths, stats dict).
    """
    config.PANORAMA_DIR.mkdir(parents=True, exist_ok=True)

    if not scroll_data:
        print("No scroll data to stitch.")
        return [], {}

    first_img = cv2.imread(str(scroll_data[0]["path"]))
    if first_img is None:
        print(f"Cannot read first frame: {scroll_data[0]['path']}")
        return [], {}

    frame_h, frame_w = first_img.shape[:2]
    total_scroll = scroll_data[-1]["cumulative_offset"]
    total_panorama_w = frame_w + total_scroll

    print(f"Ticker frame size: {frame_w}x{frame_h}")
    print(f"Total scroll: {total_scroll}px")
    print(f"Estimated panorama width: {total_panorama_w}px")

    chunk_width = config.PANORAMA_MAX_WIDTH
    panorama_paths = []
    chunk_stats = []

    chunk_start_offset = 0
    chunk_idx = 0

    while chunk_start_offset < total_scroll:
        chunk_end_offset = min(chunk_start_offset + chunk_width, total_scroll + frame_w)
        pano_w = int(chunk_end_offset - chunk_start_offset)

        if pano_w <= 0:
            break

        panorama = np.zeros((frame_h, pano_w, 3), dtype=np.uint8)
        filled = np.zeros(pano_w, dtype=bool)

        chunk_frame_count = 0
        for entry in tqdm(scroll_data, desc=f"Stitching chunk {chunk_idx}", leave=False):
            cum = entry["cumulative_offset"]
            frame_global_start = cum
            frame_global_end = cum + frame_w

            local_start = int(frame_global_start - chunk_start_offset)
            local_end = int(frame_global_end - chunk_start_offset)

            if local_end <= 0 or local_start >= pano_w:
                continue

            img = cv2.imread(str(entry["path"]))
            if img is None:
                continue

            chunk_frame_count += 1

            src_x1 = max(0, -local_start)
            dst_x1 = max(0, local_start)
            src_x2 = min(frame_w, pano_w - local_start)
            dst_x2 = min(pano_w, local_end)

            if dst_x2 <= dst_x1:
                continue

            for x in range(dst_x1, dst_x2):
                if not filled[x]:
                    src_x = x - local_start
                    if 0 <= src_x < frame_w:
                        panorama[:, x] = img[:, src_x]
                        filled[x] = True

        filled_count = int(filled.sum())
        fill_pct = round(filled_count / pano_w * 100, 1)

        out_path = config.PANORAMA_DIR / f"panorama_chunk_{chunk_idx:03d}.png"
        cv2.imwrite(str(out_path), panorama)
        panorama_paths.append(out_path)
        print(f"  Chunk {chunk_idx}: {pano_w}x{frame_h}px, {chunk_frame_count} frames, {filled_count}/{pano_w} cols filled ({fill_pct}%)")

        chunk_stats.append({
            "chunk_index": chunk_idx,
            "width_px": pano_w,
            "height_px": frame_h,
            "frames_used": chunk_frame_count,
            "columns_filled": filled_count,
            "columns_total": pano_w,
            "fill_rate_pct": fill_pct,
        })

        chunk_idx += 1
        chunk_start_offset += chunk_width - config.PANORAMA_OVERLAP_FOR_OCR
        if chunk_end_offset >= total_scroll + frame_w:
            break

    print(f"Created {len(panorama_paths)} panorama chunk(s)")

    total_filled = sum(c["columns_filled"] for c in chunk_stats)
    total_cols = sum(c["columns_total"] for c in chunk_stats)

    stats = {
        "stitching": {
            "total_panorama_width_px": total_panorama_w,
            "total_panorama_height_px": frame_h,
            "num_chunks": len(panorama_paths),
            "chunk_max_width_px": chunk_width,
            "total_columns_filled": total_filled,
            "total_columns": total_cols,
            "overall_fill_rate_pct": round(total_filled / max(total_cols, 1) * 100, 1),
            "chunks": chunk_stats,
        },
    }

    return panorama_paths, stats


if __name__ == "__main__":
    print("Run this through main.py (needs scroll data from step 2)")
