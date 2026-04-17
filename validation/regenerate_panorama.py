"""
Regenerate the panorama image for a specific frame range of the video.

v6's panorama images were cleaned up after OCR to save disk space. This
script re-runs steps 1-3 of v6 (frame extraction, scroll detection,
panorama stitching) for a chosen slice of the video and saves the
panorama as a PNG, so it can be used as ground-truth annotation
reference for validation.

Usage:
    python validation/regenerate_panorama.py --chunk 17
    python validation/regenerate_panorama.py --chunk 26

The chunk index corresponds to the 30-minute chunk used in v6's chunked
mode. Chunk 0 = minutes 0-30, chunk 17 = minutes 510-540 (hours 8:30-9:00),
chunk 26 = minutes 780-810 (hours 13:00-13:30).
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
V6_DIR = PROJECT_DIR / "ticker_extraction_v6"
VIDEOS_DIR = PROJECT_DIR / "videos"
GT_DIR = PROJECT_DIR / "validation" / "ground_truth"

# Add v6 to path so we can import its modules
sys.path.insert(0, str(V6_DIR))

import cv2  # noqa: E402
import config as v6_config  # noqa: E402
from step1_extract_ticker import extract_ticker_frames  # noqa: E402
from step2_scroll_detection import detect_all_scrolls  # noqa: E402
from step3_stitch_image import stitch_panorama  # noqa: E402

# Chunk size used by v6 chunked mode (30 minutes)
CHUNK_MINUTES = 30


def find_video() -> Path:
    """Find the 14-hour AlJazeera video in videos/."""
    candidates = list(VIDEOS_DIR.glob("*.mp4"))
    for c in candidates:
        if "14hrs" in c.name or "14hr" in c.name:
            return c
    # fallback: prefer any AlJazeera-named file
    for c in candidates:
        if "Aljazeera" in c.name or "AlJazeera" in c.name:
            return c
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No .mp4 files in {VIDEOS_DIR}")


def regenerate_panorama(chunk_idx: int, slice_letter: str) -> None:
    """Run v6 steps 1-3 on the frame range for the given chunk."""
    video_path = find_video()
    print(f"Video: {video_path.name}")

    # Determine frame range
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_sec = total_frames / fps if fps > 0 else 0
    print(f"  Duration: {duration_sec/3600:.2f} hours ({total_frames} frames at {fps:.0f} fps)")

    chunk_frames = int(CHUNK_MINUTES * 60 * fps)
    start_frame = chunk_idx * chunk_frames
    end_frame = min((chunk_idx + 1) * chunk_frames, total_frames)
    start_sec = start_frame / fps
    end_sec = end_frame / fps

    print(f"  Chunk {chunk_idx}: frames {start_frame}-{end_frame} "
          f"({start_sec/60:.0f}min - {end_sec/60:.0f}min)")

    # Working directory for this slice
    work_dir = GT_DIR / f"_work_slice_{slice_letter}"
    frames_dir = work_dir / "ticker_frames"
    panorama_dir = work_dir / "panorama"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # Patch v6 config to point to our working dir
    orig_panorama_dir = v6_config.PANORAMA_DIR
    v6_config.PANORAMA_DIR = panorama_dir

    try:
        # Step 1
        print("\n[Step 1/3] Extracting ticker frames...")
        t0 = time.time()
        frame_paths, _ = extract_ticker_frames(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            output_dir=frames_dir,
        )
        print(f"  Done in {time.time()-t0:.0f}s ({len(frame_paths)} frames)")
        if not frame_paths:
            print("ERROR: no frames extracted")
            return

        # Step 2
        print("\n[Step 2/3] Detecting scroll offsets...")
        t0 = time.time()
        scroll_data, _ = detect_all_scrolls(frame_paths)
        print(f"  Done in {time.time()-t0:.0f}s")
        if not scroll_data:
            print("ERROR: scroll detection failed")
            return

        # Step 3
        print("\n[Step 3/3] Stitching panorama...")
        t0 = time.time()
        panorama_paths, _ = stitch_panorama(scroll_data)
        print(f"  Done in {time.time()-t0:.0f}s ({len(panorama_paths)} chunks)")
    finally:
        v6_config.PANORAMA_DIR = orig_panorama_dir

    if not panorama_paths:
        print("ERROR: stitching failed")
        return

    # Copy the first (and usually only) panorama chunk to ground_truth/
    GT_DIR.mkdir(parents=True, exist_ok=True)
    output_png = GT_DIR / f"slice_{slice_letter}_panorama.png"

    if len(panorama_paths) == 1:
        shutil.copy(panorama_paths[0], output_png)
    else:
        # Multiple panorama chunks — concatenate horizontally for annotation
        print(f"  Concatenating {len(panorama_paths)} panorama chunks...")
        import numpy as np
        imgs = [cv2.imread(str(p)) for p in panorama_paths]
        max_h = max(i.shape[0] for i in imgs)
        total_w = sum(i.shape[1] for i in imgs)
        combined = np.zeros((max_h, total_w, 3), dtype=np.uint8)
        x_off = 0
        for img in imgs:
            h, w = img.shape[:2]
            combined[:h, x_off:x_off + w] = img
            x_off += w
        cv2.imwrite(str(output_png), combined)

    # Clean up working dir (frames are huge)
    shutil.rmtree(work_dir)

    size_mb = output_png.stat().st_size / 1e6
    print(f"\n[DONE] Panorama saved: {output_png}")
    print(f"       Size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Regenerate v6 panorama for a chunk")
    parser.add_argument("--chunk", type=int, required=True, help="Chunk index (0-29 for 14h video)")
    parser.add_argument("--slice", type=str, default=None,
                        help="Slice letter for output filename (default: derive from chunk)")
    args = parser.parse_args()

    slice_letter = args.slice or chr(ord("A") + args.chunk % 26)
    regenerate_panorama(args.chunk, slice_letter)


if __name__ == "__main__":
    main()
