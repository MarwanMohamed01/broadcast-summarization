"""
Step 1: Extract and crop the ticker region from video frames.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config


def extract_ticker_frames(video_path: Path,
                          start_frame: int = 0,
                          end_frame: int = 0,
                          output_dir: Path = None) -> tuple[list[Path], dict]:
    """
    Extract ticker region from sampled video frames.

    Args:
        video_path: path to input video.
        start_frame: first frame to process (0 = beginning).
        end_frame: last frame to process (0 = end of video).
        output_dir: override for FRAMES_DIR (used by chunked mode).

    Returns (list of saved ticker frame paths, stats dict).
    """
    frames_dir = output_dir if output_dir else config.FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Apply frame range
    if end_frame <= 0:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)
    start_frame = max(0, start_frame)
    range_frames = end_frame - start_frame

    y1 = int(frame_h * config.TICKER_Y_START_PCT)
    y2 = int(frame_h * config.TICKER_Y_END_PCT)
    x1 = int(frame_w * config.TICKER_X_START_PCT)
    x2 = int(frame_w * config.TICKER_X_END_PCT)
    ticker_w = x2 - x1
    ticker_h = y2 - y1

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {frame_w}x{frame_h}, FPS: {fps:.1f}, Total frames: {total_frames}")
    print(f"  Duration: {duration_sec:.1f}s")
    if start_frame > 0 or end_frame < total_frames:
        chunk_start_sec = start_frame / fps if fps > 0 else 0
        chunk_end_sec = end_frame / fps if fps > 0 else 0
        print(f"  Frame range: {start_frame}-{end_frame} ({chunk_start_sec:.0f}s - {chunk_end_sec:.0f}s)")
    print(f"  Ticker crop: y=[{y1}:{y2}], x=[{x1}:{x2}] -> {ticker_w}x{ticker_h}px")
    print(f"  Sampling every {config.FRAME_SAMPLE_RATE} frames ({fps/config.FRAME_SAMPLE_RATE:.1f} effective fps)")

    # Seek to start_frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved_paths = []
    frame_idx = start_frame
    frames_sampled = 0
    frames_skipped_black = 0
    pbar = tqdm(total=range_frames, desc="Extracting ticker frames")

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % config.FRAME_SAMPLE_RATE == 0:
            frames_sampled += 1
            ticker_crop = frame[y1:y2, x1:x2]
            mean_brightness = np.mean(ticker_crop)
            if mean_brightness > 20:
                fname = f"ticker_{frame_idx:06d}.png"
                out_path = frames_dir / fname
                cv2.imwrite(str(out_path), ticker_crop)
                saved_paths.append(out_path)
            else:
                frames_skipped_black += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"  Saved {len(saved_paths)} ticker frames to {frames_dir}")

    stats = {
        "video": {
            "filename": video_path.name,
            "resolution": f"{frame_w}x{frame_h}",
            "width_px": frame_w,
            "height_px": frame_h,
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "duration_seconds": round(duration_sec, 1),
            "duration_formatted": f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s",
        },
        "extraction": {
            "frame_sample_rate": config.FRAME_SAMPLE_RATE,
            "effective_fps": round(fps / config.FRAME_SAMPLE_RATE, 1),
            "frames_sampled": frames_sampled,
            "frames_saved": len(saved_paths),
            "frames_skipped_black": frames_skipped_black,
            "black_frame_pct": round(frames_skipped_black / max(frames_sampled, 1) * 100, 1),
            "ticker_region": {
                "y_range_pct": f"{config.TICKER_Y_START_PCT}-{config.TICKER_Y_END_PCT}",
                "x_range_pct": f"{config.TICKER_X_START_PCT}-{config.TICKER_X_END_PCT}",
                "width_px": ticker_w,
                "height_px": ticker_h,
            },
        },
    }

    return saved_paths, stats


if __name__ == "__main__":
    videos = sorted(config.VIDEO_DIR.glob("*.mp4"))
    if videos:
        paths, stats = extract_ticker_frames(videos[0])
        import json
        print(json.dumps(stats, indent=2))
    else:
        print("No video found.")
