"""
Extract audio from a video file to 16 kHz mono WAV (Whisper's preferred
format) using PyAV. No ffmpeg command-line needed.

Supports extracting either the full audio or a time-bounded slice (for
testing on a 30-minute portion before running the full 14-hour pipeline).

Usage:
    python asr/extract_audio.py
    python asr/extract_audio.py --start 30600 --duration 1800  (slice A = 30 min starting at 8:30h)
    python asr/extract_audio.py --output asr/output/audio_full.wav
"""

import argparse
import wave
from pathlib import Path

import av
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.resolve()
VIDEOS_DIR = PROJECT_DIR / "videos"
OUTPUT_DIR = PROJECT_DIR / "asr" / "output"

TARGET_SAMPLE_RATE = 16000  # Whisper standard
TARGET_CHANNELS = 1         # mono


def find_video() -> Path:
    """Find the 14-hour AlJazeera video."""
    candidates = list(VIDEOS_DIR.glob("*.mp4"))
    for c in candidates:
        if "14hrs" in c.name or "14hr" in c.name:
            return c
    for c in candidates:
        if "Aljazeera" in c.name or "AlJazeera" in c.name:
            return c
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No .mp4 files in {VIDEOS_DIR}")


def extract_audio(video_path: Path, out_path: Path,
                  start_sec: float = 0, duration_sec: float = 0) -> None:
    """Extract mono 16 kHz WAV audio from a video file using PyAV.

    Args:
        video_path: source video
        out_path: output WAV path
        start_sec: seek start in seconds (0 = from beginning)
        duration_sec: slice length in seconds (0 = until end)
    """
    if out_path.exists():
        print(f"  Output already exists: {out_path} — skipping extraction.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    print(f"  Video: {video_path.name}")
    print(f"  Start: {start_sec:.0f}s  Duration: {duration_sec:.0f}s (0 = full)")
    print(f"  Target: {TARGET_SAMPLE_RATE} Hz mono WAV")

    container = av.open(str(video_path))
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        raise RuntimeError(f"No audio stream in {video_path}")

    # Seek to start (PyAV seeks in stream time units)
    if start_sec > 0:
        seek_pts = int(start_sec / audio_stream.time_base)
        container.seek(seek_pts, stream=audio_stream, any_frame=False)

    # Resample to 16 kHz mono s16
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)

    end_sec = start_sec + duration_sec if duration_sec > 0 else float("inf")
    total_samples = 0

    with wave.open(str(tmp_path), "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(TARGET_SAMPLE_RATE)

        last_print = 0
        for frame in container.decode(audio_stream):
            # Current position in seconds
            pts_sec = float(frame.pts * frame.time_base) if frame.pts is not None else 0
            if pts_sec < start_sec:
                continue
            if pts_sec >= end_sec:
                break

            # Resample and write each resampled frame
            resampled = resampler.resample(frame)
            for r in resampled:
                arr = r.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]  # mono after layout=mono
                wav.writeframes(arr.astype(np.int16).tobytes())
                total_samples += arr.shape[-1]

            # Progress log every ~30 seconds of audio
            if pts_sec - last_print >= 30:
                elapsed_min = (pts_sec - start_sec) / 60
                print(f"    {elapsed_min:.1f} min extracted  (pos {pts_sec/60:.1f} min)")
                last_print = pts_sec

        # Flush resampler
        try:
            for r in resampler.resample(None):
                arr = r.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                wav.writeframes(arr.astype(np.int16).tobytes())
                total_samples += arr.shape[-1]
        except Exception:
            pass

    container.close()

    # Atomically rename tmp → final
    tmp_path.replace(out_path)

    duration_wrote = total_samples / TARGET_SAMPLE_RATE
    size_mb = out_path.stat().st_size / 1e6
    print(f"  [DONE] wrote {size_mb:.1f} MB ({duration_wrote/60:.1f} min) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio for Whisper")
    parser.add_argument("--video", type=str, help="Video file (default: auto-detect 14h)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (0 = full)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output WAV path (default: auto)")
    args = parser.parse_args()

    video = Path(args.video) if args.video else find_video()

    if args.output:
        out = Path(args.output)
    elif args.duration > 0:
        # Test-slice output
        out = OUTPUT_DIR / f"audio_slice_{int(args.start)}_{int(args.duration)}.wav"
    else:
        out = OUTPUT_DIR / "audio_full.wav"

    extract_audio(video, out, start_sec=args.start, duration_sec=args.duration)


if __name__ == "__main__":
    main()
