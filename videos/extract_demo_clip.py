"""
Extract a 30-second demo MP4 + WAV from the AlJazeera source video.

Time range: 08:30:00 -> 08:30:30 (matches validation Slice A so the
OCR walkthrough on the demo site lines up with the annotated panorama).

Outputs (idempotent — skip if already exist):
    videos/demo_clip_30s.mp4  (~3-4 MB, 480p H.264)
    videos/demo_clip_30s.wav  (~1 MB, 16 kHz mono PCM)

Both files are gitignored.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.resolve()
VIDEOS_DIR = PROJECT_DIR / "videos"
LOGS_DIR = PROJECT_DIR / "logs"

START_SEC = 8 * 3600 + 30 * 60   # 08:30:00
DURATION_SEC = 30
TARGET_HEIGHT = 480              # downsample to 480p for tiny file size
AUDIO_RATE = 16000


def find_source_video() -> Path:
    candidates = list(VIDEOS_DIR.glob("*.mp4"))
    for c in candidates:
        if "14hrs" in c.name or "14hr" in c.name:
            return c
    if not candidates:
        raise FileNotFoundError(f"No .mp4 in {VIDEOS_DIR}")
    return candidates[0]


def extract_video(src: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [skip] {out_path.name} exists")
        return
    print(f"  Extracting {DURATION_SEC}s of video to {out_path.name} ...")
    container = av.open(str(src))
    in_video = container.streams.video[0]
    in_video.thread_type = "AUTO"

    fps = float(in_video.average_rate or 30)
    end_sec = START_SEC + DURATION_SEC

    # Compute target width keeping aspect ratio
    src_w = in_video.codec_context.width
    src_h = in_video.codec_context.height
    target_h = TARGET_HEIGHT
    target_w = int(src_w * target_h / src_h)
    if target_w % 2:                     # H.264 needs even dims
        target_w -= 1

    out = av.open(str(out_path), mode="w")
    out_stream = out.add_stream("h264", rate=int(round(fps)))
    out_stream.width = target_w
    out_stream.height = target_h
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "26"}

    # Seek to start. PyAV's seek is in stream time-base.
    container.seek(int(START_SEC / float(in_video.time_base)),
                   stream=in_video, any_frame=False)

    written = 0
    for frame in container.decode(video=0):
        ts = float(frame.pts * in_video.time_base)
        if ts < START_SEC:
            continue
        if ts >= end_sec:
            break
        frame = frame.reformat(width=target_w, height=target_h, format="yuv420p")
        for packet in out_stream.encode(frame):
            out.mux(packet)
        written += 1

    for packet in out_stream.encode():    # flush
        out.mux(packet)
    out.close()
    container.close()
    print(f"    wrote {written} frames -> {out_path.stat().st_size/1024:.1f} KB")


def extract_audio(src: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [skip] {out_path.name} exists")
        return
    print(f"  Extracting {DURATION_SEC}s of audio to {out_path.name} ...")
    container = av.open(str(src))
    in_audio = container.streams.audio[0]
    in_audio.thread_type = "AUTO"

    out = av.open(str(out_path), mode="w")
    out_stream = out.add_stream("pcm_s16le", rate=AUDIO_RATE, layout="mono")

    resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_RATE)

    container.seek(int(START_SEC / float(in_audio.time_base)),
                   stream=in_audio, any_frame=False)

    end_sec = START_SEC + DURATION_SEC
    samples_written = 0
    for frame in container.decode(audio=0):
        ts = float(frame.pts * in_audio.time_base) if frame.pts is not None else 0
        if ts < START_SEC:
            continue
        if ts >= end_sec:
            break
        for resampled in resampler.resample(frame):
            for packet in out_stream.encode(resampled):
                out.mux(packet)
            samples_written += resampled.samples

    for packet in out_stream.encode():
        out.mux(packet)
    out.close()
    container.close()
    print(f"    wrote ~{samples_written} samples -> {out_path.stat().st_size/1024:.1f} KB")


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        src = find_source_video()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Source: {src.name}")
    mp4_out = VIDEOS_DIR / "demo_clip_30s.mp4"
    wav_out = VIDEOS_DIR / "demo_clip_30s.wav"

    try:
        extract_video(src, mp4_out)
    except Exception as e:
        print(f"ERROR extracting video: {e}", file=sys.stderr)
        return 2

    try:
        extract_audio(src, wav_out)
    except Exception as e:
        print(f"ERROR extracting audio: {e}", file=sys.stderr)
        return 3

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
