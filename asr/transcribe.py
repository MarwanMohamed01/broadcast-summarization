"""
Transcribe audio using faster-whisper, with internal chunking for long files.

faster-whisper loads the entire audio into memory for feature extraction,
which fails (MemoryError ~7 GB) on 14-hour audio. This script solves it
by reading the WAV in N-minute chunks via the `wave` module, passing
each chunk as a numpy array to the model, and combining the resulting
segments (with proper time offsets) into a single transcript.

Outputs:
  - transcript.json — full segments with global timestamps
  - transcript.txt  — plain text one-segment-per-line with timestamps
  - transcript.srt  — SRT subtitles

Resume-friendly: if a per-chunk intermediate JSON already exists in
asr/output/_chunks/, that chunk is skipped on re-run.

Usage:
    python asr/transcribe.py                                         (full audio)
    python asr/transcribe.py --audio asr/output/audio_slice_30600_1800.wav
    python asr/transcribe.py --chunk-minutes 20   (tune internal chunk size)
"""

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

PROJECT_DIR = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_DIR / "asr" / "output"


def format_timestamp(seconds: float, always_include_hours: bool = True) -> str:
    """Format seconds as HH:MM:SS,mmm."""
    assert seconds >= 0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    if always_include_hours or h > 0:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{m:02d}:{s:02d},{ms:03d}"


def read_wav_chunk(wav_file: wave.Wave_read, start_sample: int,
                   chunk_samples: int) -> np.ndarray:
    """
    Read `chunk_samples` samples starting at `start_sample` from a WAV
    file, returning a float32 mono numpy array normalized to [-1, 1].
    Returns an empty array if the position is past end-of-file.
    """
    wav_file.setpos(start_sample)
    remaining = wav_file.getnframes() - start_sample
    to_read = min(chunk_samples, remaining)
    if to_read <= 0:
        return np.array([], dtype=np.float32)

    frames = wav_file.readframes(to_read)
    sample_width = wav_file.getsampwidth()
    n_channels = wav_file.getnchannels()

    if sample_width == 2:
        dtype = np.int16
        scale = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    arr = np.frombuffer(frames, dtype=dtype)
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return (arr.astype(np.float32) / scale)


def transcribe_chunked(audio_path: Path, model_size: str = "small",
                       language: str = "en", out_prefix: str = "transcript",
                       chunk_minutes: int = 20) -> None:
    """
    Stream-transcribe a WAV file in chunks to avoid loading full audio into RAM.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks_intermediate_dir = OUTPUT_DIR / f"_chunks_{out_prefix}"
    chunks_intermediate_dir.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"{out_prefix}.json"
    txt_path = OUTPUT_DIR / f"{out_prefix}.txt"
    srt_path = OUTPUT_DIR / f"{out_prefix}.srt"

    # Open WAV and get metadata
    wav_file = wave.open(str(audio_path), "rb")
    sample_rate = wav_file.getframerate()
    total_samples = wav_file.getnframes()
    total_duration = total_samples / sample_rate

    print(f"Audio: {audio_path.name}")
    print(f"  Sample rate: {sample_rate} Hz, total samples: {total_samples:,}")
    print(f"  Duration: {total_duration:.0f}s ({total_duration/60:.1f} min)")

    chunk_seconds = chunk_minutes * 60
    chunk_samples = chunk_seconds * sample_rate
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    print(f"  Splitting into {num_chunks} x {chunk_minutes}-minute chunks")

    print(f"\nLoading Whisper model: {model_size} ({language})")
    t0 = time.time()
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    all_segments = []
    overall_start = time.time()

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        start_sec = start_sample / sample_rate
        end_sec = min((chunk_idx + 1) * chunk_seconds, total_duration)
        chunk_inter_file = chunks_intermediate_dir / f"chunk_{chunk_idx:03d}.json"

        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks}  "
              f"({start_sec/60:.1f}-{end_sec/60:.1f} min)  ---")

        # Resume: skip if already transcribed
        if chunk_inter_file.exists():
            cached = json.loads(chunk_inter_file.read_text(encoding="utf-8"))
            all_segments.extend(cached["segments"])
            print(f"  [cached] {len(cached['segments'])} segments loaded from {chunk_inter_file.name}")
            continue

        # Read this chunk's audio from WAV (mono float32, normalized)
        audio_arr = read_wav_chunk(wav_file, start_sample, chunk_samples)
        if audio_arr.size == 0:
            break

        duration_this_chunk = len(audio_arr) / sample_rate
        print(f"  Loaded {duration_this_chunk:.0f}s of audio ({len(audio_arr):,} samples)")

        # Transcribe this chunk
        t_chunk = time.time()
        segments_iter, info = model.transcribe(
            audio_arr,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        chunk_segments = []
        for segment in segments_iter:
            # Apply chunk offset to timestamps
            chunk_segments.append({
                "id": segment.id,
                "start": round(segment.start + start_sec, 3),
                "end": round(segment.end + start_sec, 3),
                "text": segment.text.strip(),
                "avg_logprob": round(segment.avg_logprob, 4),
                "no_speech_prob": round(segment.no_speech_prob, 4),
            })

        elapsed = time.time() - t_chunk
        speed = duration_this_chunk / elapsed if elapsed > 0 else 0
        overall_elapsed = (time.time() - overall_start) / 60
        audio_processed_min = end_sec / 60
        est_remaining_min = (num_chunks - chunk_idx - 1) * (elapsed / 60)
        print(f"  [{len(chunk_segments)} segments in {elapsed:.0f}s, {speed:.2f}x realtime]")
        print(f"  Overall: {audio_processed_min:.0f}/{total_duration/60:.0f} min audio, "
              f"{overall_elapsed:.0f} min elapsed, "
              f"~{est_remaining_min:.0f} min remaining")
        if chunk_segments:
            print(f"  Sample: \"{chunk_segments[len(chunk_segments)//2]['text'][:100]}\"")

        # Save intermediate chunk result (resume-friendly)
        with open(chunk_inter_file, "w", encoding="utf-8") as f:
            json.dump({"chunk_idx": chunk_idx, "start_sec": start_sec,
                       "end_sec": end_sec, "segments": chunk_segments},
                      f, indent=2, ensure_ascii=False)

        all_segments.extend(chunk_segments)

    wav_file.close()

    total_elapsed = time.time() - overall_start
    print(f"\n  Total: {len(all_segments)} segments in {total_elapsed/60:.1f} min")
    if total_elapsed > 0:
        print(f"  Overall speed: {total_duration/total_elapsed:.2f}x realtime")

    # Renumber segment IDs globally
    for i, s in enumerate(all_segments):
        s["id"] = i

    # Write final JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "audio_file": str(audio_path),
            "duration_seconds": total_duration,
            "language": language,
            "model": model_size,
            "segment_count": len(all_segments),
            "segments": all_segments,
        }, f, indent=2, ensure_ascii=False)

    # TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in all_segments:
            ts = format_timestamp(seg["start"], always_include_hours=False)
            f.write(f"[{ts}] {seg['text']}\n")

    # SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(all_segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")

    print(f"\n  [DONE]")
    print(f"    JSON: {json_path}")
    print(f"    TXT:  {txt_path}")
    print(f"    SRT:  {srt_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe long audio with Whisper (chunked)")
    parser.add_argument("--audio", type=str, default=None,
                        help="Audio WAV file (default: asr/output/audio_full.wav)")
    parser.add_argument("--model", type=str, default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: small)")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code (default: en)")
    parser.add_argument("--prefix", type=str, default="transcript",
                        help="Output filename prefix (default: transcript)")
    parser.add_argument("--chunk-minutes", type=int, default=20,
                        help="Internal chunk size in minutes (default: 20)")
    args = parser.parse_args()

    audio_path = Path(args.audio) if args.audio else OUTPUT_DIR / "audio_full.wav"
    transcribe_chunked(audio_path, args.model, args.language, args.prefix, args.chunk_minutes)


if __name__ == "__main__":
    main()
