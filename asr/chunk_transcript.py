"""
Split a Whisper transcript JSON into time-based chunks for summarization.

Default chunk size: 15 minutes. For a 14-hour video, this produces
~56 chunks. Each chunk is saved as a plain-text file containing all
segments that started within that 15-minute window.

Usage:
    python asr/chunk_transcript.py
    python asr/chunk_transcript.py --input asr/output/transcript_full.json
    python asr/chunk_transcript.py --chunk-minutes 10
"""

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_DIR / "asr" / "output"


def chunk_transcript(json_path: Path, chunk_minutes: int = 15,
                     out_dir: Path = None) -> list[Path]:
    """
    Split a transcript JSON into time-based chunks.

    Returns the list of written chunk files.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Transcript not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = data["segments"]
    total_duration = data.get("duration_seconds", 0)

    if out_dir is None:
        out_dir = OUTPUT_DIR / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing chunk files from prior runs
    for old in out_dir.glob("chunk_*.txt"):
        old.unlink()

    chunk_seconds = chunk_minutes * 60
    chunks_count = max(1, int((total_duration + chunk_seconds - 1) // chunk_seconds))
    print(f"Transcript: {json_path.name}")
    print(f"  Duration: {total_duration:.0f}s ({total_duration/60:.1f} min)")
    print(f"  Segments: {len(segments)}")
    print(f"  Chunking into {chunks_count} x {chunk_minutes}-minute chunks")

    # Bucket segments by chunk index
    buckets: dict[int, list[dict]] = {}
    for seg in segments:
        chunk_idx = int(seg["start"] // chunk_seconds)
        buckets.setdefault(chunk_idx, []).append(seg)

    written = []
    for idx in sorted(buckets.keys()):
        bucket = buckets[idx]
        start_sec = idx * chunk_seconds
        end_sec = min((idx + 1) * chunk_seconds, total_duration)
        out_file = out_dir / f"chunk_{idx:03d}.txt"

        # Write header + segments
        lines = [
            f"# Chunk {idx:03d}   {start_sec/60:.0f}-{end_sec/60:.0f} min",
            f"# {len(bucket)} segments, "
            f"{sum(len(s['text'].split()) for s in bucket)} words",
            "",
        ]
        # Plain text, one segment per line (no timestamps inside, but header has range)
        for seg in bucket:
            lines.append(seg["text"].strip())

        out_file.write_text("\n".join(lines), encoding="utf-8")
        written.append(out_file)

        word_count = sum(len(s["text"].split()) for s in bucket)
        print(f"  chunk_{idx:03d}: {start_sec/60:.0f}-{end_sec/60:.0f} min "
              f"({len(bucket)} seg, {word_count} words)")

    print(f"\n  [DONE] Wrote {len(written)} chunks to {out_dir}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Chunk a Whisper transcript by time")
    parser.add_argument("--input", type=str, default=None,
                        help="Transcript JSON (default: transcript_full.json)")
    parser.add_argument("--chunk-minutes", type=int, default=15,
                        help="Chunk size in minutes (default: 15)")
    args = parser.parse_args()

    json_path = Path(args.input) if args.input else OUTPUT_DIR / "transcript_full.json"
    chunk_transcript(json_path, args.chunk_minutes)


if __name__ == "__main__":
    main()
