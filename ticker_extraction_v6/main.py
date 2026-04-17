"""
Ticker Extraction v6 - Main Pipeline Orchestrator

Pipeline:
  1. Extract ticker frames from video
  2. Detect scroll offsets between frames
  3. Stitch frames into panoramic image
  4. OCR the panoramic image
  5. Segment into individual news items (JSON)

Outputs:
  - output/final/news_items.json     (extracted news headlines)
  - output/final/pipeline_stats.json  (all pipeline statistics)

Usage:
  python main.py
  python main.py --video path/to/video.mp4
  python main.py --engine tesseract   (default: easyocr)
  python main.py --sample-rate 10     (default: 5)
  python main.py --chunk-minutes 30   (process in 30-min chunks for long videos)
"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from rapidfuzz import fuzz
import config

from step1_extract_ticker import extract_ticker_frames
from step2_scroll_detection import detect_all_scrolls
from step3_stitch_image import stitch_panorama
from step4_ocr import ocr_panorama_chunks
from step5_segment import segment_news


def find_video() -> Path:
    """Find the input video file."""
    videos = sorted(config.VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in {config.VIDEO_DIR}")
    for v in videos:
        if "AlJazeera" in v.name or "Aljazeera" in v.name:
            return v
    return videos[0]


def _get_video_info(video_path: Path) -> dict:
    """Get video metadata without processing."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    return info


def run_pipeline_single(video_path: Path, args) -> tuple[list[dict], dict]:
    """
    Run the full 5-step pipeline on a video (or a portion of it).
    This is the original v6 pipeline logic, extracted into a function.
    Returns (news_items, all_stats).
    """
    pipeline_start = time.time()
    all_stats = {}
    step_times = {}

    # Step 1: Extract ticker frames
    print("\n[Step 1/5] Extracting ticker frames...")
    t0 = time.time()
    frame_paths, step1_stats = extract_ticker_frames(video_path)
    step_times["step1_extract_frames"] = round(time.time() - t0, 1)
    all_stats.update(step1_stats)
    if not frame_paths:
        print("ERROR: No frames extracted. Check video path and ticker coordinates.")
        return [], all_stats

    # Step 2: Detect scroll
    print("\n[Step 2/5] Detecting scroll offsets...")
    t0 = time.time()
    scroll_data, step2_stats = detect_all_scrolls(frame_paths)
    step_times["step2_scroll_detection"] = round(time.time() - t0, 1)
    all_stats.update(step2_stats)
    if not scroll_data:
        print("ERROR: Scroll detection failed.")
        return [], all_stats

    # Step 3: Stitch panorama
    print("\n[Step 3/5] Stitching panoramic image...")
    t0 = time.time()
    panorama_paths, step3_stats = stitch_panorama(scroll_data)
    step_times["step3_stitch_panorama"] = round(time.time() - t0, 1)
    all_stats.update(step3_stats)
    if not panorama_paths:
        print("ERROR: Panorama stitching failed.")
        return [], all_stats

    # Step 4: OCR
    print("\n[Step 4/5] Running OCR...")
    t0 = time.time()
    full_text, step4_stats = ocr_panorama_chunks(panorama_paths)
    step_times["step4_ocr"] = round(time.time() - t0, 1)
    all_stats.update(step4_stats)
    if not full_text:
        print("ERROR: OCR produced no text.")
        return [], all_stats

    # Step 5: Segment
    print("\n[Step 5/5] Segmenting news items...")
    t0 = time.time()
    news_items, step5_stats = segment_news(full_text)
    step_times["step5_segmentation"] = round(time.time() - t0, 1)
    all_stats.update(step5_stats)

    total_time = round(time.time() - pipeline_start, 1)
    all_stats["timing"] = {
        "total_seconds": total_time,
        "total_formatted": f"{int(total_time // 60)}m {int(total_time % 60)}s",
        "steps": step_times,
    }

    return news_items, all_stats


def run_pipeline_chunk(video_path: Path, start_frame: int, end_frame: int,
                       chunk_idx: int, chunk_output_dir: Path) -> tuple[list[dict], dict]:
    """
    Run the 5-step pipeline on a specific frame range of the video.
    Uses a dedicated output directory for this chunk.
    Returns (news_items, stats).
    """
    pipeline_start = time.time()
    all_stats = {}
    step_times = {}

    # Set up chunk-specific output dirs
    frames_dir = chunk_output_dir / "ticker_frames"
    panorama_dir = chunk_output_dir / "panorama"
    ocr_dir = chunk_output_dir / "ocr"
    final_dir = chunk_output_dir / "final"

    # Step 1: Extract ticker frames for this chunk
    print(f"\n[Step 1/5] Extracting ticker frames (chunk {chunk_idx})...")
    t0 = time.time()
    frame_paths, step1_stats = extract_ticker_frames(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        output_dir=frames_dir,
    )
    step_times["step1_extract_frames"] = round(time.time() - t0, 1)
    all_stats.update(step1_stats)
    if not frame_paths:
        print(f"  Chunk {chunk_idx}: No frames extracted, skipping.")
        return [], all_stats

    # Step 2: Detect scroll
    print(f"\n[Step 2/5] Detecting scroll offsets (chunk {chunk_idx})...")
    t0 = time.time()
    scroll_data, step2_stats = detect_all_scrolls(frame_paths)
    step_times["step2_scroll_detection"] = round(time.time() - t0, 1)
    all_stats.update(step2_stats)
    if not scroll_data:
        print(f"  Chunk {chunk_idx}: Scroll detection failed, skipping.")
        return [], all_stats

    # Step 3: Stitch panorama — temporarily override config dirs
    orig_panorama_dir = config.PANORAMA_DIR
    config.PANORAMA_DIR = panorama_dir
    print(f"\n[Step 3/5] Stitching panoramic image (chunk {chunk_idx})...")
    t0 = time.time()
    panorama_paths, step3_stats = stitch_panorama(scroll_data)
    step_times["step3_stitch_panorama"] = round(time.time() - t0, 1)
    all_stats.update(step3_stats)
    config.PANORAMA_DIR = orig_panorama_dir
    if not panorama_paths:
        print(f"  Chunk {chunk_idx}: Panorama stitching failed, skipping.")
        return [], all_stats

    # Step 4: OCR — temporarily override config dirs
    orig_ocr_dir = config.OCR_DIR
    config.OCR_DIR = ocr_dir
    print(f"\n[Step 4/5] Running OCR (chunk {chunk_idx})...")
    t0 = time.time()
    full_text, step4_stats = ocr_panorama_chunks(panorama_paths)
    step_times["step4_ocr"] = round(time.time() - t0, 1)
    all_stats.update(step4_stats)
    config.OCR_DIR = orig_ocr_dir
    if not full_text:
        print(f"  Chunk {chunk_idx}: OCR produced no text, skipping.")
        return [], all_stats

    # Step 5: In chunked mode, do raw headline splitting instead of
    # step5's segmentation (which was tuned for one specific video).
    # Cross-chunk merge will handle dedup and quality filtering.
    print(f"\n[Step 5/5] Splitting headlines (chunk {chunk_idx})...")
    t0 = time.time()
    raw_items = _split_headlines_raw(full_text)
    news_items = [{"id": i, "text": t} for i, t in enumerate(raw_items, 1)]
    step5_stats = {"segmentation": {"raw_items": len(raw_items)}}
    step_times["step5_segmentation"] = round(time.time() - t0, 1)
    all_stats.update(step5_stats)

    # Save to chunk's final dir
    final_dir.mkdir(parents=True, exist_ok=True)
    with open(final_dir / "news_items.json", "w", encoding="utf-8") as f:
        json.dump(news_items, f, indent=2, ensure_ascii=False)

    total_time = round(time.time() - pipeline_start, 1)
    all_stats["timing"] = {
        "total_seconds": total_time,
        "total_formatted": f"{int(total_time // 60)}m {int(total_time % 60)}s",
        "steps": step_times,
    }

    print(f"\n  Chunk {chunk_idx}: {len(news_items)} news items extracted in {all_stats['timing']['total_formatted']}")
    return news_items, all_stats


HEADLINE_PATTERN = re.compile(
    r"(?:"
    r"IRAN(?:'?S)?\s+(?:REVOLUTIONARY|FOREIGN|ATOMIC|PARLIAMENT|MISSION|SAYS)"
    r"|IRANIAN\s+(?:MEDIA|AUTHORITIES|PRESIDENT|FOREIGN|MISSILE)"
    r"|U\.?S\.?\s+(?:FORCES|RESCUE)"
    r"|TRUMP\s+(?:THREATENS|SAYS|TELLS)"
    r"|PRESIDENT\s+TRUMP"
    r"|ISRAELI\s+(?:AIR|MILITARY|FORCES|ARMY|ATTACKS?)"
    r"|GAZA\s+HEALTH"
    r"|HAMAS\s+SAYS"
    r"|LEBANESE\s+(?:ARMY|HEALTH)"
    r"|LEBANON'?S?\s+HEALTH"
    r"|MEDICS?\s+SAY"
    r"|EIGHT\s+OPEC"
    r"|QATAR|QATARI"
    r"|D\.?R\.?\s+CONGO"
    r"|RESCUE\s+CHARITIES"
    r"|ARTEMIS"
    r"|MULTIPLE\s+INJURIES"
    r"|TWO\s+PEOPLE\s+KILLED"
    r"|SIX\s+PEOPLE\s+INJURED"
    r"|MORE\s+THAN\s+(?:A\s+)?DOZEN"
    r"|WORLD\s+HEALTH"
    r"|KUWAIT"
    r"|BAHRAIN"
    r"|STATE\s+MEDIA"
    r"|AT\s+LEAST\s+\w+\s+PEOPLE"
    r"|BREAKING\s+NEWS"
    r"|(?:CHINA|RUSSIA|INDIA|INDONESIA|POLAND|UKRAINE|SUDAN|SAUDI|UAE)"
    r"(?:'?S)?\s+[A-Z]"
    r")",
    re.IGNORECASE,
)


def _split_headlines_raw(text: str) -> list[str]:
    """
    Split OCR text at headline-start boundaries.
    Returns all items >= 20 chars, removing 'FOR MORE GO TO' items.
    No quality filtering — that happens in merge_chunk_results.
    """
    text = re.sub(r"[|{}\[\]\\~`^;]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    positions = sorted(set(m.start() for m in HEADLINE_PATTERN.finditer(text)))
    if not positions:
        return []

    items = []
    for i in range(len(positions)):
        start = positions[i]
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        raw = text[start:end].strip()
        # Trim trailing 1-2 char fragments
        raw = re.sub(r"\s+\S{1,2}$", "", raw).strip()
        # Skip very short and "FOR MORE" items
        if len(raw) < 20:
            continue
        if re.match(r"(?i)for\s+more", raw):
            continue
        if re.match(r"(?i)breaking\s+news\s*$", raw):
            continue
        # Remove "BREAKING NEWS" prefix if followed by real content
        raw = re.sub(r"^BREAKING\s+NEWS\s+", "", raw).strip()
        if len(raw) >= 20:
            items.append(raw)

    return items


def _item_quality(text: str) -> float:
    """
    Score how clean/readable a news item is (0-1).
    Penalizes: consonant clusters, mixed case within words,
    repeated characters, very short fragments, broken punctuation.
    """
    if len(text) < 20:
        return 0.0
    words = text.split()
    if len(words) < 3:
        return 0.0

    good_words = 0
    total_scored = 0
    for w in words:
        letters = re.sub(r"[^a-zA-Z]", "", w)
        if len(letters) == 0:
            continue  # skip numbers/punctuation
        total_scored += 1

        is_good = True

        # Consonant soup: >=4 letters, no vowels
        if len(letters) >= 4:
            vowels = sum(1 for c in letters if c.lower() in "aeiou")
            if vowels == 0:
                is_good = False

        # Excessive case switching within word (like "tIrOcH", "ISRAEUSKE")
        if len(letters) >= 4 and is_good:
            switches = sum(
                1 for a, b in zip(letters, letters[1:])
                if a.isupper() != b.isupper()
            )
            # Allow words that are ALL CAPS or Capitalized
            if switches > 2 and switches >= len(letters) * 0.4:
                is_good = False

        # Repeated character runs (like "HEALTISTRY", "CONGCONGO")
        if len(letters) >= 5 and is_good:
            for j in range(len(letters) - 4):
                if len(set(letters[j:j+5].lower())) <= 2:
                    is_good = False
                    break

        if is_good:
            good_words += 1

    if total_scored == 0:
        return 0.0

    word_q = good_words / total_scored

    # Penalize items that end mid-word or with junk
    last_word = words[-1] if words else ""
    last_letters = re.sub(r"[^a-zA-Z]", "", last_word)
    ending_penalty = 0
    if len(last_letters) <= 2 and len(last_letters) > 0:
        ending_penalty = 0.05
    if text.endswith("-") or text.endswith(","):
        ending_penalty = 0.05

    # Length score (prefer items 40-120 chars)
    length = len(text)
    if length < 40:
        len_q = 0.3
    elif length <= 120:
        len_q = 1.0
    else:
        len_q = max(0.5, 1.0 - (length - 120) / 200)

    return word_q * 0.6 + len_q * 0.4 - ending_penalty


def _is_cross_chunk_duplicate(item_text: str, existing_texts: list[str]) -> int:
    """
    Check if item is a duplicate of any existing item.
    Returns index of the match, or -1 if no match.
    Uses aggressive matching: digit-stripped comparison + prefix matching.
    """
    item_lower = item_text.lower()
    item_nodigit = re.sub(r"[\d,.]", "", item_lower)
    item_words = item_lower.split()[:6]
    item_prefix = " ".join(item_words) if len(item_words) >= 4 else ""

    for idx, existing in enumerate(existing_texts):
        existing_lower = existing.lower()

        # Direct comparison with lower thresholds
        full_sim = fuzz.ratio(item_lower, existing_lower)
        if full_sim > 50:
            return idx

        token_sim = fuzz.token_sort_ratio(item_lower, existing_lower)
        if token_sim > 55:
            return idx

        # Partial match
        partial_sim = fuzz.partial_ratio(item_lower, existing_lower)
        len_ratio = min(len(item_text), len(existing)) / max(len(item_text), len(existing))
        if partial_sim > 70 and len_ratio > 0.3:
            return idx

        # Digit-stripped comparison (72,292 vs 7,292 vs 72,290 etc.)
        existing_nodigit = re.sub(r"[\d,.]", "", existing_lower)
        if len(item_nodigit) > 15 and len(existing_nodigit) > 15:
            if fuzz.ratio(item_nodigit, existing_nodigit) > 55:
                return idx

        # Prefix comparison (first 6 words)
        if item_prefix:
            existing_words = existing_lower.split()[:6]
            existing_prefix = " ".join(existing_words) if len(existing_words) >= 4 else ""
            if existing_prefix and fuzz.ratio(item_prefix, existing_prefix) > 65:
                return idx

    return -1


def merge_chunk_results(all_chunk_items: list[list[dict]]) -> list[dict]:
    """
    Merge news items from all chunks with aggressive deduplication.

    Strategy: collect ALL items from all chunks, then for each unique
    headline, pick the cleanest version across all chunks.
    """
    # Determine a minimum items threshold to exclude bad chunks.
    # Good chunks typically have 70+ items (multiple ticker cycles with delimiters).
    # Bad chunks (early hours, no scrolling, OCR noise) have < 30.
    items_per_chunk = [len(c) for c in all_chunk_items]
    good_chunks = [n for n in items_per_chunk if n > 0]
    if good_chunks:
        median_items = sorted(good_chunks)[len(good_chunks) // 2]
        min_items_threshold = max(10, median_items // 3)
    else:
        min_items_threshold = 10
    print(f"  Chunk quality threshold: >= {min_items_threshold} items (median={median_items if good_chunks else 0})")

    # Flatten all items, skipping low-quality chunks
    all_items = []
    for chunk_idx, chunk_items in enumerate(all_chunk_items):
        if len(chunk_items) < min_items_threshold:
            print(f"  Skipping chunk {chunk_idx} ({len(chunk_items)} items < threshold)")
            continue
        for item in chunk_items:
            text = item["text"]
            # Remove "BREAKING NEWS" prefix
            text = re.sub(r"^BREAKING\s+NEWS\s+", "", text).strip()
            quality = _item_quality(text)
            all_items.append({"text": text, "quality": quality, "chunk": chunk_idx})

    print(f"  Total items across all chunks: {len(all_items)}")

    # Sort by: chunk text length descending (longer OCR = more ticker cycles = better quality),
    # then by item length descending (longer items are more complete)
    chunk_sizes = {}
    for item in all_items:
        c = item["chunk"]
        if c not in chunk_sizes:
            chunk_sizes[c] = sum(len(it["text"]) for it in all_items if it["chunk"] == c)
    all_items.sort(key=lambda x: (chunk_sizes.get(x["chunk"], 0), len(x["text"])), reverse=True)

    # Deduplicate: for each item, check against existing unique set
    unique_texts = []
    unique_qualities = []
    unique_chunks = []
    new_per_chunk = [0] * len(all_chunk_items)
    dup_per_chunk = [0] * len(all_chunk_items)

    for item in all_items:
        text = item["text"]
        quality = item["quality"]
        chunk = item["chunk"]

        match_idx = _is_cross_chunk_duplicate(text, unique_texts)
        if match_idx >= 0:
            # Replace if this version is clearly better
            if quality > unique_qualities[match_idx] + 0.03:
                unique_texts[match_idx] = text
                unique_qualities[match_idx] = quality
                unique_chunks[match_idx] = chunk
            elif (abs(quality - unique_qualities[match_idx]) <= 0.03
                  and len(text) > len(unique_texts[match_idx]) * 1.15
                  and len(text) < 200):
                unique_texts[match_idx] = text
                unique_qualities[match_idx] = quality
                unique_chunks[match_idx] = chunk
            dup_per_chunk[chunk] += 1
        else:
            unique_texts.append(text)
            unique_qualities.append(quality)
            unique_chunks.append(chunk)
            new_per_chunk[chunk] += 1

    for i in range(len(all_chunk_items)):
        if new_per_chunk[i] > 0 or dup_per_chunk[i] > 0:
            print(f"  Chunk {i}: {new_per_chunk[i]} new, {dup_per_chunk[i]} duplicates")

    # Filter out garbage
    filtered = []
    for text, quality in zip(unique_texts, unique_qualities):
        if len(text) < 40:
            continue
        if quality < 0.55:
            continue
        if re.match(r"(?i)for\s+more", text.strip()):
            continue
        filtered.append(text)

    print(f"  After quality filter: {len(filtered)}")

    # Re-number
    results = [{"id": i, "text": t} for i, t in enumerate(filtered, 1)]
    return results


def main():
    parser = argparse.ArgumentParser(description="Ticker Extraction v6")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--engine", choices=["tesseract", "easyocr"], help="OCR engine")
    parser.add_argument("--sample-rate", type=int, help="Frame sample rate")
    parser.add_argument("--chunk-minutes", type=int, default=0,
                        help="Process video in N-minute chunks (0 = no chunking)")
    args = parser.parse_args()

    if args.engine:
        config.OCR_ENGINE = args.engine
    if args.sample_rate:
        config.FRAME_SAMPLE_RATE = args.sample_rate

    print("=" * 60)
    print("  Ticker Extraction v6 Pipeline")
    print("=" * 60)

    # Find video
    if args.video:
        video_path = Path(args.video)
    else:
        video_path = find_video()

    # ──────────────────────────────────────────────
    # CHUNKED MODE
    # ──────────────────────────────────────────────
    if args.chunk_minutes > 0:
        print(f"\n  Mode: CHUNKED ({args.chunk_minutes}-minute chunks)")
        overall_start = time.time()

        video_info = _get_video_info(video_path)
        fps = video_info["fps"]
        total_frames = video_info["total_frames"]
        duration_sec = video_info["duration_sec"]
        chunk_frames = int(args.chunk_minutes * 60 * fps)
        num_chunks = max(1, (total_frames + chunk_frames - 1) // chunk_frames)

        print(f"  Video: {video_path.name}")
        print(f"  Duration: {duration_sec/3600:.1f} hours ({total_frames} frames at {fps:.0f} fps)")
        print(f"  Chunks: {num_chunks} x {args.chunk_minutes} min")

        chunks_dir = config.OUTPUT_DIR / "chunks"
        all_chunk_items = []
        all_chunk_stats = []

        for i in range(num_chunks):
            start_f = i * chunk_frames
            end_f = min((i + 1) * chunk_frames, total_frames)
            start_sec = start_f / fps
            end_sec = end_f / fps

            print(f"\n{'='*60}")
            print(f"  CHUNK {i}/{num_chunks-1}: {start_sec/60:.0f}min - {end_sec/60:.0f}min")
            print(f"  Frames {start_f} - {end_f}")
            print(f"{'='*60}")

            chunk_dir = chunks_dir / f"chunk_{i:03d}"

            # Check if this chunk was already processed
            chunk_final = chunk_dir / "final" / "news_items.json"
            if chunk_final.exists():
                print(f"  Already processed, loading existing results...")
                try:
                    with open(chunk_final, "r", encoding="utf-8") as f:
                        items = json.load(f)
                    all_chunk_items.append(items)
                    print(f"  Loaded {len(items)} items from previous run")
                    continue
                except Exception:
                    pass

            items, stats = run_pipeline_chunk(
                video_path, start_f, end_f, i, chunk_dir
            )
            all_chunk_items.append(items)
            all_chunk_stats.append(stats)

            # Clean up ticker frames and panorama images to save disk space
            for cleanup_dir in [chunk_dir / "ticker_frames", chunk_dir / "panorama"]:
                if cleanup_dir.exists():
                    shutil.rmtree(cleanup_dir)
                    print(f"  Cleaned up {cleanup_dir.name}/ to save disk space")

        # Merge across chunks
        print(f"\n{'='*60}")
        print(f"  MERGING {num_chunks} chunks...")
        print(f"{'='*60}")

        total_before_merge = sum(len(items) for items in all_chunk_items)
        merged_items = merge_chunk_results(all_chunk_items)

        print(f"\n  Total items across chunks: {total_before_merge}")
        print(f"  After cross-chunk dedup: {len(merged_items)}")

        # Save final merged results
        config.FINAL_DIR.mkdir(parents=True, exist_ok=True)
        output_path = config.FINAL_DIR / "news_items.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_items, f, indent=2, ensure_ascii=False)

        overall_time = round(time.time() - overall_start, 1)

        # Save stats
        merged_stats = {
            "mode": "chunked",
            "chunk_minutes": args.chunk_minutes,
            "num_chunks": num_chunks,
            "total_items_before_merge": total_before_merge,
            "final_unique_items": len(merged_items),
            "per_chunk_items": [len(items) for items in all_chunk_items],
            "video": {
                "filename": video_path.name,
                "duration_seconds": round(duration_sec, 1),
                "total_frames": total_frames,
                "fps": round(fps, 1),
            },
            "timing": {
                "total_seconds": overall_time,
                "total_formatted": f"{int(overall_time // 60)}m {int(overall_time % 60)}s",
            },
            "config": {
                "frame_sample_rate": config.FRAME_SAMPLE_RATE,
                "ocr_engine": config.OCR_ENGINE,
                "quality_threshold": config.QUALITY_THRESHOLD,
                "max_news_length": config.MAX_NEWS_LENGTH,
                "dedup_similarity_threshold": config.DEDUP_SIMILARITY_THRESHOLD,
                "dedup_partial_threshold": config.DEDUP_PARTIAL_THRESHOLD,
                "min_news_length": config.MIN_NEWS_LENGTH,
            },
        }
        stats_path = config.FINAL_DIR / "pipeline_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(merged_stats, f, indent=2, ensure_ascii=False)

        # Summary
        print(f"\n{'='*60}")
        print(f"  Pipeline Complete! (chunked mode)")
        print(f"  Total time: {merged_stats['timing']['total_formatted']}")
        print(f"  Chunks processed: {num_chunks}")
        print(f"  News items: {total_before_merge} total -> {len(merged_items)} unique")
        print(f"  Output: {output_path}")
        print(f"  Stats:  {stats_path}")
        print(f"{'='*60}")

        if merged_items:
            print(f"\nAll extracted items ({len(merged_items)}):")
            for item in merged_items:
                print(f"  [{item['id']:3d}] {item['text'][:120]}{'...' if len(item['text']) > 120 else ''}")

        return

    # ──────────────────────────────────────────────
    # ORIGINAL MODE (no chunking)
    # ──────────────────────────────────────────────
    news_items, all_stats = run_pipeline_single(video_path, args)

    # Pipeline config used
    all_stats["config"] = {
        "frame_sample_rate": config.FRAME_SAMPLE_RATE,
        "ocr_engine": config.OCR_ENGINE,
        "quality_threshold": config.QUALITY_THRESHOLD,
        "max_news_length": config.MAX_NEWS_LENGTH,
        "dedup_similarity_threshold": config.DEDUP_SIMILARITY_THRESHOLD,
        "dedup_partial_threshold": config.DEDUP_PARTIAL_THRESHOLD,
        "min_news_length": config.MIN_NEWS_LENGTH,
    }

    # Save stats
    config.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = config.FINAL_DIR / "pipeline_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print(f"  Total time: {all_stats['timing']['total_formatted']}")
    print(f"  News items extracted: {len(news_items)}")
    print(f"  Output: {config.FINAL_DIR / 'news_items.json'}")
    print(f"  Stats:  {stats_path}")
    print("=" * 60)

    # Print first few items as preview
    if news_items:
        print("\nPreview (first 5 items):")
        for item in news_items[:5]:
            print(f"  [{item['id']}] {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")

    # Print key stats summary
    print("\n--- Key Statistics ---")
    print(f"  Video: {all_stats['video']['filename']} ({all_stats['video']['duration_formatted']})")
    print(f"  Frames: {all_stats['extraction']['frames_saved']}/{all_stats['extraction']['frames_sampled']} saved (black frames skipped: {all_stats['extraction']['frames_skipped_black']})")
    print(f"  Scroll: {all_stats['scroll_detection']['total_scroll_px']}px total, {all_stats['scroll_detection']['avg_scroll_per_frame_px']}px avg/frame")
    print(f"  Panorama: {all_stats['stitching']['num_chunks']} chunks, {all_stats['stitching']['overall_fill_rate_pct']}% fill rate")
    print(f"  OCR: {all_stats['ocr']['total_words_extracted']} words, {all_stats['ocr']['total_text_length_chars']} chars, avg confidence: {all_stats['ocr']['word_confidence']['avg']}%")
    print(f"  Segmentation: {all_stats['segmentation']['raw_items_after_split']} raw -> {all_stats['segmentation']['final_unique_news_items']} unique items")
    print(f"  Estimated ticker cycles: {all_stats['segmentation']['estimated_ticker_cycles']}")
    print(f"  Timing: {' | '.join(f'{k}: {v}s' for k, v in all_stats['timing']['steps'].items())}")


if __name__ == "__main__":
    main()
