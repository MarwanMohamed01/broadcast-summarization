"""
Configuration for Ticker Extraction v6
"""
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
VIDEO_DIR = PROJECT_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output"

# Sub-output directories (created automatically)
FRAMES_DIR = OUTPUT_DIR / "ticker_frames"
PANORAMA_DIR = OUTPUT_DIR / "panorama"
OCR_DIR = OUTPUT_DIR / "ocr"
FINAL_DIR = OUTPUT_DIR / "final"

# ── Video ──
# Will auto-detect the first .mp4 in VIDEO_DIR
FRAME_SAMPLE_RATE = 5  # Process every Nth frame (5 = every 5th frame at 30fps = 6fps effective)

# ── Ticker Region ──
# Crop coordinates as percentages of frame dimensions
TICKER_Y_START_PCT = 0.88   # Top of ticker (88% from top = bottom 12%)
TICKER_Y_END_PCT = 1.0      # Bottom of ticker
TICKER_X_START_PCT = 0.14   # Skip the ALJAZEERA logo on the left
TICKER_X_END_PCT = 0.96     # Skip right edge noise

# ── Scroll Detection ──
SCROLL_TEMPLATE_HEIGHT = None  # Use full ticker height (set at runtime)
SCROLL_TEMPLATE_WIDTH_PCT = 0.5  # Use middle 50% of ticker as template
SCROLL_MAX_SHIFT = 80  # Max pixels the ticker can scroll between sampled frames
SCROLL_MIN_SHIFT = 1   # Minimum shift to count as movement

# ── Panorama ──
PANORAMA_MAX_WIDTH = 50000  # Max width before starting a new panorama chunk
PANORAMA_OVERLAP_FOR_OCR = 200  # Overlap between chunks to avoid cutting words

# ── OCR ──
OCR_ENGINE = "easyocr"  # "tesseract" or "easyocr"
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── Segmentation ──
NEWS_DELIMITERS = [" - ", " | ", " • ", " · "]
MIN_NEWS_LENGTH = 20  # Minimum characters for a valid news item
QUALITY_THRESHOLD = 0.86  # Slightly lower than v5's 0.88 to capture U.S./abbreviation items
MAX_NEWS_LENGTH = 150  # Same as v5
DEDUP_SIMILARITY_THRESHOLD = 60  # Same as v5
DEDUP_PARTIAL_THRESHOLD = 80  # Same as v5
