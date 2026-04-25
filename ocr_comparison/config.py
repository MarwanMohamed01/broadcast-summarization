"""
Configuration for the multi-OCR comparison experiment.

All engines read the SAME two panorama PNGs (Slice A and Slice B) from
validation/ground_truth/ and write their outputs to
ocr_comparison/output/<slice>/<engine>_*.json.
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
GT_DIR = PROJECT_DIR / "validation" / "ground_truth"
OUTPUT_DIR = Path(__file__).parent / "output"
LOGS_DIR = PROJECT_DIR / "logs"

# Slice panoramas (already annotated with ground truth)
SLICES = {
    "slice_A": {
        "panorama": GT_DIR / "slice_A_panorama.png",
        "ground_truth": GT_DIR / "slice_A_headlines.txt",
    },
    "slice_B": {
        "panorama": GT_DIR / "slice_B_panorama.png",
        "ground_truth": GT_DIR / "slice_B_headlines.txt",
    },
}

# Engine registry — name => module path
# Gemini is omitted from the reported comparison because free-tier rate limits
# make timing measurements non-comparable (tiles stall on 429 retries, not on
# real inference cost). The engine file is kept under engines/ for future use.
ENGINES = [
    "tesseract",
    "easyocr",
    "paddle",
    "doctr",
    "craft_trocr",  # hybrid (reported separately)
]

# v6 panorama-slicing params (so each engine gets identical slice inputs)
SLICE_W = 3000
STRIDE = SLICE_W // 2

# Tesseract executable (Windows)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Gemini settings
GEMINI_MODEL = "gemini-2.5-flash"
# Wider tiles mean fewer API calls — important because free-tier is 20 RPM.
# 3000 px matches the v6 slicing width, so Gemini sees the same chunks.
GEMINI_TILE_W = 3000
GEMINI_TILE_OVERLAP = 300   # overlap so dashes near tile borders aren't split
GEMINI_UPSCALE = 3          # upscale ticker strip before sending (87px -> 261px)
GEMINI_THROTTLE_S = 7.0     # seconds to sleep between calls (stays under 10 RPM)

# Confidence thresholds (per engine)
CONF_THRESHOLDS = {
    "tesseract": 20,      # 0-100 (Tesseract scale)
    "easyocr": 0.30,      # 0-1
    "paddle": 0.50,       # 0-1
    "doctr": 0.30,        # 0-1
}
