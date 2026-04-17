"""
Run the llm_summarization pipeline on news_items_cleaned.json
(the LLM-cleaned version) instead of the raw news_items.json.

This does NOT modify llm_summarization/ — it monkey-patches the config
paths before importing summarize, so the same pipeline processes the
cleaned input and writes its output to a separate folder.

Output: llm_summarization/output_cleaned/summaries_<timestamp>.json
        llm_summarization/output_cleaned/latest.json
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LLM_DIR = PROJECT_DIR / "llm_summarization"

if not LLM_DIR.exists():
    raise RuntimeError(f"llm_summarization not found: {LLM_DIR}")

# Add llm_summarization to path so its internal imports resolve
sys.path.insert(0, str(LLM_DIR))

# Import the original config first, then override paths in memory.
# When summarize.py later does `import config`, Python returns this same
# already-loaded module, so our overrides are seen.
import config as llm_config

CLEANED_INPUT = (
    PROJECT_DIR / "ticker_extraction_v6" / "output" / "final" / "news_items_cleaned.json"
)
OUTPUT_DIR_CLEANED = LLM_DIR / "output_cleaned"

if not CLEANED_INPUT.exists():
    raise FileNotFoundError(
        f"Cleaned news items not found: {CLEANED_INPUT}\n"
        "Run clean_news_items.py first."
    )

llm_config.NEWS_ITEMS_PATH = CLEANED_INPUT
llm_config.OUTPUT_DIR = OUTPUT_DIR_CLEANED
OUTPUT_DIR_CLEANED.mkdir(parents=True, exist_ok=True)

print(f"Input:  {llm_config.NEWS_ITEMS_PATH}")
print(f"Output: {llm_config.OUTPUT_DIR}")
print()

# Now import and run summarize with the patched config
import summarize  # noqa: E402

# Reset argv so summarize.main()'s argparse doesn't see our wrapper's args
sys.argv = ["summarize_cleaned.py"]
summarize.main()
