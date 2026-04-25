"""
Reuse v6's step5 segmentation logic to convert raw OCR text produced by
any engine into a clean list of news-item headlines.

We import v6's step5_segment unmodified (v6 is frozen — we only read its
functions) and call its public `segment_news()` on whatever text the
engine produced. The result is written to
ocr_comparison/output/<slice>/<engine>_headlines.json.
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
V6_DIR = PROJECT_DIR / "ticker_extraction_v6"
sys.path.insert(0, str(V6_DIR))


def segment_text_for_engine(full_text: str, slice_name: str, engine: str,
                            output_dir: Path) -> list[dict]:
    """
    Run v6 segmentation on the given OCR text and save the headline list.

    We monkey-patch v6 config's FINAL_DIR to our per-engine output path
    so step5 writes its JSON in the right place. config is restored at
    end so repeated calls don't leak state.
    """
    import config as v6_config
    from step5_segment import segment_news

    target_dir = output_dir / slice_name
    target_dir.mkdir(parents=True, exist_ok=True)

    orig_final = v6_config.FINAL_DIR
    v6_config.FINAL_DIR = target_dir

    try:
        items, stats = segment_news(full_text)
    finally:
        v6_config.FINAL_DIR = orig_final

    # Step 5 writes news_items.json in FINAL_DIR — rename to engine-scoped
    default_out = target_dir / "news_items.json"
    scoped_out = target_dir / f"{engine}_headlines.json"
    if default_out.exists():
        default_out.replace(scoped_out)

    stats_out = target_dir / f"{engine}_segmentation_stats.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return items
