"""
Consistent prompt template for all LLMs.

All models receive the exact same system prompt and user prompt
to ensure fair comparison.
"""

SYSTEM_PROMPT = (
    "You are a professional news editor. Your task is to read a list of "
    "news headlines extracted from a TV news ticker and produce a single, "
    "coherent summary paragraph that covers all the key stories. "
    "Write in clear, neutral journalistic language. "
    "Do not add information that is not in the headlines. "
    "Do not use bullet points or lists — write one flowing paragraph."
)


def build_user_prompt(news_items: list[dict]) -> str:
    """
    Build the user prompt from the extracted news items.
    Each item is a dict with 'id' and 'text'.
    """
    headlines = "\n".join(
        f"{item['id']}. {item['text']}" for item in news_items
    )

    return (
        f"Below are {len(news_items)} news headlines extracted from an "
        f"Al Jazeera news ticker. Summarize ALL of them into one coherent "
        f"paragraph.\n\n"
        f"{headlines}"
    )
