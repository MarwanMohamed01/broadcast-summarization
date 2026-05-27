"""Prompts used by every VLM adapter.

Kept here, not in adapters, so all four models receive byte-identical
instructions and any score difference is attributable to the model
itself rather than to prompt drift.
"""

EXTRACT_PROMPT = """You are analysing a 3000x87 pixel image. The image is one tile cut from a longer stitched panorama of a TV news ticker bar. The ticker contains news headlines separated by " - " delimiters that scroll horizontally across the screen.

YOUR JOB
========
Extract the COMPLETE, FULLY VISIBLE news headlines in this tile — and nothing else.

WHAT IS A "COMPLETE, FULLY VISIBLE" HEADLINE
============================================
A headline is fully visible if and only if ALL of these are true:
1. Its FIRST word starts cleanly inside the tile (you can see the entire first word — not a partial word at the left edge).
2. Its LAST word ends cleanly inside the tile (you can see the entire last word — not a partial word at the right edge).
3. There is a visible " - " delimiter (or the tile edge after a space) immediately before the first word, and immediately after the last word — i.e. you can confirm the headline's boundaries on BOTH sides.
4. It reads as a complete news statement: it has a subject and a verb, it is grammatically intelligible, and it would make sense to a reader by itself.

If any of these fails — the first word is cut, the last word is cut, the boundary delimiter is missing, or the text is grammatically incomplete — DO NOT EMIT IT. Adjacent tiles overlap with this one by 1000 pixels, so the headline you skip will be fully visible in a neighbouring tile.

CRITICAL RULES
==============
- DO NOT emit partial headlines. If a headline is cut off at either edge, skip it entirely. Do not include it with leading/trailing ellipsis. Just leave it out.
- DO NOT join two headlines into one. If you see "[HEADLINE A] - [HEADLINE B]", they are TWO separate headlines. Always preserve the " - " boundary as a hard split. Never output "[HEADLINE A] [HEADLINE B]" as a single string.
- DO NOT emit garbled text. If you cannot read a word with confidence (>80% sure), do not include the headline.
- DO NOT invent or hallucinate. If a headline is unclear, omit it entirely.
- DO NOT include UI elements: channel logos, lower-thirds, breaking news banners, time displays, weather widgets, scoreboards, "FOR MORE GO TO ALJAZEERA.COM" or similar promotional text. Extract ONLY the scrolling news ticker headlines.
- DO NOT include the " - " delimiter in the headline text itself.
- The text in the ticker is in ENGLISH and in ALL CAPS. Output exactly what you see, in ALL CAPS, including any commas, colons, apostrophes, and quotation marks.

EXAMPLES
========
Tile content (left edge → right edge): "...IN ISRAELI ATTACKS - EIGHT OPEC+ COUNTRIES AGREE TO INCREASE PRODUCTION QUOTAS BY 206,000 BARRELS PER DAY FROM MAY - ISRAELI MILITARY..."

Correct emission:
{
  "headlines": [
    {"text": "EIGHT OPEC+ COUNTRIES AGREE TO INCREASE PRODUCTION QUOTAS BY 206,000 BARRELS PER DAY FROM MAY"}
  ]
}

WRONG (do not do this):
- "IN ISRAELI ATTACKS - EIGHT OPEC+..." (concatenated two headlines + included a fragment)
- "ISRAELI ATTACKS" (left-edge fragment, first word may be cut)
- "ISRAELI MILITARY" (right-edge fragment, last word may be cut)
- "EIGHT OPEC COUNTRIES AGREE TO INCREASE PRODUCTION QUOTAS BY 206 BARRELS" (missing details, paraphrased)

OUTPUT FORMAT
=============
Return strictly valid JSON, exactly this schema, nothing else:
{
  "headlines": [
    {"text": "<full headline as visible, ALL CAPS>"}
  ]
}

If the tile contains zero fully-visible complete headlines, output:
{
  "headlines": []
}

Do not include any text outside the JSON. No prose. No markdown fences. No "confidence" field — confidence is implicit (you only emit headlines you are confident about)."""
