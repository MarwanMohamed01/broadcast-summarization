"""
LLM post-processing pass for VLM-extracted headlines.

Mirrors `pipeline/clean_news_items.py` architecturally — same provider,
same batching, same fallback chain — but the prompt targets the failure
modes specific to VLM extraction rather than OCR:

  1. SPLIT mashed-together headlines (Gemini sometimes concatenates
     two adjacent ticker headlines when the " - " delimiter falls on
     a tile boundary).
  2. COMPLETE fragments using cross-item context (same as OCR).
  3. DROP garbage strings (scrambled-character emissions from tiles
     where Gemini mis-segmented characters across cycle seams).
  4. DEDUP cycle-repeats — VLM sees ~75 ticker cycles in 14 h; the
     same headline appears many times with slight digit drift
     (e.g. "AT LEAST 280 PALESTINIANS" → "AT LEAST 282 ...").
  5. PRESERVE wording — do not paraphrase.

Reads:
    vlm_extraction/output/<vlm>/<slice>/run_<n>/headlines_combined.json
Writes:
    vlm_extraction/output/<vlm>/<slice>/run_<n>/headlines_combined_cleaned.json
    (same schema; the extra _cleaned suffix mirrors the OCR convention)

Usage:
    python -m pipeline.clean_vlm_headlines --vlm gemini --slice full_video --run 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz

PROJECT_DIR = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_DIR / "llm_summarization" / ".env")
load_dotenv(PROJECT_DIR / "vlm_extraction" / ".env", override=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

VLM_OUTPUT_DIR = PROJECT_DIR / "vlm_extraction" / "output"


SYSTEM_PROMPT = (
    "You are a news headline cleaner. You receive a list of news headlines "
    "extracted from a TV news ticker by a Vision-Language Model (VLM). "
    "The VLM has these specific failure modes:\n"
    "  - It sometimes CONCATENATES two adjacent ticker headlines into one "
    "    string when the ' - ' delimiter is partially cut by a tile "
    "    boundary (e.g. 'EIGHT OPEC+ COUNTRIES AGREE TO INCREASE OIL "
    "    PRODUCTION QUOTAS BY 206,000 BARRELS PER DAY ISRAELI MILITARY "
    "    CONTINUES TO STRIKE BEIRUT' — these are TWO headlines glued "
    "    together with no delimiter).\n"
    "  - It sometimes EMITS FRAGMENTS where the leading or trailing word "
    "    is cut off by the tile edge (e.g. 'STINIANS SINCE OCT 10 "
    "    CEASEFIRE STARTED' — missing the leading 'PALE').\n"
    "  - It sometimes EMITS GARBLED TEXT where characters are "
    "    scrambled (e.g. 'IRI SA HARIPCH IN UAE, KUAHRAIN').\n"
    "  - The same headline appears many times across the 14 h "
    "    broadcast with SLIGHT DIGIT DRIFT (e.g. 'AT LEAST 280 "
    "    PALESTINIANS KILLED' vs 'AT LEAST 282 PALESTINIANS KILLED' — "
    "    the casualty count updates over the day).\n\n"
    "Your task:\n"
    "1. SPLIT any item that contains two or more distinct headlines "
    "   into separate items. Distinct headlines have different "
    "   subjects (different countries, people, or topics).\n"
    "2. COMPLETE truncated headlines using cross-item context — if "
    "   another item in the list contains the full version, output "
    "   the full version.\n"
    "3. DROP garbled strings where >30% of words are not real English "
    "   words. DROP fragments shorter than 30 characters that don't "
    "   contain a complete news statement.\n"
    "4. DEDUP cycle-repeats: when two headlines differ only in a "
    "   number (casualty count, time, etc.), keep ONE — prefer the "
    "   higher number (latest cycle wins). When two headlines are "
    "   near-duplicates with different wordings, keep the most "
    "   complete and grammatical one.\n"
    "5. PRESERVE wording. Do NOT paraphrase. Do NOT add information "
    "   that isn't in the input.\n\n"
    "Output ONLY a JSON array of strings. Each string is one cleaned "
    "headline in ALL CAPS. No explanation, no markdown fences."
)


def call_gemini(user_prompt: str) -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192,
                             "responseMimeType": "application/json"},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    return next(p["text"] for p in reversed(parts) if not p.get("thought"))


def call_groq(user_prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def extract_json_array(text: str) -> list[str]:
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    # Try a top-level array first.
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed
                        if isinstance(x, str) and str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Then try a {"headlines": [...]} object (Groq json_object mode).
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, list):
                        return [str(x).strip() for x in v
                                if isinstance(x, str) and str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Line-based fallback.
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip().strip(",").strip('"').strip("'")
        if len(line) > 20:
            out.append(line)
    return out


def deduplicate(items: list[str]) -> list[str]:
    """Final fuzzy dedup pass after the LLM has done semantic merging.

    Thresholds match pipeline/clean_news_items.py (ratio>70, partial>85)
    so VLM and OCR cleanup behave consistently. Prefers longer headline
    when two are duplicates (truncations lose to complete versions).
    """
    # Sort longest first so a fragment is dropped against the full version,
    # not the other way round.
    candidates = sorted({i for i in items if i and i.strip()},
                        key=len, reverse=True)
    unique: list[str] = []
    for item in candidates:
        is_dup = False
        il = item.lower()
        for existing in unique:
            el = existing.lower()
            if fuzz.ratio(il, el) > 70:
                is_dup = True
                break
            if fuzz.partial_ratio(il, el) > 85:
                ratio = (min(len(item), len(existing))
                         / max(len(item), len(existing)))
                # Loosened from 0.5 → 0.3 so truncated fragments
                # (like "ARTEMIS II ... 'GRAND CANYON" at 50 chars vs
                # the full 110-char version) are correctly dropped.
                if ratio > 0.3:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(item)
    return unique


FINAL_MERGE_PROMPT = (
    "You are a CONSERVATIVE news headline deduplicator. You receive a JSON "
    "array of cleaned news headlines from a 14-hour TV news broadcast. "
    "Your job is ONLY to remove obvious truncation duplicates and "
    "fragments. You are NOT summarising the news. You are NOT merging "
    "stories that just happen to be on similar topics. ALMOST ALL items "
    "should be KEPT.\n\n"
    "ONLY merge in these specific cases:\n"
    "  CASE 1 — TRUNCATION: One item ends mid-word and another item "
    "  contains the complete version. Drop the truncated one.\n"
    "    Example: drop 'ARTEMIS II ASTRONAUTS GLIMPSE MOON' GRAND CANYON' "
    "    (truncated, no closing quote) when "
    "    'ARTEMIS II ASTRONAUTS GLIMPSE MOON GRAND CANYON' AHEAD OF "
    "    HISTORIC LUNAR FLYBY' is also present.\n\n"
    "  CASE 2 — SUBSTRING FRAGMENT: One item is a STRICT substring of "
    "  another item (or of its concatenation with one other word). "
    "  Drop the shorter one.\n"
    "    Example: drop 'COMMANDER KILLED IN U.S.-ISRAELI ATTACKS' when "
    "    'SENIOR IRANIAN MILITARY COMMANDER HAS BEEN KILLED IN "
    "    U.S.-ISRAELI STRIKES' is present (the same story, just shorter).\n\n"
    "  CASE 3 — IDENTICAL STORY, DIFFERENT WORDING: Two items report "
    "  the EXACT SAME EVENT with the same key facts (people, place, "
    "  numbers) but slightly different wording.\n"
    "    Example: drop 'RESCUE CHARITIES SAY 71 PEOPLE REMAIN MISSING "
    "    AT SEA NEAR COAST' when 'RESCUE CHARITIES SAY 71 PEOPLE "
    "    MISSING AT SEA AFTER 32 RESCUED NEAR ITALY' is present.\n\n"
    "Things you MUST NOT do:\n"
    "  X DO NOT drop items that report different events (different "
    "    people / places / dates / numbers).\n"
    "  X DO NOT drop items just because they share subject matter "
    "    (e.g. multiple Gaza headlines about different incidents must "
    "    all be kept).\n"
    "  X DO NOT drop items that simply update a number on the same "
    "    story (e.g. '71 KILLED' and '78 KILLED' on different days "
    "    are SEPARATE — keep both).\n"
    "  X DO NOT paraphrase or rewrite — only delete or keep.\n\n"
    "If you are uncertain whether two items are duplicates, KEEP BOTH. "
    "When in doubt, do nothing.\n\n"
    "You should expect to remove only 10-30% of items. If you would "
    "remove more than that, you are being too aggressive — keep more.\n\n"
    "Output ONLY a JSON array of strings — the kept items in their "
    "original wording, ALL CAPS. No explanation, no markdown fences."
)


# Hard safety: if the LLM removes more than this fraction, reject the
# merge entirely (it's hallucinated dedup decisions).
MAX_FINAL_MERGE_REMOVAL = 0.50


def final_merge(items: list[str]) -> list[str]:
    """One LLM call that sees the entire list and folds semantic duplicates."""
    if len(items) <= 5:
        return items
    numbered = "\n".join(f"{i+1}. {it}" for i, it in enumerate(items))
    prompt = (
        f"Below is a JSON list of {len(items)} cleaned news headlines from "
        f"a 14-hour TV news broadcast. Many are semantic duplicates of the "
        f"same story. Merge them per the rules and output the final "
        f"deduplicated JSON array.\n\nHEADLINES:\n{numbered}"
    )
    for label, fn in [("Gemini 2.5 Flash (final-merge)", _call_gemini_with_system),
                      ("Groq Llama 3.3 70B (final-merge)", _call_groq_with_system)]:
        try:
            print(f"  Final merge: calling {label}...")
            text = fn(FINAL_MERGE_PROMPT, prompt)
            parsed = extract_json_array(text)
            removed = len(items) - len(parsed)
            removal_frac = removed / len(items) if items else 0
            print(f"    -> {len(parsed)} items after final merge "
                  f"(removed {removed}, {removal_frac*100:.0f}%)")
            if not parsed:
                continue
            if removal_frac > MAX_FINAL_MERGE_REMOVAL:
                print(f"    ⚠ removal exceeded {MAX_FINAL_MERGE_REMOVAL*100:.0f}% "
                      f"safety guard — rejecting and keeping pre-merge list")
                continue
            return parsed
        except Exception as exc:
            print(f"    {label} failed: {exc}")
    return items


def _call_gemini_with_system(system: str, user: str) -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192,
                             "responseMimeType": "application/json"},
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    return next(p["text"] for p in reversed(parts) if not p.get("thought"))


def _call_groq_with_system(system: str, user: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def clean_batch(items: list[str], idx: int, total: int) -> list[str]:
    numbered = "\n".join(f"{i+1}. {it}" for i, it in enumerate(items))
    prompt = (
        f"Below are {len(items)} VLM-extracted news headlines from a TV news "
        f"ticker (batch {idx+1}/{total}). Clean them following the "
        f"instructions. Output a JSON array of strings.\n\n"
        f"RAW HEADLINES:\n{numbered}"
    )
    for label, fn in [("Gemini 2.5 Flash", call_gemini),
                      ("Groq Llama 3.3 70B", call_groq)]:
        try:
            print(f"  Batch {idx+1}/{total}: calling {label}...")
            text = fn(prompt)
            parsed = extract_json_array(text)
            print(f"    -> {len(parsed)} cleaned items")
            return parsed
        except Exception as exc:
            print(f"    {label} failed: {exc}")
    return []


def clean_run(vlm: str, slice_name: str, run_idx: int,
              batch_size: int = 30) -> Path:
    in_path = (VLM_OUTPUT_DIR / vlm / slice_name / f"run_{run_idx}"
               / "headlines_combined.json")
    out_path = (VLM_OUTPUT_DIR / vlm / slice_name / f"run_{run_idx}"
                / "headlines_combined_cleaned.json")
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    raw_items = json.loads(in_path.read_text(encoding="utf-8"))
    raw_texts = [it["text"] for it in raw_items]
    print(f"  Loaded {len(raw_texts)} raw VLM headlines from {in_path.name}")

    batches = [raw_texts[i:i + batch_size]
               for i in range(0, len(raw_texts), batch_size)]
    all_cleaned: list[str] = []
    for bi, batch in enumerate(batches):
        all_cleaned.extend(clean_batch(batch, bi, len(batches)))

    print(f"  Total after cleanup batches: {len(all_cleaned)}")
    cleaned = deduplicate(all_cleaned)
    print(f"  After fuzzy dedup: {len(cleaned)}")
    cleaned = [c for c in cleaned if 30 <= len(c) <= 200]
    print(f"  After length filter (30-200 chars): {len(cleaned)}")
    # Final cross-batch semantic merge — sees all items at once.
    cleaned = final_merge(cleaned)
    # One more fuzzy dedup pass after the LLM merge (safety net).
    cleaned = deduplicate(cleaned)
    cleaned = [c for c in cleaned if 30 <= len(c) <= 200]
    print(f"  After final-merge + dedup: {len(cleaned)}")

    out = [{"id": i, "text": t} for i, t in enumerate(cleaned, 1)]
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"  Wrote {len(out)} cleaned items -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--slice", required=True, dest="slice_name")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=30)
    args = ap.parse_args()
    print("=" * 60)
    print(f"  VLM headline cleaner: {args.vlm} / {args.slice_name} / run_{args.run}")
    print("=" * 60)
    clean_run(args.vlm, args.slice_name, args.run, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
