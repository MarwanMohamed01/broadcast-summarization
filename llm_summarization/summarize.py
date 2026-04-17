"""
LLM News Summarization - Main Script

Sends the extracted news items to each configured LLM and saves
their summaries for later comparison and evaluation.

Usage:
    python summarize.py              # Run all models
    python summarize.py --model 0    # Run only model at index 0
    python summarize.py --list       # List available models
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import config
from prompt import SYSTEM_PROMPT, build_user_prompt


# ── Provider functions ──
# Each returns (summary_text, metadata_dict)

def call_groq(model_id: str, system: str, user: str) -> tuple[str, dict]:
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    msg = response.choices[0].message.content
    meta = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return msg, meta


def call_ollama(model_id: str, system: str, user: str) -> tuple[str, dict]:
    import requests
    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    msg = data["message"]["content"]
    meta = {
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
    }
    return msg, meta


def call_gemini(model_id: str, system: str, user: str) -> tuple[str, dict]:
    import requests
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{model_id}:generateContent?key={config.GOOGLE_API_KEY}")
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Gemini 2.5 returns thinking in separate parts; get the last non-thought part
    parts = data["candidates"][0]["content"]["parts"]
    msg = next(p["text"] for p in reversed(parts) if not p.get("thought"))
    usage = data.get("usageMetadata", {})
    meta = {
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
    }
    return msg, meta


def call_huggingface(model_id: str, system: str, user: str) -> tuple[str, dict]:
    from huggingface_hub import InferenceClient
    client = InferenceClient(
        model=model_id,
        token=config.HUGGINGFACE_API_KEY,
    )
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    msg = response.choices[0].message.content
    meta = {
        "input_tokens": getattr(response.usage, "prompt_tokens", 0),
        "output_tokens": getattr(response.usage, "completion_tokens", 0),
    }
    return msg, meta


def call_openai(model_id: str, system: str, user: str) -> tuple[str, dict]:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    msg = response.choices[0].message.content
    meta = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return msg, meta


def call_cohere(model_id: str, system: str, user: str) -> tuple[str, dict]:
    import cohere
    client = cohere.ClientV2(api_key=config.COHERE_API_KEY)
    response = client.chat(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    msg = response.message.content[0].text
    usage = response.usage
    meta = {
        "input_tokens": getattr(usage.billed_units, "input_tokens", 0) if usage and usage.billed_units else 0,
        "output_tokens": getattr(usage.billed_units, "output_tokens", 0) if usage and usage.billed_units else 0,
    }
    return msg, meta


# ── Dispatch ──
PROVIDERS = {
    "groq": call_groq,
    "ollama": call_ollama,
    "gemini": call_gemini,
    "huggingface": call_huggingface,
    "openai": call_openai,
    "cohere": call_cohere,
}


def run_model(provider: str, model_id: str, display_name: str,
              system: str, user: str) -> dict:
    """Run a single model and return result dict."""
    print(f"\n  [{display_name}]")
    print(f"    Provider: {provider}, Model: {model_id}")

    call_fn = PROVIDERS[provider]

    start = time.time()
    try:
        summary, meta = call_fn(model_id, system, user)
        # Strip <think>...</think> reasoning tags (e.g. Qwen3)
        if "<think>" in summary:
            import re as _re
            summary = _re.sub(r"<think>.*?</think>", "", summary, flags=_re.DOTALL).strip()
        elapsed = round(time.time() - start, 2)
        print(f"    Done in {elapsed}s ({meta.get('output_tokens', '?')} output tokens)")
        print(f"    Summary preview: {summary[:120]}...")

        return {
            "provider": provider,
            "model_id": model_id,
            "display_name": display_name,
            "summary": summary,
            "input_tokens": meta.get("input_tokens", 0),
            "output_tokens": meta.get("output_tokens", 0),
            "latency_seconds": elapsed,
            "status": "success",
            "error": None,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"    FAILED in {elapsed}s: {e}")
        return {
            "provider": provider,
            "model_id": model_id,
            "display_name": display_name,
            "summary": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": elapsed,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description="LLM News Summarization")
    parser.add_argument("--model", type=int, help="Run only model at this index")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for i, (prov, mid, name) in enumerate(config.MODELS):
            print(f"  [{i}] {name} ({prov}/{mid})")
        return

    # Load news items
    print(f"Loading news items from {config.NEWS_ITEMS_PATH}...")
    with open(config.NEWS_ITEMS_PATH, "r", encoding="utf-8") as f:
        news_items = json.load(f)
    print(f"  Loaded {len(news_items)} items")

    # Build prompt
    system = SYSTEM_PROMPT
    user = build_user_prompt(news_items)
    print(f"  Prompt length: {len(user)} chars")

    # Select models
    if args.model is not None:
        models_to_run = [config.MODELS[args.model]]
        print(f"\nRunning single model: {models_to_run[0][2]}")
    else:
        models_to_run = config.MODELS
        print(f"\nRunning {len(models_to_run)} models...")

    # Run each model
    results = []
    for provider, model_id, display_name in models_to_run:
        result = run_model(provider, model_id, display_name, system, user)
        results.append(result)

    # Save results
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.OUTPUT_DIR / f"summaries_{timestamp}.json"

    output_data = {
        "run_timestamp": datetime.now().isoformat(),
        "news_items_count": len(news_items),
        "news_items_source": str(config.NEWS_ITEMS_PATH),
        "prompt": {
            "system": system,
            "user": user,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Also save as latest.json for easy access
    latest_path = config.OUTPUT_DIR / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "error"]

    for r in results:
        status = "OK" if r["status"] == "success" else "FAIL"
        summary_len = len(r["summary"]) if r["summary"] else 0
        print(f"  [{status}] {r['display_name']:30s} | "
              f"{r['latency_seconds']:6.1f}s | "
              f"{int(r['output_tokens']):4d} tokens | "
              f"{summary_len:4d} chars")

    print(f"\n  {len(successes)}/{len(results)} models succeeded")
    if failures:
        print(f"  Failed: {', '.join(r['display_name'] for r in failures)}")
    print(f"\n  Results saved to: {output_path}")
    print(f"  Latest copy at:   {latest_path}")


if __name__ == "__main__":
    main()
