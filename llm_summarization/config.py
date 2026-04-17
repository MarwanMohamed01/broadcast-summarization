"""
Configuration for LLM News Summarization (Phase 2)
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# Load API keys from .env
load_dotenv(Path(__file__).parent / ".env")

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
NEWS_ITEMS_PATH = PROJECT_DIR / "ticker_extraction_v6" / "output" / "final" / "news_items.json"

# ── API Keys ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# ── Models to evaluate ──
# Each entry: (provider, model_id, display_name)
MODELS = [
    # Groq (free tier - very fast cloud inference)
    ("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)"),
    ("groq", "llama-3.1-8b-instant", "Llama 3.1 8B (Groq)"),
    ("groq", "qwen/qwen3-32b", "Qwen3 32B (Groq)"),
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B (Groq)"),

    # Ollama (local, free)
    ("ollama", "llama3.2", "Llama 3.2 3B (Ollama local)"),
    ("ollama", "llama3.1:8b", "Llama 3.1 8B (Ollama local)"),

    # Google Gemini (free tier)
    ("gemini", "gemini-2.5-flash", "Gemini 2.5 Flash"),

    # Hugging Face Inference API (free tier)
    ("huggingface", "meta-llama/Meta-Llama-3-8B-Instruct", "Llama 3 8B (HuggingFace)"),

    # Cohere (free trial)
    ("cohere", "command-r-08-2024", "Command-R (Cohere)"),
]

# ── Ollama settings ──
OLLAMA_BASE_URL = "http://localhost:11434"
