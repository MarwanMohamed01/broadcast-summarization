# Variance report — visual pipeline (3 runs)

_Generated 2026-04-29T14:04:33.389763_  
_Reference: `llm_summarization\reference_summary.txt` (687 words)_

| Model | n | ROUGE-1 (mean ± std) | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash | 2 | 69.3 ± 2.0 | 37.7 ± 2.4 | 21.0 ± 0.4 | 86.5 ± 0.1 |
| Llama 4 Scout 17B (Groq) | 3 | 52.1 ± 2.6 | 24.6 ± 0.3 | 23.0 ± 2.6 | 85.5 ± 0.1 |
| Command-R (Cohere) | 3 | 41.3 ± 2.0 | 16.3 ± 3.9 | 17.1 ± 2.6 | 85.1 ± 0.3 |
| Llama 3.3 70B (Groq) | 3 | 50.7 ± 3.2 | 26.1 ± 3.8 | 22.0 ± 2.1 | 85.0 ± 0.3 |
| Qwen3 32B (Groq) | 3 | 46.2 ± 6.5 | 18.2 ± 3.9 | 19.4 ± 1.4 | 84.4 ± 0.1 |
| Llama 3.1 8B (Groq) | 3 | 39.8 ± 2.4 | 17.9 ± 2.6 | 19.1 ± 0.6 | 84.3 ± 0.6 |
| Llama 3 8B (HuggingFace) | 3 | 35.4 ± 8.8 | 15.8 ± 7.4 | 15.9 ± 3.6 | 83.9 ± 1.1 |
| Llama 3.2 3B (Ollama local) | 2 | 38.5 ± 1.6 | 17.5 ± 2.5 | 19.9 ± 3.4 | 83.8 ± 0.7 |
| Llama 3.1 8B (Ollama local) | 0 | — | — | — | — |

_All values shown as **percent**. n=number of successful runs out of 3. std requires n≥2 (else reported as 0)._