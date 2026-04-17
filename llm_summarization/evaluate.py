"""
LLM Summary Evaluation Script

Compares each LLM's summary against a human reference summary using:
  - ROUGE-1 (unigram overlap)
  - ROUGE-2 (bigram overlap)
  - ROUGE-L (longest common subsequence)
  - BERTScore (semantic similarity)

Usage:
    python evaluate.py                          # Use latest.json + reference_summary.txt
    python evaluate.py --summaries FILE.json    # Use a specific summaries file
    python evaluate.py --reference FILE.txt     # Use a specific reference summary
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from rouge_score import rouge_scorer
from bert_score import score as bert_score

import config


def load_reference(path: Path) -> str:
    """Load the human reference summary."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Reference summary is empty: {path}")
    return text


def load_summaries(path: Path) -> dict:
    """Load the LLM summaries JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_rouge(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bertscore(hypotheses: list[str], references: list[str]) -> list[dict]:
    """Compute BERTScore for a batch of hypothesis-reference pairs."""
    P, R, F1 = bert_score(
        hypotheses, references,
        lang="en",
        verbose=True,
    )
    results = []
    for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
        results.append({
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM summaries")
    parser.add_argument(
        "--summaries", type=str,
        default=str(config.OUTPUT_DIR / "latest.json"),
        help="Path to summaries JSON file",
    )
    parser.add_argument(
        "--reference", type=str,
        default=str(config.BASE_DIR / "reference_summary.txt"),
        help="Path to human reference summary",
    )
    args = parser.parse_args()

    summaries_path = Path(args.summaries)
    reference_path = Path(args.reference)

    # Load data
    print(f"Loading reference summary from {reference_path}...")
    reference = load_reference(reference_path)
    print(f"  Reference length: {len(reference)} chars, {len(reference.split())} words")

    print(f"Loading LLM summaries from {summaries_path}...")
    data = load_summaries(summaries_path)
    results = [r for r in data["results"] if r["status"] == "success"]
    print(f"  Found {len(results)} successful summaries")

    # Compute ROUGE for each model
    print("\nComputing ROUGE scores...")
    for r in results:
        r["rouge"] = compute_rouge(r["summary"], reference)
        print(f"  {r['display_name']:30s} | "
              f"R1={r['rouge']['rouge1']:.4f} "
              f"R2={r['rouge']['rouge2']:.4f} "
              f"RL={r['rouge']['rougeL']:.4f}")

    # Compute BERTScore (batch for efficiency)
    print("\nComputing BERTScore (this may take a moment on first run)...")
    hypotheses = [r["summary"] for r in results]
    references_list = [reference] * len(results)
    bert_results = compute_bertscore(hypotheses, references_list)

    for r, bs in zip(results, bert_results):
        r["bertscore"] = bs
        print(f"  {r['display_name']:30s} | "
              f"P={bs['precision']:.4f} "
              f"R={bs['recall']:.4f} "
              f"F1={bs['f1']:.4f}")

    # Build evaluation output
    evaluation = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "reference_summary_path": str(reference_path),
        "reference_summary_length_chars": len(reference),
        "reference_summary_length_words": len(reference.split()),
        "summaries_source": str(summaries_path),
        "models_evaluated": len(results),
        "metrics": ["rouge1", "rouge2", "rougeL", "bertscore_precision",
                     "bertscore_recall", "bertscore_f1"],
        "results": [],
    }

    for r in results:
        evaluation["results"].append({
            "display_name": r["display_name"],
            "provider": r["provider"],
            "model_id": r["model_id"],
            "summary_length_chars": len(r["summary"]),
            "summary_length_words": len(r["summary"].split()),
            "latency_seconds": r["latency_seconds"],
            "rouge1": r["rouge"]["rouge1"],
            "rouge2": r["rouge"]["rouge2"],
            "rougeL": r["rouge"]["rougeL"],
            "bertscore_precision": r["bertscore"]["precision"],
            "bertscore_recall": r["bertscore"]["recall"],
            "bertscore_f1": r["bertscore"]["f1"],
        })

    # Sort by BERTScore F1 descending (best first)
    evaluation["results"].sort(key=lambda x: x["bertscore_f1"], reverse=True)

    # Save results
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_path = config.OUTPUT_DIR / f"evaluation_{timestamp}.json"
    latest_eval_path = config.OUTPUT_DIR / "evaluation_latest.json"

    for path in [eval_path, latest_eval_path]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

    # Print final ranking table
    print("\n" + "=" * 100)
    print("  EVALUATION RESULTS (ranked by BERTScore F1)")
    print("=" * 100)
    print(f"  {'Rank':<5} {'Model':<30} {'ROUGE-1':<9} {'ROUGE-2':<9} "
          f"{'ROUGE-L':<9} {'BERT-F1':<9} {'Words':<7} {'Latency':<8}")
    print("-" * 100)

    for i, r in enumerate(evaluation["results"], 1):
        print(f"  {i:<5} {r['display_name']:<30} "
              f"{r['rouge1']:<9.4f} {r['rouge2']:<9.4f} "
              f"{r['rougeL']:<9.4f} {r['bertscore_f1']:<9.4f} "
              f"{r['summary_length_words']:<7} {r['latency_seconds']:<8.1f}")

    print("-" * 100)
    print(f"\n  Results saved to: {eval_path}")
    print(f"  Latest copy at:   {latest_eval_path}")


if __name__ == "__main__":
    main()
