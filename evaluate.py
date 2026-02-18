"""Evaluation pipeline for the Thesis RAG system.

Implements a multi-criterion evaluation framework inspired by the validation
methodology from the thesis itself: instead of relying on a single aggregate
score, we measure multiple orthogonal dimensions of system quality.

Criteria:
    1. Retrieval Relevance  — Are the retrieved chunks actually relevant?
    2. Faithfulness          — Is the answer grounded in the retrieved context?
    3. Answer Completeness   — Does the answer cover the expected key points?
    4. Hallucination Check   — Does the answer contain claims not in the context?

Usage:
    python evaluate.py
    python evaluate.py --top-k 3
    python evaluate.py --verbose
"""

import argparse
import json
import time
from pathlib import Path

import anthropic

import config
from retrieval import retrieve, build_context, generate_answer


EVAL_PROMPT_TEMPLATE = """You are an evaluation judge. Given a question, expected answer, retrieved context, 
and generated answer, score the system on four criteria.

Question: {question}

Expected Answer: {expected_answer}

Retrieved Context:
{context}

Generated Answer: {generated_answer}

Score each criterion from 0.0 to 1.0 and provide a brief justification.

Respond ONLY with valid JSON in this exact format:
{{
  "retrieval_relevance": {{"score": 0.0, "reason": "..."}},
  "faithfulness": {{"score": 0.0, "reason": "..."}},
  "answer_completeness": {{"score": 0.0, "reason": "..."}},
  "hallucination_free": {{"score": 0.0, "reason": "..."}}
}}

Scoring guide:
- retrieval_relevance: 1.0 if all top chunks are relevant to the question, 0.0 if none are.
- faithfulness: 1.0 if every claim in the answer is supported by the context, 0.0 if fabricated.
- answer_completeness: 1.0 if the answer covers all key points from the expected answer, 0.0 if none.
- hallucination_free: 1.0 if no claims go beyond the context, 0.0 if major fabrications present."""


def judge_single(question: str, expected: str, context: str, generated: str) -> dict:
    """Use Claude as an LLM judge to score a single Q&A pair."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = EVAL_PROMPT_TEMPLATE.format(
        question=question,
        expected_answer=expected,
        context=context,
        generated_answer=generated,
    )

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    return json.loads(text)


def run_evaluation(top_k: int = config.TOP_K, verbose: bool = False) -> dict:
    """Run the full evaluation pipeline over all ground-truth questions.

    Returns a summary dict with per-question and aggregate scores.
    """
    eval_path = Path(__file__).parent / "eval_questions.json"
    with open(eval_path) as f:
        questions = json.load(f)

    results = []
    print(f"\n{'='*60}")
    print(f"EVALUATION PIPELINE — {len(questions)} questions, top_k={top_k}")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:70]}...")
        t0 = time.time()

        # Retrieve
        chunks = retrieve(q["question"], top_k=top_k)
        context = build_context(chunks)

        # Generate
        answer = generate_answer(q["question"], context)

        # Judge
        scores = judge_single(q["question"], q["expected_answer"], context, answer)

        elapsed = time.time() - t0

        result = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "expected_answer": q["expected_answer"],
            "generated_answer": answer,
            "retrieval_scores": [c["score"] for c in chunks],
            "eval_scores": scores,
            "latency_s": round(elapsed, 2),
        }
        results.append(result)

        if verbose:
            print(f"  Answer: {answer[:120]}...")
            for criterion, val in scores.items():
                print(f"  {criterion}: {val['score']:.2f} — {val['reason']}")
        else:
            score_summary = " | ".join(
                f"{k[:5]}={v['score']:.2f}" for k, v in scores.items()
            )
            print(f"  {score_summary}  ({elapsed:.1f}s)")

        print()

    # --- Aggregate ---
    criteria = ["retrieval_relevance", "faithfulness", "answer_completeness", "hallucination_free"]
    aggregates = {}
    for criterion in criteria:
        scores_list = [r["eval_scores"][criterion]["score"] for r in results]
        aggregates[criterion] = {
            "mean": round(sum(scores_list) / len(scores_list), 3),
            "min": min(scores_list),
            "max": max(scores_list),
        }

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    category_scores = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_means = {}
        for criterion in criteria:
            vals = [r["eval_scores"][criterion]["score"] for r in cat_results]
            cat_means[criterion] = round(sum(vals) / len(vals), 3)
        category_scores[cat] = cat_means

    summary = {
        "n_questions": len(questions),
        "top_k": top_k,
        "aggregate_scores": aggregates,
        "category_scores": category_scores,
        "results": results,
    }

    # Print summary
    print(f"{'='*60}")
    print("AGGREGATE SCORES")
    print(f"{'='*60}")
    for criterion, vals in aggregates.items():
        bar = "█" * int(vals["mean"] * 20) + "░" * (20 - int(vals["mean"] * 20))
        print(f"  {criterion:<25} {bar} {vals['mean']:.3f}  (min={vals['min']:.2f}, max={vals['max']:.2f})")

    print(f"\nPER-CATEGORY BREAKDOWN")
    for cat, scores in category_scores.items():
        scores_str = " | ".join(f"{k[:5]}={v:.2f}" for k, v in scores.items())
        print(f"  {cat:<15} {scores_str}")

    # Save results
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Thesis RAG system.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_evaluation(top_k=args.top_k, verbose=args.verbose)
