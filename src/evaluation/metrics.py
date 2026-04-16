"""
Retrieval & RAG evaluation metrics.

Retrieval metrics (require ground-truth relevant chunk IDs):
  - Precision@k
  - Recall@k
  - F1@k
  - Hit Rate@k   (= 1 if any relevant doc in top-k)
  - MRR@k        (Mean Reciprocal Rank)
  - MAP@k        (Mean Average Precision)
  - NDCG@k       (Normalized Discounted Cumulative Gain)

RAG generation metrics (LLM-as-judge, no ground truth needed):
  - Faithfulness      – answer grounded in context?
  - Answer Relevance  – answer addresses the question?
  - Context Precision – are retrieved docs relevant?
"""

import math
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval metrics (single query)
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    hits = 0
    score = 0.0
    for rank, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            hits += 1
            score += hits / rank
    if not relevant:
        return 0.0
    return score / min(len(relevant), k)


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, r in enumerate(retrieved[:k], start=1):
        rel = 1.0 if r in relevant else 0.0
        dcg += rel / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    actual_dcg = dcg_at_k(retrieved, relevant, k)
    ideal_retrieved = list(relevant)[:k]
    ideal_dcg = dcg_at_k(ideal_retrieved, relevant, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate over multiple queries
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_retrieval(
    queries: list[dict],
    k_values: list[int] = (1, 3, 5, 10),
) -> dict:
    """
    queries: list of {
        "query": str,
        "relevant_ids": list[str],   # ground-truth chunk IDs
        "retrieved_ids": list[str],  # retrieved chunk IDs (ranked)
    }
    Returns dict of metric_name → mean score.
    """
    results: dict[str, list[float]] = {
        f"precision@{k}": [] for k in k_values
    }
    for k in k_values:
        results[f"recall@{k}"] = []
        results[f"f1@{k}"] = []
        results[f"hit_rate@{k}"] = []
        results[f"mrr@{k}"] = []
        results[f"map@{k}"] = []
        results[f"ndcg@{k}"] = []

    for q in queries:
        retrieved = q["retrieved_ids"]
        relevant = set(q["relevant_ids"])
        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            results[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            results[f"f1@{k}"].append(f1_at_k(retrieved, relevant, k))
            results[f"hit_rate@{k}"].append(hit_rate_at_k(retrieved, relevant, k))
            results[f"mrr@{k}"].append(reciprocal_rank(retrieved, relevant, k))
            results[f"map@{k}"].append(average_precision_at_k(retrieved, relevant, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))

    return {metric: sum(vals) / len(vals) for metric, vals in results.items() if vals}


# ══════════════════════════════════════════════════════════════════════════════
# LLM-as-judge generation metrics
# ══════════════════════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """\
Given the context below and an answer, rate how faithfully the answer is grounded in the context.
Score 0.0 (completely unfaithful) to 1.0 (fully supported by context).
Output ONLY a float number.

Context:
{context}

Answer:
{answer}

Score:"""

ANSWER_RELEVANCE_PROMPT = """\
Given the question and an answer, rate how relevant and complete the answer is.
Score 0.0 (irrelevant) to 1.0 (perfectly relevant and complete).
Output ONLY a float number.

Question:
{question}

Answer:
{answer}

Score:"""

CONTEXT_PRECISION_PROMPT = """\
Given the question and a retrieved context chunk, rate how relevant this chunk is to answering the question.
Score 0.0 (not relevant) to 1.0 (highly relevant).
Output ONLY a float number.

Question:
{question}

Context chunk:
{chunk}

Score:"""


def _llm_score(prompt: str, client, model: str) -> float:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return 0.0
    except Exception:
        # API error (rate limit, timeout, etc.) — return neutral score, don't crash
        return 0.0


# ── Batch-friendly judge interface ────────────────────────────────────────────

def build_judge_tasks(
    question: str,
    answer: str,
    context_docs: list[str],
) -> list[tuple[str, str]]:
    """
    Return a flat list of (task_key, prompt) for every judge call needed.
    Keys: "faithfulness", "answer_relevance", "cp_0", "cp_1", ...
    All tasks are independent — safe to run concurrently.
    """
    context_text = "\n\n".join(context_docs)
    tasks = [
        ("faithfulness",    FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)),
        ("answer_relevance", ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)),
    ]
    for i, chunk in enumerate(context_docs):
        tasks.append((f"cp_{i}", CONTEXT_PRECISION_PROMPT.format(question=question, chunk=chunk)))
    return tasks


def aggregate_judge_scores(scores: dict[str, float], n_docs: int) -> dict[str, float]:
    """Combine raw per-task scores into final generation metrics dict."""
    cp_scores = [scores[f"cp_{i}"] for i in range(n_docs) if f"cp_{i}" in scores]
    return {
        "faithfulness":      scores.get("faithfulness", 0.0),
        "answer_relevance":  scores.get("answer_relevance", 0.0),
        "context_precision": sum(cp_scores) / len(cp_scores) if cp_scores else 0.0,
    }


def compute_generation_metrics(
    question: str,
    answer: str,
    context_docs: list[str],
    client,
    model: str = "gpt-4o-mini",
) -> dict[str, float]:
    """Sequential version — kept for single-query use or testing."""
    tasks = build_judge_tasks(question, answer, context_docs)
    scores = {key: _llm_score(prompt, client, model) for key, prompt in tasks}
    return aggregate_judge_scores(scores, len(context_docs))
