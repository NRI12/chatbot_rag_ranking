"""
End-to-end evaluator: runs test queries through the RAG pipeline
and computes all retrieval + generation metrics.

Parallelism strategy (2-phase):
  Phase 1 — RAG pipeline  : ThreadPool(rag_workers=8)
    Limited because retrieval uses local embedding + cross-encoder models.
  Phase 2 — LLM judge     : ThreadPool(judge_workers=30)
    All judge calls are pure OpenAI I/O — safe to fan out aggressively.

Usage:
  python -m src.evaluation.evaluator
  python -m src.evaluation.evaluator --config config.yaml --rag-workers 4 --judge-workers 30
"""

import argparse
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.chat.rag_chain import RAGChain
from src.evaluation.metrics import (
    aggregate_judge_scores,
    build_judge_tasks,
    evaluate_retrieval,
    _llm_score,
)


def load_test_queries(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Phase 1: RAG pipeline ─────────────────────────────────────────────────────

def _run_rag(args: tuple) -> dict:
    """Worker: run one query through the RAG chain. Returns raw result dict."""
    idx, question, relevant_ids, chain = args
    result = chain.query(question)
    return {
        "idx": idx,
        "question": question,
        "relevant_ids": relevant_ids,
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieved_ids": [s["chunk_id"] for s in result["sources"] if s.get("chunk_id")],
    }


def run_rag_phase(
    test_queries: list[dict],
    chain: RAGChain,
    max_workers: int,
) -> list[dict]:
    """Run all queries through RAG in parallel, return results ordered by idx."""
    tasks = [
        (i, tq["question"], tq.get("relevant_ids", []), chain)
        for i, tq in enumerate(test_queries)
    ]
    results = [None] * len(tasks)
    done = 0
    lock = threading.Lock()

    print(f"Phase 1 — RAG pipeline  ({len(tasks)} queries, {max_workers} workers)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_rag, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results[res["idx"]] = res
            with lock:
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] RAG done")

    return results


# ── Phase 2: LLM judge ────────────────────────────────────────────────────────

def run_judge_phase(
    rag_results: list[dict],
    client,
    model: str,
    max_workers: int,
) -> list[dict[str, float]]:
    """
    Collect ALL judge prompts from every query, run them all concurrently,
    then reassemble per-query metric dicts.
    """
    # Build flat task list: (query_idx, task_key, prompt)
    all_tasks: list[tuple[int, str, str]] = []
    for res in rag_results:
        context_docs = [s["text"] for s in res["sources"]]
        tasks = build_judge_tasks(res["question"], res["answer"], context_docs)
        for key, prompt in tasks:
            all_tasks.append((res["idx"], key, prompt))

    total = len(all_tasks)
    print(f"\nPhase 2 — LLM judge     ({total} calls, {max_workers} workers)")

    # scores[query_idx][task_key] = score
    scores: dict[int, dict[str, float]] = defaultdict(dict)
    done = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_meta = {
            executor.submit(_llm_score, prompt, client, model): (idx, key)
            for idx, key, prompt in all_tasks
        }
        for fut in as_completed(future_meta):
            idx, key = future_meta[fut]
            score = fut.result()
            with lock:
                scores[idx][key] = score
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  [{done}/{total}] judge done")

    # Aggregate per query
    gen_metrics_list = []
    for res in rag_results:
        n_docs = len(res["sources"])
        gen_metrics_list.append(aggregate_judge_scores(scores[res["idx"]], n_docs))

    return gen_metrics_list


# ── Aggregation ───────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(
    config_path: str = "config.yaml",
    rag_workers: int = 8,
    judge_workers: int = 30,
):
    cfg = load_config(config_path)
    eval_cfg = cfg["evaluation"]
    k_values = eval_cfg["k_values"]
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    test_queries = load_test_queries(eval_cfg["test_queries_path"])
    chain = RAGChain(config_path)

    from openai import OpenAI
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge_model = cfg["llm"]["model"]

    print(f"Evaluating {len(test_queries)} queries\n" + "=" * 60)

    # ── Phase 1 ──
    rag_results = run_rag_phase(test_queries, chain, rag_workers)

    # ── Phase 2 ──
    gen_metrics_list = run_judge_phase(rag_results, oai, judge_model, judge_workers)

    # ── Build per-query output ──
    per_query_results = []
    retrieval_inputs = []

    for res, gen_metrics in zip(rag_results, gen_metrics_list):
        per_query_results.append({
            "question":     res["question"],
            "answer":       res["answer"],
            "sources":      res["sources"],
            "retrieved_ids": res["retrieved_ids"],
            "relevant_ids": res["relevant_ids"],
            **gen_metrics,
        })
        if res["relevant_ids"]:
            retrieval_inputs.append({
                "query":         res["question"],
                "relevant_ids":  res["relevant_ids"],
                "retrieved_ids": res["retrieved_ids"],
            })

    # ── Aggregate ──
    retrieval_agg = evaluate_retrieval(retrieval_inputs, k_values) if retrieval_inputs else {}
    gen_agg = {
        key: _mean([m[key] for m in gen_metrics_list])
        for key in gen_metrics_list[0]
    } if gen_metrics_list else {}

    summary = {
        "timestamp":          datetime.now().isoformat(),
        "n_queries":          len(test_queries),
        "retrieval_metrics":  retrieval_agg,
        "generation_metrics": gen_agg,
    }

    # ── Save ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with (output_dir / f"summary_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (output_dir / f"details_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump(per_query_results, f, ensure_ascii=False, indent=2)

    # ── Print ──
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if retrieval_agg:
        print("\n── Retrieval Metrics ──")
        for metric, val in sorted(retrieval_agg.items()):
            print(f"  {metric:<20}: {val:.4f}")
    if gen_agg:
        print("\n── Generation Metrics (LLM-as-judge) ──")
        for metric, val in gen_agg.items():
            print(f"  {metric:<20}: {val:.4f}")
    print(f"\nResults saved to: {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rag-workers", type=int, default=8,
                        help="Parallel workers for RAG pipeline (default: 8)")
    parser.add_argument("--judge-workers", type=int, default=30,
                        help="Parallel workers for LLM judge calls (default: 30)")
    args = parser.parse_args()
    run_evaluation(args.config, args.rag_workers, args.judge_workers)
