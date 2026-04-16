"""
Experiment: sweep retrieval top_k values and compare retrieval metrics.

Tests combinations of:
  - retrieval_k  : top_k_dense / top_k_sparse / top_k_fusion  (all set equally)
  - reranker_k   : reranking.top_k

Skips LLM judge by default (expensive). Use --full-eval to include generation metrics.

Usage:
  python -m eval.experiment_k
  python -m eval.experiment_k --retrieval-ks 10 20 30 50 --reranker-ks 3 4 5
  python -m eval.experiment_k --full-eval
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.chat.rag_chain import RAGChain
from src.evaluation.metrics import evaluate_retrieval, build_judge_tasks, _llm_score, aggregate_judge_scores


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_queries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _patch_k(chain: RAGChain, retrieval_k: int, reranker_k: int):
    """Patch k values on existing chain — no model reload, no GPU allocation."""
    chain.retriever._top_k_dense = retrieval_k
    chain.retriever._top_k_sparse = retrieval_k
    chain.retriever._top_k_fusion = retrieval_k
    chain.reranker.top_k = reranker_k


# ── RAG phase (retrieval only) ────────────────────────────────────────────────

def run_rag_phase(test_queries: list[dict], chain: RAGChain, workers: int) -> list[dict]:
    def _run(args):
        idx, question, relevant_ids = args
        result = chain.query(question)
        return {
            "idx": idx,
            "question": question,
            "relevant_ids": relevant_ids,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_ids": [s["chunk_id"] for s in result["sources"] if s.get("chunk_id")],
        }

    tasks = [(i, tq["question"], tq.get("relevant_ids", [])) for i, tq in enumerate(test_queries)]
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results[res["idx"]] = res
    return results


# ── LLM judge phase ───────────────────────────────────────────────────────────

def run_judge_phase(rag_results: list[dict], client, model: str, workers: int) -> list[dict]:
    all_tasks = []
    for res in rag_results:
        context_docs = [s["text"] for s in res["sources"]]
        for key, prompt in build_judge_tasks(res["question"], res["answer"], context_docs):
            all_tasks.append((res["idx"], key, prompt))

    scores: dict[int, dict] = {res["idx"]: {} for res in rag_results}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_meta = {
            executor.submit(_llm_score, prompt, client, model): (idx, key)
            for idx, key, prompt in all_tasks
        }
        for fut in as_completed(future_meta):
            idx, key = future_meta[fut]
            with lock:
                scores[idx][key] = fut.result()

    return [
        aggregate_judge_scores(scores[res["idx"]], len(res["sources"]))
        for res in rag_results
    ]


# ── Single experiment run ─────────────────────────────────────────────────────

def run_experiment(
    chain: RAGChain,
    base_cfg: dict,
    test_queries: list[dict],
    retrieval_k: int,
    reranker_k: int,
    rag_workers: int,
    full_eval: bool,
    oai_client=None,
    judge_model: str = "gpt-4o-mini",
    judge_workers: int = 30,
) -> dict:
    # Patch k values in-place — reuses loaded models, zero extra GPU memory
    _patch_k(chain, retrieval_k, reranker_k)

    rag_results = run_rag_phase(test_queries, chain, rag_workers)

    # Retrieval metrics
    retrieval_inputs = [
        {"query": r["question"], "relevant_ids": r["relevant_ids"], "retrieved_ids": r["retrieved_ids"]}
        for r in rag_results if r["relevant_ids"]
    ]
    k_values = base_cfg["evaluation"]["k_values"]
    retrieval_agg = evaluate_retrieval(retrieval_inputs, k_values)

    result = {
        "retrieval_k": retrieval_k,
        "reranker_k": reranker_k,
        "retrieval_metrics": retrieval_agg,
    }

    # Generation metrics (optional)
    if full_eval and oai_client:
        gen_metrics_list = run_judge_phase(rag_results, oai_client, judge_model, judge_workers)
        result["generation_metrics"] = {
            key: _mean([m[key] for m in gen_metrics_list])
            for key in gen_metrics_list[0]
        }

    return result


# ── Print table ───────────────────────────────────────────────────────────────

def print_table(results: list[dict], full_eval: bool):
    key_metrics = ["hit_rate@1", "hit_rate@3", "hit_rate@5", "mrr@3", "ndcg@3"]
    gen_metrics = ["faithfulness", "answer_relevance", "context_precision"]

    header_cols = ["ret_k", "rnk_k"] + key_metrics
    if full_eval:
        header_cols += gen_metrics

    col_w = 10
    header = "  ".join(f"{c:>{col_w}}" for c in header_cols)
    print("\n" + "=" * len(header))
    print("EXPERIMENT RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    best_hit3 = max(r["retrieval_metrics"].get("hit_rate@3", 0) for r in results)

    for r in results:
        rm = r["retrieval_metrics"]
        row = [str(r["retrieval_k"]), str(r["reranker_k"])]
        for m in key_metrics:
            val = rm.get(m, 0)
            marker = " *" if m == "hit_rate@3" and abs(val - best_hit3) < 1e-6 else ""
            row.append(f"{val:.4f}{marker}")
        if full_eval and "generation_metrics" in r:
            gm = r["generation_metrics"]
            for m in gen_metrics:
                row.append(f"{gm.get(m, 0):.4f}")
        print("  ".join(f"{c:>{col_w}}" for c in row))

    print("=" * len(header))
    print("* = best hit_rate@3\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep retrieval k values")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--retrieval-ks", type=int, nargs="+", default=[10, 15, 20, 30, 50],
                        help="Values for top_k_dense/sparse/fusion (default: 10 15 20 30 50)")
    parser.add_argument("--reranker-ks", type=int, nargs="+", default=[4],
                        help="Values for reranker top_k (default: 4)")
    parser.add_argument("--rag-workers", type=int, default=8)
    parser.add_argument("--judge-workers", type=int, default=30)
    parser.add_argument("--full-eval", action="store_true",
                        help="Also run LLM judge for generation metrics (slow, costs money)")
    parser.add_argument("--out", default="eval/result/k_experiment.json")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    test_queries = load_test_queries(base_cfg["evaluation"]["test_queries_path"])

    oai_client = None
    if args.full_eval:
        from openai import OpenAI
        oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    combos = [
        (rk, rnk)
        for rk in args.retrieval_ks
        for rnk in args.reranker_ks
    ]

    print(f"Test queries : {len(test_queries)}")
    print(f"Experiments  : {len(combos)}  (retrieval_k × reranker_k)")
    print(f"Full eval    : {args.full_eval}")
    print(f"Combos       : {combos}")
    print("Loading RAG chain once (models stay in GPU across all runs)...\n")

    # Load models ONCE — all experiments reuse the same GPU-loaded models
    chain = RAGChain(args.config)

    all_results = []
    for i, (rk, rnk) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] retrieval_k={rk}  reranker_k={rnk} ...", flush=True)
        result = run_experiment(
            chain=chain,
            base_cfg=base_cfg,
            test_queries=test_queries,
            retrieval_k=rk,
            reranker_k=rnk,
            rag_workers=args.rag_workers,
            full_eval=args.full_eval,
            oai_client=oai_client,
            judge_model=base_cfg["llm"]["model"],
            judge_workers=args.judge_workers,
        )
        all_results.append(result)
        rm = result["retrieval_metrics"]
        print(f"       hit@1={rm.get('hit_rate@1', 0):.3f}  hit@3={rm.get('hit_rate@3', 0):.3f}  mrr@3={rm.get('mrr@3', 0):.3f}")

    print_table(all_results, args.full_eval)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_queries": len(test_queries),
            "experiments": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
