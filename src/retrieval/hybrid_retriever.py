"""Hybrid retriever: RRF fusion of dense + sparse results."""

from typing import Any

from .bm25_retriever import BM25Retriever
from .vector_store import VectorRetriever


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    id_key: str = "_id",
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for ranked_list in result_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc[id_key]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            docs[doc_id] = doc

    merged = sorted(docs.values(), key=lambda d: scores[d[id_key]], reverse=True)
    for doc in merged:
        doc["_score_rrf"] = scores[doc[id_key]]
    return merged


class HybridRetriever:
    def __init__(self, cfg: dict):
        self._dense = VectorRetriever(cfg)
        self._sparse = BM25Retriever(cfg)
        self._top_k_dense = cfg["retrieval"]["top_k_dense"]
        self._top_k_sparse = cfg["retrieval"]["top_k_sparse"]
        self._top_k_fusion = cfg["retrieval"]["top_k_fusion"]
        self._rrf_k = cfg["retrieval"]["rrf_k"]

    def search(self, query: str) -> list[dict[str, Any]]:
        dense_results = self._dense.search(query, top_k=self._top_k_dense)
        sparse_results = self._sparse.search(query, top_k=self._top_k_sparse)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            id_key="_id",
            k=self._rrf_k,
        )
        return fused[: self._top_k_fusion]
