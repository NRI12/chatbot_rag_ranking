"""Reranker: Cohere API or local cross-encoder, with dedup + source diversity."""

import os
from typing import Any


def _dedup(docs: list[dict]) -> list[dict]:
    """Remove duplicates by chunk_id, then by text prefix."""
    seen_ids, seen_texts, out = set(), set(), []
    for d in docs:
        cid = d.get("chunk_id") or d.get("_id")
        text_key = d.get("text", "")[:80]
        if cid and cid in seen_ids:
            continue
        if text_key in seen_texts:
            continue
        if cid:
            seen_ids.add(cid)
        seen_texts.add(text_key)
        out.append(d)
    return out


def _diversify(scored: list[tuple[float, dict]], top_k: int, max_per_source: int) -> list[dict]:
    """Pick top_k docs while capping how many come from the same source file."""
    source_count: dict[str, int] = {}
    result = []
    # First pass: pick within cap
    for score, doc in scored:
        src = doc.get("source", "")
        if source_count.get(src, 0) < max_per_source:
            doc = dict(doc)
            doc["_score_rerank"] = float(score)
            result.append(doc)
            source_count[src] = source_count.get(src, 0) + 1
        if len(result) == top_k:
            return result
    # Second pass: fill remaining slots ignoring cap
    for score, doc in scored:
        if len(result) == top_k:
            break
        if doc not in [r for r in result]:
            doc = dict(doc)
            doc["_score_rerank"] = float(score)
            result.append(doc)
    return result


class Reranker:
    def __init__(self, cfg: dict):
        rr = cfg["reranking"]
        self.enabled = rr["enabled"]
        self.top_k = rr["top_k"]
        self.provider = rr["provider"]
        self.max_per_source = rr.get("max_per_source", 2)

        if not self.enabled:
            return

        if self.provider == "cohere":
            import cohere
            self._co = cohere.Client(os.environ["COHERE_API_KEY"])
            self._model = rr["cohere_model"]
        elif self.provider == "cross-encoder":
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                rr["cross_encoder_model"],
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unknown reranker provider: {self.provider}")

    def rerank(self, query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled or not docs:
            return _dedup(docs)[: self.top_k]

        docs = _dedup(docs)
        texts = [d["text"] for d in docs]

        if self.provider == "cohere":
            resp = self._co.rerank(
                query=query,
                documents=texts,
                model=self._model,
                top_n=self.top_k * 3,  # lấy nhiều hơn để diversity lọc
            )
            scored = [(r.relevance_score, docs[r.index]) for r in resp.results]

        elif self.provider == "cross-encoder":
            pairs = [[query, t] for t in texts]
            scores = self._model.predict(pairs).tolist()
            scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        else:
            return docs[: self.top_k]

        return _diversify(scored, self.top_k, self.max_per_source)
