"""Sparse (BM25) retriever – loads pre-built index from disk."""

import pickle
import re
from pathlib import Path
from typing import Any

try:
    from underthesea import word_tokenize as _vi_tokenize
    _HAS_UNDERTHESEA = True
except ImportError:
    _HAS_UNDERTHESEA = False


def simple_tokenize(text: str) -> list[str]:
    if _HAS_UNDERTHESEA:
        return _vi_tokenize(text.lower(), format="text").split()
    text = text.lower()
    tokens = re.split(r"[^\w]+", text, flags=re.UNICODE)
    return [t for t in tokens if t]


class BM25Retriever:
    def __init__(self, cfg: dict):
        index_path = Path(cfg["bm25"]["index_path"])
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_path}. "
                "Run src/pipeline/03_bm25_index.py first."
            )
        with index_path.open("rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._records = payload["records"]

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        tokens = simple_tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # get indices of top_k scores (descending)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue
            doc = dict(self._records[idx])
            doc["_score_sparse"] = float(scores[idx])
            doc["_id"] = doc.get("chunk_id", str(idx))
            results.append(doc)
        return results
