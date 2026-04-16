"""Dense retriever using Qdrant."""

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter


class VectorRetriever:
    def __init__(self, cfg: dict):
        vs = cfg["vector_store"]
        api_key = vs.get("qdrant_api_key") or os.environ.get("QDRANT_API_KEY", "")
        url = vs["qdrant_url"]
        self.client = QdrantClient(url=url, api_key=api_key if api_key else None)
        self.collection = vs["collection_name"]

        emb_cfg = cfg["embedding"]
        self._provider = emb_cfg["provider"]
        if self._provider == "openai":
            from openai import OpenAI
            self._oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self._model = emb_cfg["openai_model"]
            self._embed = self._embed_openai
        else:
            from sentence_transformers import SentenceTransformer
            self._local_model_name = emb_cfg["local_model"]
            self._st = SentenceTransformer(self._local_model_name)
            self._embed = self._embed_local

    def _embed_openai(self, text: str) -> list[float]:
        resp = self._oai.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding

    def _embed_local(self, text: str) -> list[float]:
        # multilingual-e5-* models require "query: " prefix for queries
        if "e5" in self._local_model_name.lower():
            text = f"query: {text}"
        return self._st.encode([text], normalize_embeddings=True)[0].tolist()

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        vector = self._embed(query)
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for hit in response.points:
            doc = dict(hit.payload)
            doc["_score_dense"] = hit.score
            # Use chunk_id from payload so RRF can deduplicate with BM25 results
            doc["_id"] = doc.get("chunk_id") or str(hit.id)
            results.append(doc)
        return results
