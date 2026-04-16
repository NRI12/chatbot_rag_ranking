"""RAG chain: retrieval → rerank → LLM generation."""

import os
from typing import Any

from src.config import load_config
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


def build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.get("source", "unknown")
        page = doc.get("page", "?")
        text = doc.get("text", "")
        parts.append(f"[{i}] Nguồn: {source} (trang {page})\n{text}")
    return "\n\n---\n\n".join(parts)


class RAGChain:
    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.retriever = HybridRetriever(self.cfg)
        self.reranker = Reranker(self.cfg)

        llm_cfg = self.cfg["llm"]
        self.system_prompt = llm_cfg["system_prompt"]
        self.model = llm_cfg["model"]
        self.temperature = llm_cfg["temperature"]
        self.max_tokens = llm_cfg["max_tokens"]

        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve and rerank documents for a query."""
        candidates = self.retriever.search(query)
        reranked = self.reranker.rerank(query, candidates)
        return reranked

    def generate(self, query: str, docs: list[dict]) -> str:
        """Generate answer given query and retrieved docs."""
        if not docs:
            return "Tôi không tìm thấy thông tin này trong tài liệu hiện có."

        context = build_context(docs)
        user_msg = (
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {query}\n\n"
            f"Hướng dẫn trả lời:\n"
            f"- Dựa hoàn toàn vào ngữ cảnh được cung cấp để trả lời.\n"
            f"- Nếu ngữ cảnh có thông tin liên quan (dù chỉ một phần), hãy trả lời dựa trên thông tin đó và ghi rõ nguồn.\n"
            f"- Chỉ trả lời \"Tôi không tìm thấy thông tin này trong tài liệu hiện có.\" khi ngữ cảnh hoàn toàn không có thông tin liên quan đến câu hỏi.\n"
            f"- Tuyệt đối không thêm thông tin ngoài ngữ cảnh."
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()

    def query(self, question: str) -> dict[str, Any]:
        """Full RAG pipeline: retrieve → rerank → generate."""
        docs = self.retrieve(question)
        answer = self.generate(question, docs)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "chunk_id": d.get("chunk_id"),
                    "source": d.get("source"),
                    "page": d.get("page"),
                    "text": d.get("text", "")[:800],
                    "score_rrf": d.get("_score_rrf"),
                    "score_rerank": d.get("_score_rerank"),
                }
                for d in docs
            ],
        }
