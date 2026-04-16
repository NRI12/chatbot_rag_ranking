"""
FastAPI backend for the RAG chatbot.

Endpoints:
  POST /chat          – full RAG query
  POST /retrieve      – retrieve only (no LLM generation)
  POST /evaluate      – run evaluation suite
  GET  /health        – health check
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chat.rag_chain import RAGChain
from src.evaluation.evaluator import run_evaluation

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")

app = FastAPI(
    title="RAG Chatbot API",
    description="Hybrid search + reranking RAG over Vietnamese university regulations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded chain
_chain: Optional[RAGChain] = None


def get_chain() -> RAGChain:
    global _chain
    if _chain is None:
        _chain = RAGChain(CONFIG_PATH)
    return _chain


# ─── Schemas ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None  # override config if provided


class RetrieveRequest(BaseModel):
    query: str


class SourceDoc(BaseModel):
    chunk_id: Optional[str]
    source: Optional[str]
    page: Optional[int]
    text: str
    score_rrf: Optional[float]
    score_rerank: Optional[float]


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDoc]


class RetrieveResponse(BaseModel):
    query: str
    results: list[SourceDoc]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        chain = get_chain()
        result = chain.query(req.question)
        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[SourceDoc(**s) for s in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    try:
        chain = get_chain()
        docs = chain.retrieve(req.query)
        sources = [
            SourceDoc(
                chunk_id=d.get("chunk_id"),
                source=d.get("source"),
                page=d.get("page"),
                text=d.get("text", "")[:500],
                score_rrf=d.get("_score_rrf"),
                score_rerank=d.get("_score_rerank"),
            )
            for d in docs
        ]
        return RetrieveResponse(query=req.query, results=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate():
    """
    Run full evaluation suite.
    Offloads the blocking ThreadPoolExecutor work to a separate thread
    so it never blocks uvicorn's event loop.
    """
    try:
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, run_evaluation, CONFIG_PATH)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    uvicorn.run(
        "api.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=True,
    )
