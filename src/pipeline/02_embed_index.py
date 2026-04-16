"""
Step 02 – Embed chunks and upsert to Qdrant vector store.

Reads:  data/processed/*.jsonl
Writes: Qdrant collection (local or Cloud)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Generator

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def load_config(path: str = "config.yaml") -> dict:
    from src.config import load_config as _load
    return _load(path)


# ─── Embedding providers ───────────────────────────────────────────────────────

def get_embedder(cfg: dict):
    provider = cfg["embedding"]["provider"]
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = cfg["embedding"]["openai_model"]

        def embed(texts: list[str]) -> list[list[float]]:
            resp = client.embeddings.create(input=texts, model=model)
            return [d.embedding for d in resp.data]

    elif provider == "local":
        import torch
        from sentence_transformers import SentenceTransformer
        model_name = cfg["embedding"]["local_model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding device: {device}")
        model = SentenceTransformer(model_name, device=device)
        is_e5 = "e5" in model_name.lower()

        def embed(texts: list[str]) -> list[list[float]]:
            # multilingual-e5-* requires "passage: " prefix for documents
            prefixed = [f"passage: {t}" if is_e5 else t for t in texts]
            return model.encode(
                prefixed, batch_size=32, show_progress_bar=False, normalize_embeddings=True
            ).tolist()

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    return embed


# ─── Data loader ──────────────────────────────────────────────────────────────

def iter_chunks(processed_dir: Path) -> Generator[dict, None, None]:
    for jsonl in sorted(processed_dir.glob("*.jsonl")):
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# ─── Qdrant helpers ────────────────────────────────────────────────────────────

def get_qdrant_client(cfg: dict) -> QdrantClient:
    vs = cfg["vector_store"]
    api_key = vs.get("qdrant_api_key") or os.environ.get("QDRANT_API_KEY", "")
    url = vs["qdrant_url"]
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient, name: str, vector_size: int):
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        info = client.get_collection(name)
        existing_dim = info.config.params.vectors.size
        if existing_dim != vector_size:
            print(f"Collection '{name}' has dim={existing_dim}, expected {vector_size} – recreating.")
            client.delete_collection(name)
            existing.discard(name)
        else:
            print(f"Collection '{name}' already exists (dim={vector_size}) – upserting.")

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection '{name}' (dim={vector_size})")


# ─── Main ──────────────────────────────────────────────────────────────────────

def run(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    processed_dir = Path(cfg["data"]["processed_dir"])
    batch_size = cfg["embedding"]["batch_size"]
    collection = cfg["vector_store"]["collection_name"]
    vector_size = cfg["vector_store"]["vector_size"]

    embed = get_embedder(cfg)
    client = get_qdrant_client(cfg)
    ensure_collection(client, collection, vector_size)

    total = 0
    for batch in batched(iter_chunks(processed_dir), batch_size):
        texts = [r["text"] for r in batch]
        vectors = embed(texts)
        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload={k: v for k, v in rec.items() if k != "text"},
                # store text in payload too for retrieval
            )
            for i, (rec, vec) in enumerate(zip(batch, vectors), start=total)
        ]
        # inject text into payload
        for p, rec in zip(points, batch):
            p.payload["text"] = rec["text"]

        client.upsert(collection_name=collection, points=points)
        total += len(batch)
        print(f"  Upserted {total} vectors…", end="\r")

    print(f"\nDone. Total vectors in '{collection}': {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed & index chunks to Qdrant")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
