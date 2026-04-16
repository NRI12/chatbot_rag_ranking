"""
Step 03 – Build BM25 index from processed JSONL chunks.

Writes: data/processed/bm25.pkl  (corpus list + BM25Okapi object)
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

try:
    from underthesea import word_tokenize as _vi_tokenize
    _HAS_UNDERTHESEA = True
except ImportError:
    _HAS_UNDERTHESEA = False


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simple_tokenize(text: str) -> list[str]:
    """Vietnamese-aware tokenizer using underthesea if available, else char-split fallback."""
    if _HAS_UNDERTHESEA:
        return _vi_tokenize(text.lower(), format="text").split()
    text = text.lower()
    tokens = re.split(r"[^\w]+", text, flags=re.UNICODE)
    return [t for t in tokens if t]


def load_chunks(processed_dir: Path) -> list[dict]:
    records = []
    for jsonl in sorted(processed_dir.glob("*.jsonl")):
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def run(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    processed_dir = Path(cfg["data"]["processed_dir"])
    index_path = Path(cfg["bm25"]["index_path"])
    index_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading chunks…")
    records = load_chunks(processed_dir)
    print(f"  {len(records)} chunks loaded.")

    print("Tokenizing…")
    tokenized = [simple_tokenize(r["text"]) for r in records]

    print("Building BM25 index…")
    bm25 = BM25Okapi(tokenized, k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])

    payload = {"records": records, "tokenized": tokenized, "bm25": bm25}
    with index_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"BM25 index saved → {index_path}")
    print(f"Corpus size: {len(records)} documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 keyword index")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
