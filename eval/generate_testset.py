"""
Synthetic test set generator for RAG evaluation.

Reads all chunks from data/processed/*.jsonl, uses LLM to generate diverse
QA pairs with ground-truth relevant_ids, then exports to eval/test_queries.json.

Query taxonomy (following RAGAS / RGB benchmark):
  - factoid     : "X là gì? / Điều kiện Y?" — single fact from one chunk
  - procedural  : "Thủ tục/quy trình làm Z?" — steps from one chunk
  - conditional : "Nếu sinh viên A thì được B không?" — one chunk
  - multi_hop   : requires combining 2 chunks from same or different docs

Parallelism:
  All LLM calls run concurrently via ThreadPoolExecutor (I/O-bound).
  --workers controls concurrency (default: 20).

Usage:
  python -m eval.generate_testset
  python -m eval.generate_testset --target 200 --workers 30 --out eval/test_queries.json
"""

import argparse
import json
import os
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from src.config import load_config

# ── Constants ──────────────────────────────────────────────────────────────────

QUERY_TYPES = ["factoid", "procedural", "conditional", "multi_hop"]

TYPE_DISTRIBUTION = {
    "factoid":     0.40,
    "procedural":  0.30,
    "conditional": 0.15,
    "multi_hop":   0.15,
}

MIN_CHUNK_LEN = 120
MAX_CHUNK_LEN = 3000
# Submit this many candidates per needed query to absorb LLM refusals
OVERSAMPLE = 2

# ── Prompts ────────────────────────────────────────────────────────────────────

SINGLE_CHUNK_PROMPT = """\
Bạn là chuyên gia tạo bộ câu hỏi đánh giá hệ thống RAG cho tài liệu quy chế sinh viên Đại học Quốc tế (ĐHQG-HCM).

Dưới đây là một đoạn văn bản từ tài liệu:
---
{chunk_text}
---
Nguồn: {source}, trang {page}

Hãy tạo MỘT câu hỏi loại "{query_type}" dựa trên đoạn văn trên.

Định nghĩa loại câu hỏi:
- factoid     : hỏi về một sự kiện, con số, điều kiện cụ thể ("X là gì?", "Điều kiện để Y?")
- procedural  : hỏi về quy trình, thủ tục, các bước thực hiện ("Cách/thủ tục để làm Z?")
- conditional : hỏi về kết quả phụ thuộc điều kiện ("Nếu sinh viên A thì được B không?")

Yêu cầu:
1. Câu hỏi phải trả lời được HOÀN TOÀN từ đoạn văn trên, không cần thêm nguồn khác.
2. Câu hỏi phải tự nhiên như sinh viên thật sự hỏi, không trích dẫn tên văn bản.
3. Câu trả lời tham chiếu phải trích dẫn trực tiếp từ đoạn văn.
4. Nếu đoạn văn không đủ thông tin để tạo câu hỏi loại "{query_type}", trả về null.

Trả về JSON (và CHỈ JSON):
{{
  "question": "...",
  "answer": "...",
  "type": "{query_type}",
  "is_answerable": true
}}
Hoặc nếu không tạo được: {{"is_answerable": false}}"""


MULTI_HOP_PROMPT = """\
Bạn là chuyên gia tạo bộ câu hỏi đánh giá hệ thống RAG cho tài liệu quy chế sinh viên Đại học Quốc tế (ĐHQG-HCM).

[Đoạn 1] Nguồn: {source1}, trang {page1}
---
{chunk_text1}
---

[Đoạn 2] Nguồn: {source2}, trang {page2}
---
{chunk_text2}
---

Hãy tạo MỘT câu hỏi multi-hop: câu hỏi mà để trả lời đầy đủ cần thông tin từ CẢ HAI đoạn trên.

Yêu cầu:
1. Câu hỏi PHẢI cần cả 2 đoạn để trả lời — không thể trả lời chỉ từ 1 đoạn.
2. Câu hỏi phải tự nhiên như sinh viên thật sự hỏi.
3. Câu trả lời phải tổng hợp thông tin từ cả 2 đoạn.
4. Nếu 2 đoạn không liên quan đủ, trả về null.

Trả về JSON (và CHỈ JSON):
{{
  "question": "...",
  "answer": "...",
  "type": "multi_hop",
  "is_answerable": true
}}
Hoặc nếu không tạo được: {{"is_answerable": false}}"""


PARAPHRASE_PROMPT = """\
Viết lại câu hỏi sau theo cách một sinh viên thật sự sẽ hỏi — dùng từ ngữ khác, \
đặc biệt tránh dùng lại từ kỹ thuật/văn phong của văn bản quy chế.
Giữ nguyên ý nghĩa. Chỉ trả về câu hỏi đã viết lại, không giải thích.

Câu hỏi gốc: {question}"""


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_chunks(processed_dir: str) -> list[dict]:
    chunks = []
    for jsonl_path in Path(processed_dir).glob("*.jsonl"):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if len(chunk.get("text", "")) >= MIN_CHUNK_LEN:
                    chunks.append(chunk)
    return chunks


def group_by_source(chunks: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        groups[c["source"]].append(c)
    return dict(groups)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(prompt: str, client: OpenAI, model: str, temperature: float = 0.7, max_tokens: int = 512) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception:
        return None


def _paraphrase(question: str, client: OpenAI, model: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.8,
            max_tokens=128,
            messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(question=question)}],
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten if len(rewritten) > 10 else question
    except Exception:
        return question


# ── Adjacent chunk discovery ──────────────────────────────────────────────────

def _get_relevant_ids(chunk: dict, pos_index: dict) -> list[str]:
    """
    Return chunk_id of the given chunk plus any immediately adjacent chunks
    (±1 chunk_index on the same page/source) that likely contain related content
    due to chunking boundaries splitting a single điều khoản.
    """
    source = chunk["source"]
    page = chunk["page"]
    idx = chunk.get("chunk_index", 0)
    ids = [chunk["chunk_id"]]
    for delta in (-1, 1):
        neighbor = pos_index.get((source, page, idx + delta))
        if neighbor:
            ids.append(neighbor)
    return ids


# ── Per-task functions (run inside thread pool) ───────────────────────────────

def _task_single(
    chunk: dict, query_type: str, client: OpenAI, model: str, pos_index: dict
) -> dict | None:
    prompt = SINGLE_CHUNK_PROMPT.format(
        chunk_text=chunk["text"][:MAX_CHUNK_LEN],
        source=chunk["source"],
        page=chunk.get("page", "?"),
        query_type=query_type,
    )
    result = _call_llm(prompt, client, model)
    if not result or not result.get("is_answerable"):
        return None
    question = _paraphrase(result["question"], client, model)
    return {
        "question": question,
        "question_raw": result["question"],
        "answer": result["answer"],
        "type": query_type,
        "relevant_ids": _get_relevant_ids(chunk, pos_index),
        "sources": [{"source": chunk["source"], "page": chunk.get("page")}],
    }


def _task_multi_hop(chunk1: dict, chunk2: dict, client: OpenAI, model: str) -> dict | None:
    prompt = MULTI_HOP_PROMPT.format(
        chunk_text1=chunk1["text"][:MAX_CHUNK_LEN // 2],
        source1=chunk1["source"],
        page1=chunk1.get("page", "?"),
        chunk_text2=chunk2["text"][:MAX_CHUNK_LEN // 2],
        source2=chunk2["source"],
        page2=chunk2.get("page", "?"),
    )
    result = _call_llm(prompt, client, model)
    if not result or not result.get("is_answerable"):
        return None
    question = _paraphrase(result["question"], client, model)
    return {
        "question": question,
        "question_raw": result["question"],
        "answer": result["answer"],
        "type": "multi_hop",
        "relevant_ids": [chunk1["chunk_id"], chunk2["chunk_id"]],
        "sources": [
            {"source": chunk1["source"], "page": chunk1.get("page")},
            {"source": chunk2["source"], "page": chunk2.get("page")},
        ],
    }


# ── Candidate sampling ─────────────────────────────────────────────────────────

def _sample_multi_hop_pairs(
    rng: random.Random,
    by_source: dict[str, list[dict]],
    all_sources: list[str],
    n: int,
) -> list[tuple[dict, dict]]:
    pairs = []
    for _ in range(n):
        if rng.random() < 0.5 and len(all_sources) > 1:
            src = rng.choice(all_sources)
            pool = by_source[src]
            if len(pool) < 2:
                c1 = rng.choice(pool)
                src2 = rng.choice([s for s in all_sources if s != src] or all_sources)
                c2 = rng.choice(by_source[src2])
            else:
                c1, c2 = rng.sample(pool, 2)
        else:
            if len(all_sources) >= 2:
                src1, src2 = rng.sample(all_sources, 2)
            else:
                src1 = src2 = all_sources[0]
            c1 = rng.choice(by_source[src1])
            c2 = rng.choice(by_source[src2])
        pairs.append((c1, c2))
    return pairs


# ── Deduplication ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def deduplicate(items: list[dict], prefix_len: int = 20) -> list[dict]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = _normalize(item["question"])[:prefix_len]
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ── Main generation (parallel) ─────────────────────────────────────────────────

def generate_testset(
    chunks: list[dict],
    target: int,
    client: OpenAI,
    model: str,
    seed: int = 42,
    max_workers: int = 20,
) -> list[dict]:
    rng = random.Random(seed)
    by_source = group_by_source(chunks)
    all_sources = list(by_source.keys())

    # Build position index for adjacent chunk lookup: (source, page, chunk_index) → chunk_id
    pos_index: dict[tuple, str] = {
        (c["source"], c["page"], c.get("chunk_index", 0)): c["chunk_id"]
        for c in chunks
    }

    type_targets = {t: max(1, int(target * frac)) for t, frac in TYPE_DISTRIBUTION.items()}
    diff = target - sum(type_targets.values())
    type_targets["factoid"] += diff

    shuffled = list(chunks)
    rng.shuffle(shuffled)

    print(f"Corpus  : {len(chunks)} chunks across {len(all_sources)} documents")
    print(f"Target  : {target} queries  |  workers: {max_workers}")
    print(f"Per type: {type_targets}\n")

    # ── Build all tasks upfront ──────────────────────────────────────────────
    tasks: list[tuple] = []   # (future_key, query_type)

    for query_type in ["factoid", "procedural", "conditional"]:
        needed = type_targets[query_type]
        candidates = shuffled[: needed * OVERSAMPLE]
        for chunk in candidates:
            tasks.append(("single", query_type, chunk))

    needed_mh = type_targets["multi_hop"]
    mh_pairs = _sample_multi_hop_pairs(rng, by_source, all_sources, needed_mh * OVERSAMPLE)
    for c1, c2 in mh_pairs:
        tasks.append(("multi_hop", "multi_hop", c1, c2))

    total_tasks = len(tasks)
    print(f"Submitting {total_tasks} tasks to thread pool …\n")

    # ── Run in parallel ──────────────────────────────────────────────────────
    collected: dict[str, list[dict]] = {t: [] for t in QUERY_TYPES}
    lock = threading.Lock()
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_type: dict = {}

        for task in tasks:
            if task[0] == "single":
                _, query_type, chunk = task
                fut = executor.submit(_task_single, chunk, query_type, client, model, pos_index)
            else:
                _, _, c1, c2 = task
                fut = executor.submit(_task_multi_hop, c1, c2, client, model)
                query_type = "multi_hop"
            future_to_type[fut] = query_type

        for fut in as_completed(future_to_type):
            query_type = future_to_type[fut]
            result = fut.result()

            with lock:
                done_count += 1
                if result:
                    collected[query_type].append(result)
                if done_count % 20 == 0 or done_count == total_tasks:
                    counts = {t: len(v) for t, v in collected.items()}
                    print(f"  [{done_count}/{total_tasks}] collected: {counts}")

    # ── Trim each type to target, merge, dedup ────────────────────────────────
    all_items: list[dict] = []
    for query_type in QUERY_TYPES:
        items = collected[query_type][: type_targets[query_type]]
        all_items.extend(items)

    all_items = deduplicate(all_items)
    rng.shuffle(all_items)
    return all_items[:target]


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RAG test set")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", type=int, default=150)
    parser.add_argument("--workers", type=int, default=20,
                        help="ThreadPoolExecutor concurrency (default: 20)")
    parser.add_argument("--out", default="eval/test_queries.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    chunks = load_all_chunks(cfg["data"]["processed_dir"])
    if not chunks:
        raise RuntimeError(f"No chunks found in {cfg['data']['processed_dir']}. Run pipeline step 01 first.")

    testset = generate_testset(
        chunks=chunks,
        target=args.target,
        client=client,
        model=cfg["llm"]["model"],
        seed=args.seed,
        max_workers=args.workers,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)

    from collections import Counter
    type_counts = Counter(item["type"] for item in testset)
    source_counts = Counter(s["source"] for item in testset for s in item.get("sources", []))

    print("\n" + "=" * 60)
    print(f"Saved → {out_path}  ({len(testset)} queries)")
    print("\nBy type:")
    for t, n in sorted(type_counts.items()):
        print(f"  {t:<15}: {n}")
    print("\nTop sources:")
    for src, n in source_counts.most_common(8):
        print(f"  {src[:55]:<55}: {n}")


if __name__ == "__main__":
    main()
