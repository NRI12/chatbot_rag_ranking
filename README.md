# Chatbot RAG Ranking — Hỏi Đáp Quy Chế Trường Đại Học Quốc Tế (ĐHQG-HCM)

Hệ thống chatbot hỏi đáp thông minh dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)** với **Hybrid Search** và **Reranking**, chuyên biệt cho việc tra cứu quy chế, quy định của Trường Đại học Quốc tế – ĐHQG-HCM.

---

## Mục Lục

- [Tổng Quan](#tổng-quan)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Luồng Chạy (Data Flow)](#luồng-chạy-data-flow)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Công Nghệ Sử Dụng](#công-nghệ-sử-dụng)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Cấu Hình](#cấu-hình)
- [Chạy Pipeline Lập Chỉ Mục](#chạy-pipeline-lập-chỉ-mục)
- [Khởi Động Ứng Dụng](#khởi-động-ứng-dụng)
- [API Endpoints](#api-endpoints)
- [Giao Diện Web (Streamlit UI)](#giao-diện-web-streamlit-ui)
- [Đánh Giá Hệ Thống](#đánh-giá-hệ-thống)
- [Kết Quả Đánh Giá](#kết-quả-đánh-giá)
- [Tùy Chỉnh & Mở Rộng](#tùy-chỉnh--mở-rộng)

---

## Tổng Quan

Hệ thống trả lời câu hỏi của sinh viên liên quan đến:
- Quy chế học vụ, kỷ luật sinh viên
- Luật và quyết định của nhà trường
- Thông báo, phụ lục quy định

**Điểm nổi bật:**
- **Hybrid Search**: Kết hợp Dense Retrieval (vector embedding) + Sparse Retrieval (BM25) để tối đa độ bao phủ
- **Reciprocal Rank Fusion (RRF)**: Hợp nhất kết quả từ hai nguồn tìm kiếm mà không cần học tham số
- **Cross-Encoder Reranking**: Mô hình reranker tiếng Việt đặc biệt để sắp xếp lại kết quả chính xác hơn
- **Source Diversity**: Giới hạn tối đa 3 chunk từ cùng một tài liệu để tránh lặp lại
- **LLM-as-Judge Evaluation**: Sử dụng GPT-4o để tự động đánh giá chất lượng câu trả lời
- **Vietnamese-Aware Chunking**: Phân đoạn văn bản theo cấu trúc pháp lý (Điều, Chương, Mục)

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE LẬP CHỈ MỤC                    │
│                        (Chạy một lần)                           │
├───────────────┬────────────────────────┬────────────────────────┤
│  01_parse_    │   02_embed_index.py    │  03_bm25_index.py      │
│  chunk.py     │                        │                        │
│               │                        │                        │
│  PDF → Chunk  │  Chunk → Embedding     │  Chunk → BM25 Index    │
│  (PyMuPDF +   │  (vietnamese-bi-       │  (rank-bm25,           │
│   Legal regex)│   encoder, 768-dim)    │   underthesea)         │
│               │  → Qdrant collection   │  → bm25.pkl            │
└───────────────┴────────────────────────┴────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      LUỒNG XỬ LÝ CÂU HỎI                       │
│                      (Thời gian thực)                           │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ├──────────────────────────────────────────┐              │
│      │                                          │              │
│      ▼                                          ▼              │
│  ┌──────────────┐                    ┌──────────────────┐      │
│  │ Dense Search │                    │  Sparse Search   │      │
│  │  (Qdrant)    │                    │    (BM25)        │      │
│  │              │                    │                  │      │
│  │ Embed query  │                    │ Tokenize query   │      │
│  │ → Cosine     │                    │ → BM25 scoring   │      │
│  │   similarity │                    │                  │      │
│  │ → top_k=20   │                    │ → top_k=20       │      │
│  └──────┬───────┘                    └────────┬─────────┘      │
│         │                                     │                │
│         └──────────────┬────────────────────--┘                │
│                        │                                        │
│                        ▼                                        │
│              ┌──────────────────┐                              │
│              │  RRF Fusion      │                              │
│              │ (Reciprocal Rank │                              │
│              │  Fusion, k=60)   │                              │
│              │ → top_fusion=20  │                              │
│              └────────┬─────────┘                              │
│                       │                                         │
│                       ▼                                         │
│              ┌──────────────────┐                              │
│              │  Cross-Encoder   │                              │
│              │   Reranking      │                              │
│              │ (AITeamVN/       │                              │
│              │  Vietnamese_     │                              │
│              │  Reranker)       │                              │
│              │ + Dedup          │                              │
│              │ + Diversity (≤3  │                              │
│              │   per source)    │                              │
│              │ → top_k=4        │                              │
│              └────────┬─────────┘                              │
│                       │                                         │
│                       ▼                                         │
│              ┌──────────────────┐                              │
│              │  LLM Generation  │                              │
│              │  (GPT-4o)        │                              │
│              │                  │                              │
│              │ Context = top-4  │                              │
│              │ chunks formatted │                              │
│              │ → Vietnamese     │                              │
│              │   answer         │                              │
│              └────────┬─────────┘                              │
│                       │                                         │
│                       ▼                                         │
│              Answer + Sources + Scores                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Luồng Chạy (Data Flow)

### 1. Indexing Pipeline (Offline — chạy một lần)

```
data/raw/**/*.pdf
       │
       ▼
[01_parse_chunk.py]
  - Trích xuất văn bản bằng PyMuPDF
  - Phát hiện cấu trúc pháp lý: Điều X, Chương X, Mục X
  - Chia chunk theo cấu trúc (chunk_size=800, overlap=150)
  - Fallback: chia theo câu nếu chunk quá lớn
  - Output: data/processed/*.jsonl
    {chunk_id, source, page, chunk_index, text, source_path}
       │
       ├──────────────────────────────────┐
       │                                  │
       ▼                                  ▼
[02_embed_index.py]              [03_bm25_index.py]
  - Đọc tất cả JSONL files         - Tokenize text (underthesea)
  - Embed bằng vietnamese-          - Xây BM25Okapi index
    bi-encoder (768 chiều)            (k1=1.5, b=0.75)
  - Upsert vào Qdrant               - Lưu bm25.pkl
    collection "rag_docs"
```

### 2. Query Pipeline (Online — mỗi câu hỏi)

```
User Question
    │
    ├─── Dense: Embed → Tìm cosine similarity trong Qdrant → 20 kết quả
    │
    ├─── Sparse: Tokenize → BM25 scoring → 20 kết quả
    │
    ▼
RRF Fusion: Hợp nhất 2 danh sách → score(d) = Σ 1/(k + rank_i(d)) → 20 kết quả
    │
    ▼
Cross-Encoder Reranking:
    - Chấm điểm lại từng cặp (query, chunk)
    - Loại trùng lặp (chunk_id + tiền tố văn bản)
    - Giới hạn 3 chunk mỗi nguồn tài liệu
    - Giữ lại top 4
    │
    ▼
Context Assembly:
    "[1] Nguồn: file.pdf, trang X\nNội dung: ...\n\n[2] ..."
    │
    ▼
GPT-4o:
    System: Trả lời CHỈ dựa trên ngữ cảnh. Nếu không có thông tin, nói rõ.
    User: {question}\n\nNGỮ CẢNH:\n{context}
    │
    ▼
Response: {answer, sources: [{source, page, rrf_score, rerank_score}]}
```

---

## Cấu Trúc Dự Án

```
chatbot_rag_ranking/
├── api/                          # FastAPI backend
│   ├── __init__.py
│   └── main.py                  # REST API server
│
├── src/                          # Core logic
│   ├── config.py                # Đọc config.yaml + env vars
│   ├── chat/
│   │   └── rag_chain.py        # Điều phối toàn bộ RAG pipeline
│   ├── pipeline/
│   │   ├── 01_parse_chunk.py   # Parse PDF → chunks JSONL
│   │   ├── 02_embed_index.py   # Embed + index vào Qdrant
│   │   └── 03_bm25_index.py    # Xây BM25 sparse index
│   ├── retrieval/
│   │   ├── vector_store.py     # Dense retrieval (Qdrant)
│   │   ├── bm25_retriever.py   # Sparse retrieval (BM25)
│   │   ├── hybrid_retriever.py # RRF fusion
│   │   └── reranker.py         # Cross-encoder + dedup + diversity
│   └── evaluation/
│       ├── evaluator.py        # Pipeline đánh giá đa luồng
│       └── metrics.py          # Precision/Recall/NDCG/MRR/Faithfulness
│
├── ui/
│   └── app.py                   # Streamlit frontend (3 chế độ)
│
├── eval/
│   ├── generate_testset.py      # Tạo bộ câu hỏi test tổng hợp
│   ├── experiment_k.py          # Thử nghiệm hyperparameter top_k
│   ├── test_queries.json        # 132 câu hỏi + ground-truth
│   └── results/                 # Kết quả đánh giá (JSON)
│
├── data/
│   ├── raw/                     # PDFs gốc theo danh mục
│   │   ├── Luat-quoc-gia/      # Luật quốc gia
│   │   ├── Quyet-dinh-quy-che/ # Quyết định, quy chế
│   │   ├── Thong-bao/          # Thông báo
│   │   └── ...
│   └── processed/               # Dữ liệu đã xử lý
│       ├── *.jsonl              # Chunks (35 file)
│       └── bm25.pkl             # BM25 index (~2.3 MB)
│
├── config.yaml                  # Cấu hình toàn hệ thống
├── requirements.txt             # Python dependencies
├── start.sh                     # Khởi động API + UI
└── run_pipeline.sh              # Chạy indexing pipeline
```

---

## Công Nghệ Sử Dụng

| Thành phần | Công nghệ | Chi tiết |
|-----------|-----------|----------|
| **LLM** | OpenAI GPT-4o | Sinh câu trả lời + LLM judge |
| **Embeddings** | `bkai-foundation-models/vietnamese-bi-encoder` | 768 chiều, tiếng Việt |
| **Vector DB** | Qdrant | Cosine similarity, cloud hoặc local |
| **Sparse Search** | rank-bm25 (BM25Okapi) | k1=1.5, b=0.75 |
| **Reranker** | `AITeamVN/Vietnamese_Reranker` | Cross-encoder tiếng Việt |
| **PDF Parsing** | PyMuPDF | Trích xuất văn bản + số trang |
| **Tokenization** | underthesea | Tách từ tiếng Việt |
| **Backend** | FastAPI + Uvicorn | ASGI, async |
| **Frontend** | Streamlit | Interactive web UI |
| **Config** | PyYAML + python-dotenv | Env var substitution |

---

## Yêu Cầu Hệ Thống

- **Python**: 3.10 hoặc cao hơn
- **RAM**: Tối thiểu 16 GB (khuyến nghị 4 GB VRAM cho embedding local)
- **GPU**: Không bắt buộc, nhưng tăng tốc embedding và reranking đáng kể dùng cpu lâu hơn
- **Disk**: ~5 GB (model weights + data + index)
- **Qdrant**: Local (Docker) hoặc Qdrant Cloud
- **OpenAI API Key**: Bắt buộc cho LLM generation và evaluation
---

## Cài Đặt

### Bước 1: Clone dự án

```bash
cd chatbot_rag_ranking
```

### Bước 2: Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate.bat     # Windows
```

### Bước 3: Cài dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý:** Lần đầu chạy sẽ tự download models:
> - `bkai-foundation-models/vietnamese-bi-encoder` (~1.3 GB)
> - `AITeamVN/Vietnamese_Reranker` (~500 MB)

### Bước 4: Cài đặt Qdrant (Local)

```bash
# Dùng Docker
docker run -p 6333:6333 qdrant/qdrant

# Hoặc Qdrant Cloud: https://cloud.qdrant.io/ (nên dùng cái này nhé)
```

### Bước 5: Tạo file `.env`

```bash
cp .env.example .env   # nếu có
# Hoặc tạo mới:
```

```dotenv
OPENAI_API_KEY=sk-...your-key...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                   # để trống nếu dùng local
```

---

## Cấu Hình

File `config.yaml` chứa toàn bộ cấu hình hệ thống:

```yaml
# Đường dẫn dữ liệu
data:
  raw_dir: "data"
  processed_dir: "data/processed"

# Chunking
chunking:
  chunk_size: 800        # ký tự tối đa mỗi chunk
  chunk_overlap: 150     # overlap giữa các chunk

# Embedding
embedding:
  provider: "local"      # "local" | "openai"
  local_model: "bkai-foundation-models/vietnamese-bi-encoder"
  batch_size: 32

# Vector Store
vector_store:
  provider: "qdrant"
  qdrant_url: "${QDRANT_URL}"     # lấy từ biến môi trường
  qdrant_api_key: "${QDRANT_API_KEY}"
  collection_name: "rag_docs"
  vector_size: 768

# BM25 Sparse Index
bm25:
  index_path: "data/processed/bm25.pkl"
  k1: 1.5
  b: 0.75

# Retrieval
retrieval:
  top_k_dense: 20        # số kết quả dense
  top_k_sparse: 20       # số kết quả sparse
  top_k_fusion: 20       # số kết quả sau RRF
  rrf_k: 60              # hằng số RRF

# Reranking
reranking:
  enabled: true
  provider: "cross-encoder"   # "cross-encoder" | "cohere"
  cross_encoder_model: "AITeamVN/Vietnamese_Reranker"
  top_k: 4               # số chunk cuối cùng đưa vào LLM
  max_per_source: 3      # tối đa 3 chunk cùng nguồn

# LLM
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.0
  max_tokens: 1024

# Evaluation
evaluation:
  test_queries_path: "eval/test_queries.json"
  k_values: [1, 3, 5, 10]
  output_dir: "eval/results"
```

---

## Chạy Pipeline Lập Chỉ Mục

> **Chỉ cần chạy một lần** khi thêm tài liệu mới hoặc lần đầu cài đặt.

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Script này chạy tuần tự 3 bước:

```bash
# Bước 1: Parse PDF → JSONL chunks
python src/pipeline/01_parse_chunk.py

# Bước 2: Embed chunks → Qdrant
python src/pipeline/02_embed_index.py

# Bước 3: Xây BM25 sparse index
python src/pipeline/03_bm25_index.py
```

**Output:**
- `data/processed/*.jsonl` — 35 file JSONL chứa các chunks
- Qdrant collection `rag_docs` — dense vectors
- `data/processed/bm25.pkl` — BM25 index (~2.3 MB)

---

## Khởi Động Ứng Dụng

### Chạy Local

```bash
chmod +x start.sh
./start.sh
```

| Service | URL |
|---------|-----|
| API Backend | http://localhost:8000 |
| Streamlit UI | http://localhost:8501 |
| API Docs (Swagger) | http://localhost:8000/docs |

### Chạy với Cloudflare Tunnel (expose public URL)

```bash
./start.sh --tunnel
```

> Yêu cầu [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) đã được cài đặt.

### Chạy thủ công từng service

```bash
# Khởi động API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Khởi động UI (cửa sổ terminal khác)
streamlit run ui/app.py --server.port 8501
```

---

## API Endpoints

### `POST /chat`

Gửi câu hỏi và nhận câu trả lời từ RAG pipeline đầy đủ.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Sinh viên vi phạm quy định thi hộ lần đầu bị xử lý như thế nào?",
    "top_k": 4
  }'
```

**Response:**
```json
{
  "answer": "Theo quy định, sinh viên vi phạm thi hộ lần đầu sẽ bị kỷ luật đình chỉ học tập 01 năm...",
  "sources": [
    {
      "source": "3.-Phu-luc-1-30122022-Signed-2.pdf",
      "page": 2,
      "text": "Điều 5. Hình thức kỷ luật...",
      "rrf_score": 0.0312,
      "rerank_score": 4.87
    }
  ]
}
```

---

### `POST /retrieve`

Chỉ truy xuất tài liệu liên quan, không gọi LLM.

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "hoãn thi",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "abc123...",
      "source": "quy-che-hoc-vu.pdf",
      "page": 7,
      "text": "...",
      "rrf_score": 0.029,
      "rerank_score": 3.21
    }
  ]
}
```

---

### `POST /evaluate`

Chạy bộ đánh giá tự động trên `eval/test_queries.json`.

```bash
curl -X POST http://localhost:8000/evaluate
```

---

### `GET /health`

Kiểm tra trạng thái server.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Giao Diện Web (Streamlit UI)

Truy cập `http://localhost:8501` và chọn một trong 3 chế độ:

### 1. Chatbot (`💬 Chatbot`)

- Gõ câu hỏi bằng tiếng Việt
- Nhận câu trả lời kèm nguồn tài liệu và số trang
- Lưu lịch sử hội thoại trong phiên

### 2. Retrieve Only (`🔍 Retrieve Only`)

- Debug quá trình tìm kiếm
- Hiển thị top-k chunks kèm:
  - RRF score (điểm hợp nhất)
  - Rerank score (điểm cross-encoder)
  - Nguồn tài liệu và số trang
  - Nội dung đoạn văn

### 3. Evaluation (`📊 Evaluation`)

- Xem kết quả đánh giá gần nhất
- Chạy lại bộ đánh giá mới
- Hiển thị biểu đồ Precision/Recall/NDCG theo k
- Hiển thị điểm Faithfulness và Answer Relevance

---

## Đánh Giá Hệ Thống

### Tạo bộ câu hỏi test

```bash
python eval/generate_testset.py
```

Tự động sinh 132 câu hỏi với 4 dạng:
- **Factoid**: `"X là gì?"` — hỏi về một sự kiện cụ thể
- **Procedural**: `"Thủ tục để làm X?"` — hỏi về quy trình
- **Conditional**: `"Nếu... thì...?"` — hỏi về điều kiện
- **Multi-hop**: Kết hợp thông tin từ 2+ chunk

### Chạy đánh giá

```bash
python src/evaluation/evaluator.py
# Kết quả lưu vào eval/results/summary_<timestamp>.json
```

### Thử nghiệm hyperparameter

```bash
python eval/experiment_k.py
# So sánh các cấu hình top_k khác nhau
```

---

## Kết Quả Đánh Giá

> Bộ test: **132 câu hỏi** (factoid, procedural, conditional, multi-hop)

### Retrieval Metrics (chạy lại validate nhé)

| Metric | k=1 | k=3 | k=5 | k=10 |
|--------|-----|-----|-----|------|
| Precision@k | 0.417 | 0.278 | 0.205 | 0.126 |
| Recall@k | 0.278 | 0.347 | 0.317 | 0.333 |
| Hit Rate@k | 0.417 | 0.556 | 0.625 | 0.694 |
| MRR@k | 0.417 | 0.461 | 0.469 | 0.474 |
| NDCG@k | 0.278 | 0.316 | 0.335 | 0.339 |

### Generation Metrics (LLM-as-Judge)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.571 |
| Answer Relevance | 0.564 |
| Context Precision | 0.313 |

**Nhận xét:**
- Precision@1 đạt 41.7% → kết quả hàng đầu thường chính xác
- Hit Rate@5 đạt 62.5% → hơn 60% câu hỏi tìm được đúng chunk trong top-5
- Faithfulness 57% → hệ thống bám sát ngữ cảnh, ít hallucination
- Context Precision 31% → còn nhiều chunk nhiễu, có thể cải thiện reranker

---

## Tùy Chỉnh & Mở Rộng

### Thêm tài liệu mới

1. Đặt file PDF vào `data/raw/<category>/`
2. Chạy lại indexing pipeline:
   ```bash
   ./run_pipeline.sh
   ```

### Thay đổi LLM

Trong `config.yaml`:
```yaml
llm:
  model: "gpt-4o-mini"   # hoặc model khác
  temperature: 0.1
```

### Dùng Cohere Reranker

```yaml
reranking:
  provider: "cohere"
```

```bash
pip install cohere
# Thêm COHERE_API_KEY vào .env
```

### Dùng OpenAI Embeddings

```yaml
embedding:
  provider: "openai"
  openai_model: "text-embedding-3-small"
```

### Điều chỉnh số chunk trả về

```yaml
reranking:
  top_k: 6          # tăng context cho LLM
  max_per_source: 2  # giảm để tăng diversity
```

---

## Giấy Phép

Dự án được phát triển cho mục đích nghiên cứu và học thuật tại Trường Đại học Quốc tế – ĐHQG-HCM.
