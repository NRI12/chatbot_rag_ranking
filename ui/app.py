"""
Streamlit UI for the RAG Chatbot with evaluation dashboard.

Run: streamlit run ui/app.py
"""

import json
from pathlib import Path

import requests
import streamlit as st

import os
API_BASE = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Chatbot – Quy chế Sinh viên ĐHQT",
    page_icon="📚",
    layout="wide",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Cài đặt")
    api_url = st.text_input("API URL", value=API_BASE)
    mode = st.radio("Chế độ", ["💬 Chatbot", "🔍 Retrieve only", "📊 Evaluation"])
    st.markdown("---")
    st.caption("RAG | Hybrid Search | Reranking")


# ─── Chat mode ────────────────────────────────────────────────────────────────

if mode == "💬 Chatbot":
    st.title("📚 Tư vấn Quy chế Sinh viên ĐHQT")
    st.caption("Hỏi đáp dựa trên văn bản quy chế và luật pháp")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Nguồn tham khảo"):
                    for s in msg["sources"]:
                        score_info = ""
                        if s.get("score_rerank") is not None:
                            score_info = f" | rerank={s['score_rerank']:.3f}"
                        elif s.get("score_rrf") is not None:
                            score_info = f" | rrf={s['score_rrf']:.4f}"
                        st.markdown(
                            f"**{s.get('source', 'N/A')}** (trang {s.get('page', '?')}){score_info}"
                        )
                        st.markdown(f"> {s['text'][:300]}…")

    if prompt := st.chat_input("Nhập câu hỏi của bạn…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và trả lời…"):
                try:
                    resp = requests.post(
                        f"{api_url}/chat",
                        json={"question": prompt},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data["answer"]
                    sources = data["sources"]
                except Exception as e:
                    answer = f"Lỗi kết nối API: {e}"
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander("📄 Nguồn tham khảo"):
                    for s in sources:
                        score_info = ""
                        if s.get("score_rerank") is not None:
                            score_info = f" | rerank={s['score_rerank']:.3f}"
                        elif s.get("score_rrf") is not None:
                            score_info = f" | rrf={s['score_rrf']:.4f}"
                        st.markdown(
                            f"**{s.get('source', 'N/A')}** (trang {s.get('page', '?')}){score_info}"
                        )
                        st.markdown(f"> {s['text'][:300]}…")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


# ─── Retrieve-only mode ───────────────────────────────────────────────────────

elif mode == "🔍 Retrieve only":
    st.title("🔍 Kiểm tra Retrieval")
    query = st.text_input("Nhập truy vấn")

    if st.button("Tìm kiếm") and query:
        with st.spinner("Đang tìm…"):
            try:
                resp = requests.post(
                    f"{api_url}/retrieve",
                    json={"query": query},
                    timeout=30,
                )
                resp.raise_for_status()
                results = resp.json()["results"]
            except Exception as e:
                st.error(f"Lỗi: {e}")
                results = []

        st.subheader(f"Kết quả ({len(results)} chunks)")
        for i, r in enumerate(results, 1):
            score_label = ""
            if r.get("score_rerank") is not None:
                score_label = f"rerank: {r['score_rerank']:.3f}"
            elif r.get("score_rrf") is not None:
                score_label = f"RRF: {r['score_rrf']:.4f}"

            with st.expander(
                f"#{i} – {r.get('source', 'N/A')} p.{r.get('page', '?')}  [{score_label}]"
            ):
                st.text(r["text"])


# ─── Evaluation dashboard ─────────────────────────────────────────────────────

elif mode == "📊 Evaluation":
    st.title("📊 Evaluation Dashboard")

    col1, col2 = st.columns([2, 1])

    with col2:
        if st.button("▶ Chạy đánh giá", type="primary"):
            with st.spinner("Đang đánh giá… (có thể mất vài phút)"):
                try:
                    resp = requests.post(f"{api_url}/evaluate", timeout=600)
                    resp.raise_for_status()
                    st.session_state["eval_result"] = resp.json()
                    st.success("Hoàn thành!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # Load latest result from disk if available
    result_dir = Path("eval/results")
    result_files = sorted(result_dir.glob("summary_*.json"), reverse=True) if result_dir.exists() else []

    with col1:
        if result_files:
            selected = st.selectbox(
                "Chọn kết quả",
                [f.name for f in result_files],
                index=0,
            )
            with open(result_dir / selected, encoding="utf-8") as f:
                loaded = json.load(f)
            st.session_state["eval_result"] = loaded

    if "eval_result" in st.session_state:
        data = st.session_state["eval_result"]
        st.markdown(f"**Timestamp:** {data.get('timestamp', 'N/A')}  |  **Số queries:** {data.get('n_queries', 0)}")

        col_r, col_g = st.columns(2)

        with col_r:
            st.subheader("Retrieval Metrics")
            ret = data.get("retrieval_metrics", {})
            if ret:
                import pandas as pd
                df = pd.DataFrame(
                    [(k, round(v, 4)) for k, v in sorted(ret.items())],
                    columns=["Metric", "Score"],
                )
                st.dataframe(df, use_container_width=True)

                # Chart: NDCG and MRR per k
                ndcg = {k: v for k, v in ret.items() if k.startswith("ndcg@")}
                mrr = {k: v for k, v in ret.items() if k.startswith("mrr@")}
                if ndcg or mrr:
                    chart_data = pd.DataFrame({"NDCG": ndcg, "MRR": mrr}).T
                    st.bar_chart(chart_data)
            else:
                st.info("Không có dữ liệu retrieval (cần relevant_ids trong test_queries.json)")

        with col_g:
            st.subheader("Generation Metrics (LLM-as-judge)")
            gen = data.get("generation_metrics", {})
            if gen:
                import pandas as pd
                df_gen = pd.DataFrame(
                    [(k, round(v, 4)) for k, v in gen.items()],
                    columns=["Metric", "Score"],
                )
                st.dataframe(df_gen, use_container_width=True)
                st.bar_chart(
                    pd.DataFrame(gen, index=["score"]).T.rename(columns={"score": "Score"})
                )
            else:
                st.info("Chưa có kết quả generation metrics")
