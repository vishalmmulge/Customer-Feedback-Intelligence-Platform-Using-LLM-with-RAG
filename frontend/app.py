"""Streamlit dashboard for the Customer Feedback Intelligence Platform.

Run with:  streamlit run frontend/app.py

Start the backend first:  uvicorn backend.main:app --reload
Or use Docker Compose:    docker compose up
"""

import os
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Customer Feedback Intelligence")
st.caption(
    "Upload CSV feedback → sentiment & topic analysis → semantic search → "
    "grounded RAG Q&A → evaluation metrics"
)

# --------------------------------------------------------------------------- #
# Sidebar: upload + status
# --------------------------------------------------------------------------- #
st.sidebar.header("1. Upload feedback")
uploaded = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
source_label = st.sidebar.text_input("Source label", value="csv")
column_mapping_json = st.sidebar.text_area(
    "Column mapping (JSON)",
    value="",
    help='Map CSV columns to internal schema. Example: {"Text": "feedback_text", "User ID": "customer_id", "Date/Time": "timestamp"}',
)
if st.sidebar.button("Process upload", type="primary"):
    if uploaded is None:
        st.sidebar.error("Please select a CSV file first.")
    else:
        with st.spinner("Processing feedback…"):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                params = {"source": source_label}
                if column_mapping_json:
                    params["column_mapping"] = column_mapping_json
                resp = requests.post(
                    f"{BACKEND_URL}/feedback/upload",
                    files=files,
                    params=params,
                    timeout=300,
                )
                if resp.ok:
                    data = resp.json()
                    st.sidebar.success(
                        f"Accepted {data['accepted']} / {data['total']} rows. "
                        f"{data['message']}"
                    )
                    st.rerun()
                else:
                    st.sidebar.error(f"Upload failed: {resp.text}")
            except Exception as exc:  # pragma: no cover - network
                st.sidebar.error(f"Could not reach backend: {exc}")

# Column mapping reference
st.sidebar.divider()
st.sidebar.write(f"Backend: `{BACKEND_URL}`")
if st.sidebar.button("Refresh"):
    st.rerun()

# Backend health
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    ollama = requests.get(f"{BACKEND_URL}/health/ollama", timeout=5).json()
    st.sidebar.write("**Backend status:**", health["status"])
    st.sidebar.write("**Ollama ready:**", ollama.get("ollama_ready", "unknown"))
    st.sidebar.write("**Vector backend:**", health["vector_backend"])
    st.sidebar.write("**Database:**", health["database_backend"])
except Exception:
    st.sidebar.error("Backend not reachable")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get(path, params=None):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Backend error: {exc}")
        return None


def _post(path, json=None, files=None, params=None):
    try:
        resp = requests.post(
            f"{BACKEND_URL}{path}", json=json, files=files, params=params, timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Backend error: {exc}")
        return None


def _delete(path):
    try:
        resp = requests.delete(f"{BACKEND_URL}{path}", timeout=30)
        resp.raise_for_status()
        return True
    except Exception as exc:
        st.error(f"Delete failed: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_overview, tab_explorer, tab_insights, tab_qa, tab_eval = st.tabs(
    [
        "A. Overview",
        "B. Feedback Explorer",
        "C. Customer Insights",
        "D. AI Assistant",
        "E. Evaluation",
    ]
)

# =========================================================================== #
# A. Overview
# =========================================================================== #
with tab_overview:
    st.subheader("Feedback overview")
    summary = _get("/analytics/summary")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total feedback", summary["total_feedback"])
        c2.metric("Avg rating", summary["average_rating"] or "n/a")
        c3.metric("Sentiments", len(summary["sentiment_distribution"]))
        c4.metric("Topics", len(summary["topic_distribution"]))

        st.write("### Sentiment distribution")
        st.bar_chart(summary["sentiment_distribution"])

        st.write("### Topic distribution")
        st.bar_chart(summary["topic_distribution"])

        st.write("### Source distribution")
        st.bar_chart(summary["source_distribution"])

        st.write("### Product distribution")
        st.bar_chart(summary["product_distribution"])

# =========================================================================== #
# B. Feedback Explorer
# =========================================================================== #
with tab_explorer:
    st.subheader("Feedback explorer")
    col1, col2, col3 = st.columns(3)
    sentiment_filter = col1.selectbox(
        "Sentiment", ["", "positive", "neutral", "negative"], index=0
    )
    topic_filter = col2.text_input("Topic contains")
    source_filter = col3.text_input("Source contains")

    data = _get(
        "/feedback",
        params={
            "page_size": 100,
            "sentiment": sentiment_filter or None,
            "topic": topic_filter or None,
            "source": source_filter or None,
        },
    )
    if data:
        rows = []
        for r in data["records"]:
            fid = r.get("feedback_id") or r.get("id") or ""
            rows.append(
                {
                    "id": (fid or "")[:8],
                    "feedback_id": (fid or "")[:8],
                    "customer": r["customer_id"],
                    "product": r["product"] or "",
                    "channel": r["channel"] or "",
                    "source": r["source"] or "",
                    "rating": r["rating"] or "",
                    "sentiment": r["sentiment"] or "",
                    "topic": r["topic"] or "",
                    "feedback": r["feedback_text"],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.write(f"Showing {len(rows)} of {data['total']} total records.")

        st.divider()
        st.write("### Delete a record")
        del_id = st.text_input("Feedback ID to delete")
        if st.button("Delete record") and del_id:
            if _delete(f"/feedback/{del_id}"):
                st.success("Deleted. Refresh to see changes.")

# =========================================================================== #
# C. Customer Insights
# =========================================================================== #
with tab_insights:
    st.subheader("Customer insights")
    overview = _get("/analytics/overview")
    sentiment = _get("/analytics/sentiment")
    topics = _get("/analytics/topics")
    trends = _get("/analytics/trends")

    if overview:
        st.write("### Top topics")
        st.bar_chart(overview["topic_distribution"])

    if sentiment:
        st.write("### Sentiment breakdown (%)")
        st.bar_chart(sentiment["percentages"])

    if trends:
        st.write("### Feedback volume trend")
        import pandas as pd

        trend_df = pd.DataFrame(trends["trend"])
        if not trend_df.empty:
            st.line_chart(trend_df.set_index("date")["count"])

# =========================================================================== #
# D. AI Assistant (RAG)
# =========================================================================== #
with tab_qa:
    st.subheader("AI Feedback Assistant")
    st.write(
        "Ask a natural-language question. Answers are grounded in retrieved "
        "customer feedback with source references and similarity scores."
    )
    question = st.text_input(
        "Your question",
        placeholder="e.g. What are customers complaining about most?",
    )
    top_k = st.slider("Top K results", 1, 20, 5)
    if st.button("Ask") and question:
        result = _post("/rag/ask", {"question": question, "top_k": top_k})
        if result:
            st.write("### Answer")
            st.write(result["answer"])
            st.write("### Supporting sources")
            for src in result["sources"]:
                with st.expander(
                    f"{src['id'][:8]} — sentiment {src.get('sentiment','?')} "
                    f"(similarity {src.get('score',0):.3f})"
                ):
                    st.write(src.get("feedback_text", ""))

# =========================================================================== #
# E. Evaluation
# =========================================================================== #
with tab_eval:
    st.subheader("RAG Evaluation")
    st.write(
        "Run the evaluation suite against a manually labeled dataset. "
        "Metrics are computed from actual retrieval and generation output — "
        "nothing is fabricated."
    )
    top_k = st.slider("Top K", 1, 20, 5, key="eval_top_k")
    if st.button("Run evaluation"):
        result = _post("/evaluation/run", {"top_k": top_k})
        if result:
            agg = result["aggregate"]
            st.write("### Aggregate metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision@K", agg["precision_at_k"])
            c2.metric("Recall@K", agg["recall_at_k"])
            c3.metric("MRR", agg["mrr"])
            c4.metric("Faithfulness", agg["faithfulness"])

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Context relevance", agg["context_relevance"])
            c6.metric("Answer relevance", agg["answer_relevance"])
            c7.metric("Evidence coverage", agg["evidence_coverage"])
            c8.metric("Cases", result["num_cases"])

            st.write("### Per-case results")
            for case in result["results"]:
                with st.expander(
                    f"{case['category']} — {case['question'][:60]}..."
                ):
                    st.write("**Question:**", case["question"])
                    ret = case["retrieval"]
                    st.write(
                        f"Precision@{ret['top_k']}: {ret['precision_at_k']} · "
                        f"Recall@{ret['top_k']}: {ret['recall_at_k']} · "
                        f"MRR: {ret['mrr']}"
                    )
                    st.write("**Answer:**", case["answer"])
                    am = case["answer_metrics"]
                    st.write(
                        f"Faithfulness {am['faithfulness']} · "
                        f"Context relevance {am['context_relevance']} · "
                        f"Answer relevance {am['answer_relevance']} · "
                        f"Evidence coverage {am['evidence_coverage']}"
                    )
