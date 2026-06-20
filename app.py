"""
Context Compressor — Streamlit Web App
=========================================
Author : Ramya Sree Nagarajan

Run:
  streamlit run app/app.py
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.compressor import ContextCompressor, estimate_tokens

st.set_page_config(page_title="Context Compressor", page_icon="🗜️", layout="wide")

st.markdown("""
<style>
body, .main { background-color: #0D1117; color: #C9D1D9; }
.stTextArea textarea {
    background-color: #161B22; color: #C9D1D9;
    border: 1px solid #30363D; border-radius: 8px; font-family: monospace;
}
.metric-card {
    background: linear-gradient(135deg, #161B22, #1F2937);
    border: 1px solid #30363D; border-radius: 10px;
    padding: 18px; text-align: center;
}
.metric-value { font-size: 28px; font-weight: bold; color: #70A5FD; }
.metric-label { font-size: 13px; color: #8B949E; margin-top: 4px; }
.savings-banner {
    background: linear-gradient(135deg, #1A3D2B, #38BDAE);
    border-radius: 12px; padding: 16px; text-align: center;
    font-size: 20px; font-weight: bold; margin: 12px 0;
}
.stButton button {
    background: linear-gradient(135deg, #70A5FD, #BF91F3);
    color: white; border: none; border-radius: 8px;
    font-weight: bold; width: 100%;
}
.step-pill {
    display: inline-block; background: #1F2937; color: #38BDAE;
    padding: 4px 12px; border-radius: 14px; font-size: 12px;
    margin: 3px; border: 1px solid #30363D;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🗜️ Context Compressor")
st.markdown("**Shrink logs, documents & RAG chunks before they hit your LLM** — save tokens, cost, and context window space")
st.divider()

DEMO_LOG = """2026-06-18T10:01:02 INFO  Request received from 192.168.1.45 id=a1b2c3d4-1111-2222-3333-444455556666
2026-06-18T10:01:03 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 1
2026-06-18T10:01:04 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 2
2026-06-18T10:01:05 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 3
2026-06-18T10:01:06 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 4
2026-06-18T10:01:08 WARN  Circuit breaker opened for service payments-api
2026-06-18T10:01:10 INFO  Health check passed for service auth-api
2026-06-18T10:01:11 INFO  Health check passed for service auth-api
2026-06-18T10:01:12 INFO  Health check passed for service auth-api
2026-06-18T10:01:15 ERROR NullPointerException at OrderProcessor.java:142
2026-06-18T10:01:16 ERROR NullPointerException at OrderProcessor.java:142"""

DEMO_PROSE = """As I mentioned earlier, the quarterly report shows really strong growth across all business segments. It's worth noting that revenue increased by 23 percent year over year, driven primarily by the enterprise software division. In other words, the company is performing very well in a difficult macroeconomic environment. Basically, the management team attributes this growth to three key factors: improved sales efficiency, expansion into new markets, and successful product launches. At the end of the day, customer retention rates also improved significantly, reaching 94 percent for the fiscal year, up from 89 percent. Essentially, this demonstrates strong product-market fit. Needless to say, investors reacted positively, with the stock price rising 12 percent following the announcement."""

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Compression Settings")
    content_type = st.selectbox("Content type", ["auto", "logs", "prose", "rag_chunks"])
    dedupe = st.checkbox("Deduplicate lines", value=True)
    fold_logs = st.checkbox("Fold repeated log patterns", value=True)
    prune_filler = st.checkbox("Prune filler phrases", value=True)
    rank_sentences = st.checkbox("Rank & filter sentences (prose)", value=False)
    keep_ratio = st.slider("Sentence keep ratio", 0.2, 1.0, 0.6, 0.05,
                           disabled=not rank_sentences)

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
    Techniques applied:
    - **Dedup** — removes exact/near-duplicate lines
    - **Log folding** — collapses repeated error patterns into counts
    - **Filler pruning** — strips low-signal phrases
    - **Sentence ranking** — keeps only the highest-signal sentences

    ---
    **Author:** Ramya Sree Nagarajan
    MSc AI · IEEE Published Researcher

    [GitHub](https://github.com/ramyasreenagarajan)
    """)

# ── Main input ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    if st.button("📋 Load demo: Server Logs"):
        st.session_state.input_text = DEMO_LOG
with col2:
    if st.button("📝 Load demo: Filler-heavy Report"):
        st.session_state.input_text = DEMO_PROSE

input_text = st.text_area(
    "Paste logs, documentation, or RAG chunks here",
    value=st.session_state.get("input_text", ""),
    height=220,
    placeholder="Paste your text here, or click a demo button above...",
)

if st.button("🗜️ Compress", type="primary"):
    if not input_text.strip():
        st.warning("Please paste some text first.")
    else:
        compressor = ContextCompressor(
            dedupe=dedupe, fold_logs=fold_logs,
            prune_filler=prune_filler,
            rank_sentences=rank_sentences, keep_ratio=keep_ratio,
        )
        result = compressor.compress(input_text, content_type=content_type)
        stats = result["stats"]

        st.markdown(
            f'<div class="savings-banner">💰 {stats["compression_ratio"]*100:.1f}% '
            f'token reduction — saved {stats["tokens_saved"]:,} tokens</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        for col, label, value in [
            (m1, "Original Tokens", f"{stats['original_tokens']:,}"),
            (m2, "Compressed Tokens", f"{stats['compressed_tokens']:,}"),
            (m3, "Detected Type", stats["detected_type"]),
            (m4, "Est. Cost Saved", f"${stats['estimated_cost_saved_usd']:.6f}"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{value}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("")
        st.markdown("**Pipeline steps applied:**")
        steps_html = "".join(
            f'<span class="step-pill">{s["step"].replace("_"," ").title()}</span>'
            for s in stats["steps"]
        )
        st.markdown(steps_html, unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📄 Before")
            st.text_area("before", input_text, height=300, label_visibility="collapsed")
        with c2:
            st.markdown("#### ✅ After")
            st.text_area("after", result["compressed_text"], height=300, label_visibility="collapsed")

st.divider()
st.markdown(
    "<div style='text-align:center;color:#8B949E;font-size:13px;'>"
    "Built by <b>Ramya Sree Nagarajan</b> · MSc AI · IEEE Published Researcher · "
    "<a href='https://github.com/ramyasreenagarajan' style='color:#70A5FD;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
