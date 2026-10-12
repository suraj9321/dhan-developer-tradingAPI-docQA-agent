"""Demo UI: ask a question, see the cited answer and the exact passages behind it.

    streamlit run app/streamlit_app.py      # expects the API on DOCSQA_API (default localhost:8000)
"""

from __future__ import annotations

import os
import re

import requests
import streamlit as st

API = os.environ.get("DOCSQA_API", "http://localhost:8000")

EXAMPLES = [
    "How do I cancel an order on US stocks?",
    "What values can productType take when placing an order?",
    "What does error DH-904 mean and how should I handle it?",
    "What do I need to set up before placing my first order via the API?",
    "Which endpoint withdraws funds to my bank account?",
]

st.set_page_config(page_title="DocsQA · DhanHQ", page_icon="📘", layout="wide")
st.title("DocsQA — DhanHQ API assistant")
st.caption("Answers only from the DhanHQ documentation, with citations. Read-only: it never places orders.")

with st.sidebar:
    system = st.radio(
        "Pipeline",
        ["agent", "hybrid", "rerank", "baseline"],
        help="agent: hybrid retrieval + evidence grading, one query rewrite, citation check. "
        "hybrid: BM25 + vector with reciprocal rank fusion. rerank: hybrid + cross-encoder. baseline: vector only.",
    )
    st.markdown("**Examples**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex

question = st.text_input("Question", key="q", placeholder="e.g. What is the rate limit for order APIs?")

if question:
    with st.spinner("Searching the docs…"):
        try:
            r = requests.post(f"{API}/ask", json={"question": question, "system": system}, timeout=180)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            st.error(f"API error: {e}")
            st.stop()

    cites = {c["n"]: c for c in data["citations"]}

    def link(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[[{n}]](#cite-{n})" if n in cites else m.group(0)

    if data["refused"]:
        st.warning(data["answer"])
    else:
        st.markdown(re.sub(r"\[(\d+)\]", link, data["answer"]))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency", f"{data['latency_ms'] / 1000:.1f} s")
    m2.metric("Tokens", f"{data['input_tokens']}+{data['output_tokens']}")
    m3.metric("Cost", f"${data['cost_usd']:.5f}")
    m4.metric("Model", data["model"].split("/")[-1])

    if cites:
        st.subheader("Sources")
        for n, c in sorted(cites.items()):
            st.markdown(f'<a name="cite-{n}"></a>', unsafe_allow_html=True)
            with st.expander(f"[{n}] {c['citation']}  ·  {' › '.join([c['sub_doc'], *c['heading_path']])}"):
                st.markdown(c["text"])

    if data.get("trace"):
        with st.expander("Agent trace"):
            for step in data["trace"]:
                st.text(step)
