import json
import os
from html import escape
from pathlib import Path

import streamlit as st

from hr_rag import HRRagConfig, HRRagPipeline


st.set_page_config(page_title="Zyro HR Help Desk", layout="wide")

st.markdown(
    """
<style>
.stApp { background: #f7f8fa; }
[data-testid="stSidebar"] { background: #ffffff; }
.source-box {
  border: 1px solid #d8dee8;
  border-radius: 8px;
  padding: 10px 12px;
  background: #ffffff;
  margin-bottom: 8px;
}
.source-title { font-weight: 650; color: #172033; }
.source-preview { color: #526070; font-size: 13px; margin-top: 4px; }
</style>
""",
    unsafe_allow_html=True,
)


def make_config() -> HRRagConfig:
    return HRRagConfig(
        docs_path=st.session_state.get("docs_path", "hr_docs"),
        db_path=st.session_state.get("db_path", "chroma_hr_store"),
        embedding_provider=st.session_state.get("embedding_provider", "auto"),
        llm_provider=st.session_state.get("llm_provider", "auto"),
        chunk_size=int(st.session_state.get("chunk_size", 900)),
        chunk_overlap=int(st.session_state.get("chunk_overlap", 180)),
        retrieval_k=int(st.session_state.get("retrieval_k", 6)),
        fetch_k=int(st.session_state.get("fetch_k", 24)),
    )


@st.cache_resource(show_spinner=False)
def load_pipeline(config_json: str, rebuild: bool) -> HRRagPipeline:
    cfg = HRRagConfig(**json.loads(config_json))
    return HRRagPipeline.from_config(cfg, rebuild=rebuild)


def config_cache_key(cfg: HRRagConfig) -> str:
    return json.dumps(cfg.__dict__, sort_keys=True)


with st.sidebar:
    st.title("Zyro HR")
    st.session_state.docs_path = st.text_input("Policy docs folder", value=st.session_state.get("docs_path", "hr_docs"))
    st.session_state.db_path = st.text_input("Vector DB folder", value=st.session_state.get("db_path", "chroma_hr_store"))
    st.session_state.embedding_provider = st.selectbox(
        "Embeddings",
        ["auto", "openai", "ollama", "hash"],
        index=["auto", "openai", "ollama", "hash"].index(st.session_state.get("embedding_provider", "auto")),
    )
    st.session_state.llm_provider = st.selectbox(
        "Answer model",
        ["auto", "groq", "openai", "ollama", "extractive"],
        index=["auto", "groq", "openai", "ollama", "extractive"].index(st.session_state.get("llm_provider", "auto")),
    )
    st.session_state.chunk_size = st.slider("Chunk size", 400, 1800, int(st.session_state.get("chunk_size", 900)), 50)
    st.session_state.chunk_overlap = st.slider("Chunk overlap", 50, 400, int(st.session_state.get("chunk_overlap", 180)), 25)
    st.session_state.retrieval_k = st.slider("Retrieved chunks", 3, 10, int(st.session_state.get("retrieval_k", 6)), 1)
    st.session_state.fetch_k = st.slider("Candidate chunks", 10, 60, int(st.session_state.get("fetch_k", 24)), 2)

    rebuild = st.button("Rebuild Index", use_container_width=True)
    st.caption("Secrets are read from `.env`, environment variables, or Streamlit secrets.")


st.title("Zyro Dynamics HR Help Desk")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

cfg = make_config()

try:
    pipeline = load_pipeline(config_cache_key(cfg), rebuild)
    ready_error = None
except Exception as exc:
    pipeline = None
    ready_error = str(exc)

left, right = st.columns([2.2, 1], gap="large")

with left:
    if ready_error:
        st.error(ready_error)
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask an HR policy question")
        if user_input and pipeline is not None:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            history_pairs = []
            messages = st.session_state.messages[:-1]
            for idx in range(0, len(messages) - 1, 2):
                if messages[idx]["role"] == "user" and messages[idx + 1]["role"] == "assistant":
                    history_pairs.append((messages[idx]["content"], messages[idx + 1]["content"]))

            with st.chat_message("assistant"):
                with st.spinner("Searching Zyro policy documents..."):
                    response = pipeline.answer(user_input, chat_history=history_pairs)
                st.markdown(response.answer)

            st.session_state.messages.append({"role": "assistant", "content": response.answer})
            st.session_state.last_sources = response.sources

with right:
    st.subheader("Retrieved Sources")
    if not st.session_state.last_sources:
        st.info("Sources appear after the first answered question.")
    for source in st.session_state.last_sources:
        st.markdown(
            """
<div class="source-box">
  <div class="source-title">{source_file} · chunk {chunk_id}</div>
  <div class="source-preview">{preview}</div>
</div>
            """.format(
                source_file=escape(source["source_file"]),
                chunk_id=escape(source["chunk_id"]),
                preview=escape(source["preview"]),
            ),
            unsafe_allow_html=True,
        )

    st.subheader("Index Status")
    docs_folder = Path(cfg.docs_path)
    file_count = len([p for p in docs_folder.rglob("*") if p.is_file()]) if docs_folder.exists() else 0
    st.metric("Policy files", file_count)
    st.metric("LLM provider", os.getenv("LLM_PROVIDER", cfg.llm_provider))
    st.metric("Embedding provider", os.getenv("EMBEDDING_PROVIDER", cfg.embedding_provider))

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()
