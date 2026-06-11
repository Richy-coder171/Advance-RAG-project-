import json
import os
from html import escape
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Zyro HR Help Desk", layout="wide")


def load_streamlit_secrets() -> None:
    keys = [
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING",
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "GROQ_MODEL",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    ]
    for key in keys:
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value and not os.getenv(key):
            os.environ[key] = str(value)
    os.environ.setdefault("LANGCHAIN_PROJECT", "zyro-rag-challenge")


load_streamlit_secrets()

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
.source-meta { color: #245a3a; font-size: 12px; margin-top: 3px; }
.source-preview { color: #526070; font-size: 13px; margin-top: 4px; }
</style>
""",
    unsafe_allow_html=True,
)


def make_config():
    from hr_rag import HRRagConfig

    return HRRagConfig(
        docs_path=st.session_state.get("docs_path", "hr_docs/official"),
        db_path=st.session_state.get("db_path", "chroma_zyro_official_store"),
        embedding_provider=st.session_state.get("embedding_provider", "auto"),
        llm_provider=st.session_state.get("llm_provider", "auto"),
        chunk_size=int(st.session_state.get("chunk_size", 700)),
        chunk_overlap=int(st.session_state.get("chunk_overlap", 150)),
        retrieval_k=int(st.session_state.get("retrieval_k", 6)),
        fetch_k=int(st.session_state.get("fetch_k", 48)),
        vector_weight=float(st.session_state.get("vector_weight", 0.6)),
        keyword_weight=1.0 - float(st.session_state.get("vector_weight", 0.6)),
        min_confidence=float(st.session_state.get("min_confidence", 0.35)),
        max_chunks_per_source=int(st.session_state.get("max_chunks_per_source", 2)),
        enable_hyde=bool(st.session_state.get("enable_hyde", True)),
        enable_self_critique=bool(st.session_state.get("enable_self_critique", True)),
        critique_confidence_threshold=float(st.session_state.get("critique_confidence_threshold", 0.65)),
        append_source_block=bool(st.session_state.get("append_source_block", True)),
    )


@st.cache_resource(show_spinner="Loading HR policies and preparing the RAG pipeline...")
def load_pipeline(config_json: str, rebuild: bool):
    from hr_rag import HRRagConfig, HRRagPipeline

    cfg = HRRagConfig(**json.loads(config_json))
    return HRRagPipeline.from_config(cfg, rebuild=rebuild)


def config_cache_key(cfg) -> str:
    return json.dumps(cfg.__dict__, sort_keys=True)


with st.sidebar:
    st.title("Zyro HR")
    st.session_state.docs_path = st.text_input(
        "Policy docs folder", value=st.session_state.get("docs_path", "hr_docs/official")
    )
    st.session_state.db_path = st.text_input(
        "Vector DB folder", value=st.session_state.get("db_path", "chroma_zyro_official_store")
    )
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
    st.session_state.chunk_size = st.slider("Chunk size", 400, 1800, int(st.session_state.get("chunk_size", 700)), 50)
    st.session_state.chunk_overlap = st.slider("Chunk overlap", 50, 400, int(st.session_state.get("chunk_overlap", 150)), 25)
    st.session_state.retrieval_k = st.slider("Retrieved chunks", 3, 10, int(st.session_state.get("retrieval_k", 6)), 1)
    st.session_state.fetch_k = st.slider("Candidate chunks", 10, 60, int(st.session_state.get("fetch_k", 48)), 2)
    st.session_state.vector_weight = st.slider(
        "Vector retrieval weight", 0.0, 1.0, float(st.session_state.get("vector_weight", 0.6)), 0.05
    )
    st.session_state.min_confidence = st.slider(
        "Minimum confidence", 0.0, 1.0, float(st.session_state.get("min_confidence", 0.35)), 0.05
    )
    st.session_state.max_chunks_per_source = st.slider(
        "Max chunks per source", 1, 5, int(st.session_state.get("max_chunks_per_source", 2)), 1
    )
    st.session_state.enable_hyde = st.toggle(
        "Conditional HyDE", value=bool(st.session_state.get("enable_hyde", True))
    )
    st.session_state.enable_self_critique = st.toggle(
        "Low-confidence refinement", value=bool(st.session_state.get("enable_self_critique", True))
    )
    st.session_state.critique_confidence_threshold = st.slider(
        "Refinement threshold",
        0.0,
        1.0,
        float(st.session_state.get("critique_confidence_threshold", 0.65)),
        0.05,
    )
    st.session_state.append_source_block = st.toggle(
        "Answer citations", value=bool(st.session_state.get("append_source_block", True))
    )

    rebuild = st.button("Rebuild Index", use_container_width=True)
    st.caption("Secrets are read from `.env`, environment variables, or Streamlit secrets.")


st.title("Zyro Dynamics HR Help Desk")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_run" not in st.session_state:
    st.session_state.last_run = {}

startup_notice = st.empty()
startup_notice.info("Starting the HR assistant. The first cloud launch can take a minute...")

cfg = make_config()

try:
    pipeline = load_pipeline(config_cache_key(cfg), rebuild)
    ready_error = None
except Exception as exc:
    pipeline = None
    ready_error = exc

startup_notice.empty()

left, right = st.columns([2.2, 1], gap="large")

with left:
    if ready_error:
        st.error("The HR assistant could not finish starting. Open the details below to diagnose the deployment.")
        with st.expander("Startup error details", expanded=True):
            st.exception(ready_error)
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
            st.session_state.last_run = {
                "confidence": response.avg_confidence,
                "hyde": response.used_hyde,
                "refined": response.refined,
            }

with right:
    st.subheader("Retrieved Sources")
    if not st.session_state.last_sources:
        st.info("Sources appear after the first answered question.")
    for source in st.session_state.last_sources:
        st.markdown(
            """
<div class="source-box">
  <div class="source-title">{source_file} - chunk {chunk_id}</div>
  <div class="source-meta">confidence {confidence} - {retrieval_methods}</div>
  <div class="source-preview">{preview}</div>
</div>
            """.format(
                source_file=escape(source["source_file"]),
                chunk_id=escape(source["chunk_id"]),
                confidence=escape(source.get("confidence", "0.00")),
                retrieval_methods=escape(source.get("retrieval_methods", "")),
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
    if st.session_state.last_run:
        st.metric("Average confidence", "%.2f" % st.session_state.last_run["confidence"])
        st.metric("HyDE used", "yes" if st.session_state.last_run["hyde"] else "no")
        st.metric("Refined", "yes" if st.session_state.last_run["refined"] else "no")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_run = {}
        st.rerun()
