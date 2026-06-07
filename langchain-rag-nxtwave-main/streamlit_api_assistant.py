# streamlit-rag.py
"""
Single-file Streamlit app: Minimal RAG + Agent + Guardrails + Evaluation + LangSmith integration
Requirements (example): streamlit, langchain-ollama, langchain-chroma, langchain-community, chromadb, langsmith
"""
import os
import shutil
import json
import importlib
from pathlib import Path
from datetime import datetime
import traceback

import streamlit as st
from html import escape

import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# -------------------------
# LangChain + provider imports (v1+)
# -------------------------
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Optional LangSmith client (may not be installed)
try:
    from langsmith import Client as LangSmithClient
except Exception:
    LangSmithClient = None

# -------------------------
# Config defaults
# -------------------------
DEFAULT_CONFIG = {
    "docs_path": "api_docs",
    "db_path": "chroma_fixed_store",
    "llm_model": "llama3.1",      # pick model that supports tools
    "embedding_model": "nomic-embed-text",
    "chunk_size": 800,
    "chunk_overlap": 150,
    "retrieval_k": 5,
}

# -------------------------
# UI styling (light/professional)
# -------------------------
THEME = """
<style>
:root{--bg:#f4f6f8; --card:#ffffff; --muted:#6b7280; --accent:#0f62fe; --text:#0f1724}
body, .stApp { background: var(--bg); color:var(--text); font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto; }
.carbon-card { background:var(--card); padding:14px; border-radius:10px; box-shadow: 0 6px 18px rgba(2,6,23,0.06); border:1px solid rgba(0,0,0,0.04); margin-bottom:12px; }
.card-title { font-size:18px; font-weight:600; margin-bottom:8px; }
.small-muted { color:var(--muted); font-size:13px; }
.resp-box { background: linear-gradient(180deg, rgba(0,0,0,0.01), rgba(0,0,0,0.00)); border-radius:8px; padding:12px; border:1px solid rgba(0,0,0,0.04); white-space:pre-wrap; font-family: ui-monospace, monospace; }
.kv{ display:flex; gap:12px; align-items:center; margin-bottom:6px; }
.kv .key{ color:var(--muted); width:160px } .kv .val{ color:var(--text); font-weight:600 }
.trace-link { color: var(--accent); font-weight:600; text-decoration:none; }
</style>
"""
st.set_page_config(page_title="API Docs Assistant", layout="wide")
st.markdown(THEME, unsafe_allow_html=True)

# -------------------------
# Helpers: LangSmith client & finding runs (best-effort)
# -------------------------
def get_langsmith_client(api_key=None):
    if LangSmithClient is None:
        return None
    key = api_key or os.getenv("LANGSMITH_API_KEY", "")
    if not key:
        return None
    try:
        # Some SDK versions accept key argument, some use env var
        try:
            client = LangSmithClient(api_key=key)
        except TypeError:
            client = LangSmithClient()
        return client
    except Exception:
        return None

def find_latest_run(client, project, filter_name_substr=None, limit=20):
    """Try to find a recent run associated with this project. Return dict with url/run_id if found."""
    if client is None:
        return None
    try:
        # Attempt several SDK call patterns
        runs = None
        if hasattr(client, "runs") and hasattr(client.runs, "list"):
            runs = list(client.runs.list(project=project, limit=limit))
        elif hasattr(client, "list_runs"):
            runs = list(client.list_runs(project=project, limit=limit))
        elif hasattr(client, "list_runs_v2"):
            runs = list(client.list_runs_v2(project=project, limit=limit))
        else:
            runs = []
        if not runs:
            return None
        # If filter_name_substr provided try to match
        if filter_name_substr:
            for r in runs:
                name = getattr(r, "name", None) or (r.get("name") if isinstance(r, dict) else None)
                if name and filter_name_substr in name:
                    run_id = getattr(r, "id", None) or getattr(r, "run_id", None) or (r.get("run_id") if isinstance(r, dict) else None)
                    url = getattr(r, "url", None) or (f"https://smith.langchain.com/o/default/projects/p/{project}/runs/{run_id}" if run_id else None)
                    return {"run_id": run_id, "url": url, "name": name}
        # Otherwise return first
        r = runs[0]
        run_id = getattr(r, "id", None) or getattr(r, "run_id", None) or (r.get("run_id") if isinstance(r, dict) else None)
        url = getattr(r, "url", None) or (f"https://smith.langchain.com/o/default/projects/p/{project}/runs/{run_id}" if run_id else None)
        name = getattr(r, "name", None) or (r.get("name") if isinstance(r, dict) else None)
        return {"run_id": run_id, "url": url, "name": name}
    except Exception:
        return None

# -------------------------
# Utility functions for documents & vectorstore
# -------------------------
@st.cache_resource
def load_documents(docs_path: str):
    """Load .md and .pdf docs from docs_path."""
    docs = []
    p = Path(docs_path)
    # Markdown/text
    for f in sorted(p.glob("*.md")):
        try:
            loaded = TextLoader(str(f)).load()
            for d in loaded:
                d.metadata.update({"source_file": f.name, "full_path": str(f)})
            docs.extend(loaded)
        except Exception as e:
            # skip problematic file
            print(f"Failed to load {f}: {e}")
    # PDFs
    for f in sorted(p.glob("*.pdf")):
        try:
            loaded = PyPDFLoader(str(f)).load()
            for d in loaded:
                d.metadata.update({"source_file": f.name, "full_path": str(f)})
            docs.extend(loaded)
        except Exception as e:
            print(f"Failed to load PDF {f}: {e}")
    return docs

def build_vectorstore(docs, embedding_model, db_path, chunk_size, chunk_overlap):
    """Build and persist Chroma store. This is called only by user action (rebuild)."""
    # chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    for i, s in enumerate(splits):
        s.metadata["chunk_id"] = i
    # clear existing
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    embeddings = OllamaEmbeddings(model=embedding_model)
    vs = Chroma.from_documents(splits, embedding=embeddings, persist_directory=db_path)
    # try to persist if client supports it
    try:
        vs._client.persist()
    except Exception:
        pass
    return vs, len(splits)

@st.cache_resource
def load_vectorstore(db_path, embedding_model):
    """Load existing chroma store in read-only way (if supported)."""
    # instantiate Chroma with persist_directory and embedding function
    embeddings = OllamaEmbeddings(model=embedding_model)
    # Chroma wrapper may open existing DB
    vs = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return vs

# -------------------------
# RAG chain setup
# -------------------------
def build_rag_chain(vectorstore, llm_model, retrieval_k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
    llm = ChatOllama(model=llm_model, temperature=0.1)

    def format_context_for_chain(docs):
        out = []
        for d in docs:
            snippet = d.page_content[:400].replace("\n", " ")
            out.append(f"Source: {d.metadata.get('source_file')} (chunk {d.metadata.get('chunk_id')})\n{snippet}")
        return "\n\n---\n\n".join(out)

    def retrieve_and_format(q):
        docs = retriever.invoke(q)
        return format_context_for_chain(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """ You are a developer assistant. Use ONLY the provided context.
                    If the answer is not present, respond: "I don't have that information in the documentation."
                    Cite chunks as: [chunk X from filename]."""),

        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    rag_chain = (
        {
            "context": retrieve_and_format,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

# -------------------------
# Tools & Agent
# -------------------------
from typing import List

@tool
def calculator_tool(expr: str) -> str:
    """
    Safely evaluate a basic arithmetic expression (integers/floats + +-*/ and parentheses).
    Examples: "2+2", "1000 - 234", "(10.5*3)/2"
    Returns the numeric result as a string or an error message on invalid input.
    """
    if expr is None:
        return "Error: empty expression"
    allowed = set("0123456789+-*/(). eE")
    if any(c not in allowed for c in expr):
        return "Invalid characters in expression. Only digits, + - * / ( ) . and spaces allowed."
    try:
        # safe eval context
        val = eval(expr, {"__builtins__": {}}, {})
        return str(val)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def doc_search_tool(query: str, k: int = 3) -> str:
    """
    Retrieve top-k matching documentation chunks for the given query.
    Returns a formatted string with chunk id, source filename and preview.
    This tool relies on the 'retriever' being present in st.session_state (populated after index build).
    """
    if not query:
        return "No query provided."
    try:
        retriever = st.session_state.get("retriever")
        if retriever is None:
            return "Retriever not available (build the index first)."
        # call retriever; prefer invoke if available
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)[:k]
        else:
            docs = retriever.get_relevant_documents(query)[:k]  # fallback for some stores
        out_lines: List[str] = []
        for d in docs:
            src = d.metadata.get("source_file", "unknown")
            cid = d.metadata.get("chunk_id", "n/a")
            preview = d.page_content[:300].replace("\n", " ")
            out_lines.append(f"[chunk {cid}] {src}: {preview}")
        return "\n\n".join(out_lines) if out_lines else "No documents retrieved."
    except Exception as e:
        return f"doc_search_tool error: {e}"


# Agent factory
def create_tool_agent(llm_model, tools):
    llm = ChatOllama(model=llm_model, temperature=0.1)
    agent = create_agent(model=llm, tools=tools, system_prompt="""
You are an API documentation assistant. Use doc_search for looking up docs and calculator for math.
Provide concise final answers and cite sources when applicable.
""")
    return agent

# -------------------------
# Guardrails
# -------------------------
BLOCKED_KEYWORDS = {"password", "secret", "private key", "admin", "token", "ssn"}

def apply_guardrails(q: str):
    ql = q.lower()
    for b in BLOCKED_KEYWORDS:
        if b in ql:
            return False, f"Blocked keyword detected: '{b}'"
    if len(ql.split()) < 3:
        return False, "Query too short. Please provide more context."
    return True, None

# -------------------------
# Evaluation: retrieval metrics
# -------------------------
def retrieval_metrics(retriever, query, ground_truth_files, k=5):
    docs = retriever.invoke(query)[:k]
    retrieved_files = [d.metadata.get("source_file") for d in docs]
    relevant = ground_truth_files
    recall = len(set(retrieved_files) & set(relevant)) / len(relevant) if relevant else None
    precision = len(set(retrieved_files) & set(relevant)) / max(1, len(retrieved_files)) if retrieved_files else None
    return {"recall@k": recall, "precision@k": precision, "retrieved": retrieved_files}

# -------------------------
# Display helpers
# -------------------------
def format_context_display(docs):
    out = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        cid = d.metadata.get("chunk_id", "n/a")
        preview = d.page_content[:400].replace("\n", " ")
        out.append(f"[chunk {cid}] {src}\n{preview}\n---")
    return "\n\n".join(out)

# -------------------------
# Streamlit UI
# -------------------------
st.sidebar.title("System Controls")

# Config inputs
docs_path = st.sidebar.text_input("Docs folder", DEFAULT_CONFIG["docs_path"])
db_path = st.sidebar.text_input("Chroma DB path", DEFAULT_CONFIG["db_path"])
llm_model = st.sidebar.text_input("LLM model (ChatOllama)", DEFAULT_CONFIG["llm_model"])
embedding_model = st.sidebar.text_input("Embedding model", DEFAULT_CONFIG["embedding_model"])
chunk_size = st.sidebar.number_input("Chunk size", value=DEFAULT_CONFIG["chunk_size"], step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", value=DEFAULT_CONFIG["chunk_overlap"], step=50)
retrieval_k = st.sidebar.number_input("Retriever k", value=DEFAULT_CONFIG["retrieval_k"], step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("LangSmith (optional)")
ls_key = st.sidebar.text_input("LangSmith API key", type="password", value=os.getenv("LANGSMITH_API_KEY", ""))
ls_project = st.sidebar.text_input("LangSmith project", value=os.getenv("LANGCHAIN_PROJECT", "api-docs-assistant"))
enable_langsmith = st.sidebar.checkbox("Enable LangSmith tracing", value=False)

#if enable_langsmith and ls_key:
#    os.environ["LANGCHAIN_TRACING_V2"] = "true"
#    os.environ["LANGSMITH_API_KEY"] = ls_key
#    os.environ["LANGCHAIN_PROJECT"] = ls_project
#    st.sidebar.success("LangSmith env set for this session")


# Persist LangSmith settings across Streamlit refreshes
if "langsmith_enabled" not in st.session_state:
    st.session_state.langsmith_enabled = False

if "langsmith_key" not in st.session_state:
    st.session_state.langsmith_key = None

if "langsmith_project" not in st.session_state:
    st.session_state.langsmith_project = None

# User toggles LangSmith
if enable_langsmith and ls_key:
    st.session_state.langsmith_enabled = True
    st.session_state.langsmith_key = ls_key
    st.session_state.langsmith_project = ls_project
else:
    st.session_state.langsmith_enabled = False

# Apply environment vars ONLY once (first run)
if st.session_state.langsmith_enabled:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = st.session_state.langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = st.session_state.langsmith_project


st.sidebar.markdown("---")
st.sidebar.markdown("Index control")
if "index_built" not in st.session_state:
    st.session_state.index_built = False
if st.button("Initialize / Rebuild Index"):
    # build index now
    try:
        st.sidebar.info("Loading documents...")
        docs = load_documents(docs_path)
        if not docs:
            st.sidebar.error("No docs found - place .md/.pdf in folder and try again.")
        else:
            with st.spinner("Building vectorstore (this may take a bit)..."):
                vs, num_chunks = build_vectorstore(docs, embedding_model, db_path, chunk_size, chunk_overlap)
                st.session_state.vectorstore = vs
                st.session_state.retriever = vs.as_retriever(search_kwargs={"k": retrieval_k})
                st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(vs, llm_model, retrieval_k)
                st.session_state.index_built = True
                st.success(f"Index built with {num_chunks} chunks")
    except Exception as e:
        st.sidebar.error(f"Failed to build index: {e}")
        st.sidebar.exception(traceback.format_exc())

# If index wasn't built but DB exists, try to load read-only
if not st.session_state.get("index_built", False) and os.path.exists(db_path):
    try:
        vs = load_vectorstore(db_path, embedding_model)
        st.session_state.vectorstore = vs
        st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(vs, llm_model, retrieval_k)
        st.session_state.index_built = True
        st.sidebar.success("Loaded existing index")
    except Exception as e:
        st.sidebar.warning("Could not load existing index automatically. Rebuild via button if needed.")
        print("Load vectorstore error:", e)

st.sidebar.markdown("---")
st.sidebar.markdown("Agent control")
if "agent" not in st.session_state and st.session_state.get("index_built", False):
    try:
        agent = create_tool_agent(llm_model, tools=[calculator_tool, doc_search_tool])
        st.session_state.agent = agent
    except Exception as e:
        st.sidebar.warning("Agent creation failed (model may not support tools).")
        print("Agent create error:", e)

if st.sidebar.button("Reset Agent Memory"):
    st.session_state.agent_thread_id = "thread-" + os.urandom(6).hex()
    st.sidebar.success("Agent thread reset")

# LangSmith session trace storage
if "langsmith_runs" not in st.session_state:
    st.session_state.langsmith_runs = []

# Basic header + instructions
st.title("📘 API Documentation Assistant")
st.markdown("Ask questions (RAG), run agent tasks, inspect retrieved chunks, and view LangSmith traces.")

# Tabs for UI
tab_rag, tab_agent, tab_guard, tab_retrieval, tab_eval, tab_traces = st.tabs([
    "RAG Q&A", "Agent", "Guardrails", "Retrieved Chunks", "Evaluation / Retrieval Tests", "LangSmith Traces"
])

# -------------------------
# Tab: RAG Q&A
# -------------------------
with tab_rag:
    st.subheader("Retrieval-Augmented Generation")
    q = st.text_input("Ask a question about the docs", key="rag_input")
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Ask (RAG)"):
            if not st.session_state.get("index_built", False):
                st.error("Index not built. Initialize index in sidebar first.")
            else:
                # guardrails
                ok, err = apply_guardrails(q)
                if not ok:
                    st.error(err)
                else:
                    with st.spinner("Running RAG..."):
                        try:
                            # Provide run_name in config so LangSmith may associate a run
                            run_name = f"RAG: {q[:80]}"
                            # Many rag_chain.invoke accept config; call with best-effort
                            try:
                                #resp = st.session_state.rag_chain.invoke(q, config={"metadata": {"run_name": run_name}})
                                resp = st.session_state.rag_chain.invoke(q)
                            except TypeError:
                                resp = st.session_state.rag_chain.invoke(q)
                            st.markdown("<div class='carbon-card'>", unsafe_allow_html=True)
                            st.markdown("<div class='card-title'>Answer</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='resp-box'>{escape(str(resp))}</div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # retrieved context
                            docs = st.session_state.retriever.invoke(q)
                            ctx = format_context_display(docs)
                            st.text_area("Retrieved (top k)", ctx, height=240)

                            # LangSmith: try to find latest run and record
                            if enable_langsmith and ls_key:
                                client = get_langsmith_client(ls_key)
                                found = find_latest_run(client, ls_project, filter_name_substr=q[:40])
                                if found:
                                    st.session_state.langsmith_runs.append({
                                        "query": q,
                                        "url": found.get("url"),
                                        "run_id": found.get("run_id"),
                                        "ts": datetime.now().isoformat()
                                    })
                                    st.success("Trace recorded in session (LangSmith).")
                        except Exception as e:
                            st.error(f"RAG failed: {e}")
                            st.exception(traceback.format_exc())
    with col2:
        st.markdown("## Quick actions")
        if st.button("Show retrieved sources (last query)"):
            try:
                if "retriever" in st.session_state:
                    docs = st.session_state.retriever.invoke(q)
                    st.write(format_context_display(docs))
                else:
                    st.info("No retriever available")
            except Exception as e:
                st.error("Retriever error")
                st.exception(traceback.format_exc())

# -------------------------
# Tab: Agent
# -------------------------
with tab_agent:
    st.subheader("Agent (tools + memory)")
    agent_input = st.text_input("Agent instruction", key="agent_input")
    c1, c2 = st.columns([3,1])
    with c1:
        if st.button("Run Agent"):
            if not st.session_state.get("index_built", False):
                st.error("Index not built.")
            else:
                ok, err = apply_guardrails(agent_input)
                if not ok:
                    st.error(err)
                else:
                    with st.spinner("Running agent..."):
                        try:
                            agent = st.session_state.get("agent")
                            if agent is None:
                                st.error("Agent not available (model may not support tools).")
                            else:
                                cfg = {"configurable": {"thread_id": st.session_state.get("agent_thread_id", "thread-default")}}
                                # Many agents expose .invoke
                                try:
                                    out = agent.invoke({"messages": [{"role": "user", "content": agent_input}]}, cfg)
                                except TypeError:
                                    # fallback to run or call
                                    try:
                                        out = agent.run(agent_input)
                                    except Exception:
                                        out = agent(agent_input)
                                # extract final text
                                final = ""
                                if isinstance(out, dict) and "messages" in out:
                                    last = out["messages"][-1]
                                    final = getattr(last, "content", last.get("content") if isinstance(last, dict) else str(last))
                                else:
                                    final = str(out)
                                st.markdown("<div class='carbon-card'>", unsafe_allow_html=True)
                                st.markdown("<div class='card-title'>Agent result</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='resp-box'>{escape(final)}</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                                # LangSmith trace capture
                                if enable_langsmith and ls_key:
                                    client = get_langsmith_client(ls_key)
                                    found = find_latest_run(client, ls_project, filter_name_substr=agent_input[:40])
                                    if found:
                                        st.session_state.langsmith_runs.append({
                                            "query": agent_input,
                                            "url": found.get("url"),
                                            "run_id": found.get("run_id"),
                                            "ts": datetime.now().isoformat()
                                        })
                                        st.success("Agent run recorded in session traces.")
                        except Exception as e:
                            st.error("Agent error")
                            st.exception(traceback.format_exc())
    with c2:
        if st.button("Reset Agent Thread"):
            st.session_state.agent_thread_id = "thread-" + os.urandom(6).hex()
            st.success("Agent thread id reset (memory cleared)")

# -------------------------
# Tab: Guardrails
# -------------------------
with tab_guard:
    st.subheader("Guardrails")
    gq = st.text_input("Enter query to test guardrails", key="guard_input")
    if st.button("Test"):
        ok, err = apply_guardrails(gq)
        if not ok:
            st.error(err)
        else:
            st.success("Query passes guardrails.")

# -------------------------
# Tab: Retrieved Chunks viewer
# -------------------------
with tab_retrieval:
    st.subheader("Retrieved Chunks Viewer")
    rq = st.text_input("Query to inspect retrieved chunks", key="ret_q")
    if st.button("Inspect"):
        try:
            docs = st.session_state.retriever.invoke(rq)
            st.text_area("Top retrieved chunks", format_context_display(docs), height=420)
        except Exception as e:
            st.error("Retriever error")
            st.exception(traceback.format_exc())

# -------------------------
# Tab: Evaluation & retrieval tests
# -------------------------
with tab_eval:
    st.subheader("Retrieval Evaluation / Tests")
    st.markdown("Provide a small ground-truth mapping to run recall/precision tests on a few queries.")
    gt_json = st.text_area("Ground-truth mapping (JSON). Format: {\"query\": [\"file1.md\", \"file2.md\"]}", height=120, value='{"How do I authenticate?": ["authentication_guide.md"], "What is rate limit?": ["api_guide.md"]}')
    if st.button("Run retrieval evaluation"):
        try:
            gt = json.loads(gt_json)
            rows = []
            for q_, files in gt.items():
                metrics = retrieval_metrics(st.session_state.retriever, q_, files, k=retrieval_k)
                rows.append((q_, metrics))
            for q_, m in rows:
                st.markdown(f"**Q:** {q_}")
                st.markdown(f"- Recall@{retrieval_k}: {m['recall@k']}")
                st.markdown(f"- Precision@{retrieval_k}: {m['precision@k']}")
                st.markdown(f"- Retrieved: {m['retrieved']}")
        except Exception as e:
            st.error("Evaluation failed")
            st.exception(traceback.format_exc())

# -------------------------
# Tab: LangSmith Traces
# -------------------------
with tab_traces:
    st.subheader("LangSmith Trace Dashboard (session)")
    st.markdown("This shows traces captured during this Streamlit session (best-effort).")
    if len(st.session_state.langsmith_runs) == 0:
        st.info("No traces captured yet. Run RAG/Agent queries to create traces (if LangSmith is enabled).")
    else:
        for run in reversed(st.session_state.langsmith_runs[-20:]):
            qdisplay = run.get("query", "")[:160]
            ts = run.get("ts", "")
            url = run.get("url", "")
            st.markdown(f"**{escape(qdisplay)}**  ·  <span style='color:var(--muted);font-size:12px'>{ts}</span>", unsafe_allow_html=True)
            if url:
                st.markdown(f"[Open Trace ↗]({url})")
            st.markdown("---")
    st.markdown("### Open project dashboard")
    project = ls_project or os.getenv("LANGCHAIN_PROJECT", "api-docs-assistant")
    st.markdown(f"[Open LangSmith Project ↗](https://smith.langchain.com/o/default/projects/p/{project})")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Single-file RAG + Agent + Guardrails + LangSmith — built with LangChain v1+")
