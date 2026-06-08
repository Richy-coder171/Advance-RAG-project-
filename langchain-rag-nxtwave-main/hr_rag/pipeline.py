from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except Exception:  # pragma: no cover - optional in older/minimal installs
    create_stuff_documents_chain = None

try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover - compatibility fallback
    try:
        from langchain_community.vectorstores import Chroma
    except Exception:  # pragma: no cover - pure Python fallback
        Chroma = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def load_environment() -> None:
    """Load .env files from common local run locations without overriding shell env."""
    if load_dotenv is None:
        return

    candidates = [Path.cwd() / ".env"]
    current = Path(__file__).resolve()
    candidates.extend(parent / ".env" for parent in current.parents)

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            load_dotenv(path, override=False)

    # LangSmith has used both env names across LangChain versions.
    if os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")


load_environment()


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?%?")

HR_KEYWORDS = {
    "hr",
    "human resource",
    "employee",
    "employment",
    "leave",
    "vacation",
    "holiday",
    "sick",
    "casual",
    "earned",
    "maternity",
    "paternity",
    "bereavement",
    "attendance",
    "timesheet",
    "payroll",
    "salary",
    "compensation",
    "bonus",
    "reimbursement",
    "expense",
    "benefit",
    "insurance",
    "medical",
    "provident",
    "pf",
    "gratuity",
    "tax",
    "probation",
    "notice",
    "resignation",
    "termination",
    "onboarding",
    "offboarding",
    "remote",
    "work from home",
    "wfh",
    "hybrid",
    "conduct",
    "compliance",
    "harassment",
    "disciplinary",
    "performance",
    "appraisal",
    "training",
    "learning",
    "travel",
    "policy",
    "manager",
    "department",
}

OBVIOUS_OUT_OF_SCOPE = {
    "weather",
    "stock price",
    "movie",
    "recipe",
    "cricket score",
    "football score",
    "write code",
    "debug code",
    "python error",
    "java error",
    "politics",
    "election",
    "shopping",
    "flight booking",
}

SENSITIVE_PATTERNS = [
    re.compile(r"\b(ssn|social security|aadhaar|pan number|bank account|passport)\b", re.I),
    re.compile(r"\b(show|tell|give|share|reveal)\b.*\b(employee|coworker|colleague).*\b(salary|pay|address|phone|email|record)\b", re.I),
    re.compile(r"\b(private key|api key|password|secret token)\b", re.I),
]


@dataclass
class HRRagConfig:
    """Configuration for the HR Help Desk RAG pipeline."""

    docs_path: str = "hr_docs"
    db_path: str = "chroma_hr_store"
    collection_name: str = "zyro_hr_policies"
    embedding_provider: str = "auto"
    llm_provider: str = "auto"
    chunk_size: int = 900
    chunk_overlap: int = 180
    retrieval_k: int = 6
    fetch_k: int = 24
    temperature: float = 0.0
    max_context_chars_per_chunk: int = 1800


@dataclass
class HRRagResponse:
    answer: str
    sources: List[Dict[str, str]]
    blocked: bool = False
    reason: Optional[str] = None
    retrieved_context: str = ""


class LocalHashEmbeddings(Embeddings):
    """Small offline embedding fallback.

    This is not as strong as OpenAI/Ollama embeddings, but it keeps the
    Kaggle notebook and Streamlit app runnable when no embedding API is set.
    It uses a normalized hashing vector over unigrams and bigrams.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        tokens = tokenize(text)
        features = tokens + ["%s_%s" % pair for pair in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little", signed=False)
            idx = raw % self.dim
            sign = 1.0 if (raw >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class KeywordIndex:
    """Lightweight BM25-style lexical retriever for policy wording."""

    def __init__(self, docs: Sequence[Document]) -> None:
        self.docs = list(docs)
        self.doc_tokens: List[List[str]] = [tokenize(doc.page_content) for doc in self.docs]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        df: Dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] += 1
        total = max(1, len(self.docs))
        self.idf = {
            token: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        scored: List[Tuple[Document, float]] = []
        k1 = 1.5
        b = 0.75
        for doc, tokens, doc_len in zip(self.docs, self.doc_tokens, self.doc_lengths):
            counts = Counter(tokens)
            score = 0.0
            for term, qtf in query_counts.items():
                if term not in counts:
                    continue
                tf = counts[term]
                denom = tf + k1 * (1 - b + b * (doc_len / max(1.0, self.avg_len)))
                score += self.idf.get(term, 0.0) * ((tf * (k1 + 1)) / denom) * qtf
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


class InMemoryVectorStore:
    """Pure Python vector store fallback when Chroma is unavailable."""

    def __init__(self, docs: Sequence[Document], embeddings: Embeddings) -> None:
        self.docs = list(docs)
        self.embeddings = embeddings
        self.vectors = embeddings.embed_documents([doc.page_content for doc in self.docs])

    @classmethod
    def from_documents(cls, docs: Sequence[Document], embeddings: Embeddings) -> "InMemoryVectorStore":
        return cls(docs, embeddings)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_vec = self.embeddings.embed_query(query)
        scored = [
            (doc, cosine_similarity(query_vec, vector))
            for doc, vector in zip(self.docs, self.vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _score in scored[:k]]

    def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20) -> List[Document]:
        query_vec = self.embeddings.embed_query(query)
        candidate_idxs = sorted(
            range(len(self.docs)),
            key=lambda idx: cosine_similarity(query_vec, self.vectors[idx]),
            reverse=True,
        )[:fetch_k]
        selected: List[int] = []
        lambda_mult = 0.65
        while candidate_idxs and len(selected) < k:
            best_idx = None
            best_score = -float("inf")
            for idx in candidate_idxs:
                relevance = cosine_similarity(query_vec, self.vectors[idx])
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(
                        cosine_similarity(self.vectors[idx], self.vectors[chosen])
                        for chosen in selected
                    )
                score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            candidate_idxs.remove(best_idx)
        return [self.docs[idx] for idx in selected]


class HRRagPipeline:
    """RAG pipeline for Zyro Dynamics HR policy Q&A."""

    def __init__(
        self,
        config: HRRagConfig,
        vectorstore,
        chunks: Sequence[Document],
        llm=None,
    ) -> None:
        self.config = config
        self.vectorstore = vectorstore
        self.chunks = list(chunks)
        self.keyword_index = KeywordIndex(self.chunks)
        self.llm = llm

    @classmethod
    def from_config(cls, config: Optional[HRRagConfig] = None, rebuild: bool = False) -> "HRRagPipeline":
        cfg = config or HRRagConfig()
        embeddings = build_embeddings(cfg.embedding_provider)
        db_path = Path(cfg.db_path)
        chunks_path = db_path / "chunks.jsonl"

        if rebuild or not db_path.exists():
            docs = load_policy_documents(cfg.docs_path)
            if not docs:
                raise ValueError(
                    "No HR policy documents found. Add .md, .txt, .pdf, .docx, .csv, or .json files to %s."
                    % cfg.docs_path
                )
            chunks = split_policy_documents(docs, cfg.chunk_size, cfg.chunk_overlap)
            if db_path.exists():
                shutil.rmtree(db_path)
            vectorstore = build_vectorstore(chunks, embeddings, db_path, cfg.collection_name)
            persist_chunks(chunks, chunks_path)
        else:
            chunks = load_persisted_chunks(chunks_path)
            if not chunks:
                docs = load_policy_documents(cfg.docs_path)
                chunks = split_policy_documents(docs, cfg.chunk_size, cfg.chunk_overlap)
            vectorstore = load_vectorstore_or_memory(chunks, embeddings, db_path, cfg.collection_name)

        llm = build_chat_model(cfg.llm_provider, cfg.temperature)
        return cls(cfg, vectorstore, chunks, llm=llm)

    def answer(self, question: str, chat_history: Optional[Sequence[Tuple[str, str]]] = None) -> HRRagResponse:
        guard_ok, reason = self._guardrail(question)
        if not guard_ok:
            return HRRagResponse(
                answer=reason or "I can only answer Zyro Dynamics HR policy questions.",
                sources=[],
                blocked=True,
                reason=reason,
            )

        docs = self.retrieve(question)
        if not docs:
            return HRRagResponse(
                answer=(
                    "I could not find this in the Zyro Dynamics HR policy documents. "
                    "Please contact the HR team for confirmation."
                ),
                sources=[],
                retrieved_context="",
            )

        context = self.format_context(docs)
        sources = source_dicts(docs)

        if self.llm is None:
            answer = self._extractive_answer(question, docs)
        else:
            answer = self._llm_answer(question, docs, context, chat_history or [])

        return HRRagResponse(
            answer=answer.strip(),
            sources=sources,
            retrieved_context=context,
        )

    def retrieve(self, question: str) -> List[Document]:
        fetch_k = max(self.config.fetch_k, self.config.retrieval_k)
        candidates: Dict[str, Tuple[Document, float]] = {}

        try:
            vector_docs = self.vectorstore.max_marginal_relevance_search(
                question,
                k=min(fetch_k, max(self.config.retrieval_k, 12)),
                fetch_k=max(fetch_k, 24),
            )
        except Exception:
            vector_docs = self.vectorstore.similarity_search(question, k=min(fetch_k, 24))

        for rank, doc in enumerate(vector_docs):
            key = doc_key(doc)
            candidates[key] = (doc, candidates.get(key, (doc, 0.0))[1] + 1.0 / (rank + 1))

        for rank, (doc, lexical_score) in enumerate(self.keyword_index.search(question, fetch_k)):
            key = doc_key(doc)
            rank_score = 0.85 / (rank + 1)
            combined = candidates.get(key, (doc, 0.0))[1] + rank_score + min(lexical_score / 20.0, 0.5)
            candidates[key] = (doc, combined)

        ranked = sorted(candidates.values(), key=lambda item: item[1], reverse=True)
        return [doc for doc, _score in ranked[: self.config.retrieval_k]]

    def format_context(self, docs: Sequence[Document]) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
            chunk_id = doc.metadata.get("chunk_id", "n/a")
            text = clean_text(doc.page_content)[: self.config.max_context_chars_per_chunk]
            parts.append("Source: %s | chunk %s\n%s" % (source, chunk_id, text))
        return "\n\n---\n\n".join(parts)

    def _llm_answer(
        self,
        question: str,
        docs: Sequence[Document],
        context: str,
        chat_history: Sequence[Tuple[str, str]],
    ) -> str:
        history_text = "\n".join(
            "Employee: %s\nAssistant: %s" % (human, assistant)
            for human, assistant in list(chat_history)[-4:]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are Zyro Dynamics HR Help Desk assistant. Answer employee HR policy "
                        "questions using only the provided policy context. If the answer is missing, "
                        "unclear, or unsupported, say that you could not find it in the Zyro Dynamics "
                        "HR policy documents and suggest contacting HR. Do not invent policy dates, "
                        "benefit amounts, eligibility rules, legal conclusions, or personal employee "
                        "data. Keep the answer concise, practical, and grounded. Cite every factual "
                        "policy claim with citations like [source_file chunk 3]."
                    ),
                ),
                (
                    "human",
                    (
                        "Conversation history:\n{history}\n\n"
                        "Policy context:\n{context}\n\n"
                        "Employee question: {question}\n\n"
                        "Answer:"
                    ),
                ),
            ]
        )

        if create_stuff_documents_chain is not None:
            document_prompt = PromptTemplate.from_template(
                "Source: {source_file} | chunk {chunk_id}\n{page_content}"
            )
            stuff_chain = create_stuff_documents_chain(
                self.llm,
                prompt,
                document_prompt=document_prompt,
            )
            return stuff_chain.invoke(
                {
                    "context": docs,
                    "history": history_text or "None",
                    "question": question,
                }
            )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"history": history_text or "None", "context": context, "question": question})

    def _extractive_answer(self, question: str, docs: Sequence[Document]) -> str:
        sentences = []
        query_terms = set(tokenize(question))
        for doc in docs:
            source = doc.metadata.get("source_file", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "n/a")
            for sentence in split_sentences(doc.page_content):
                overlap = len(query_terms & set(tokenize(sentence)))
                if overlap:
                    sentences.append((overlap, source, chunk_id, clean_text(sentence)))
        sentences.sort(key=lambda item: item[0], reverse=True)
        selected = sentences[:4]
        if not selected:
            selected = [
                (0, doc.metadata.get("source_file", "unknown"), doc.metadata.get("chunk_id", "n/a"), clean_text(doc.page_content)[:350])
                for doc in docs[:2]
            ]
        lines = [
            "I found the most relevant Zyro Dynamics HR policy text below. Configure GROQ_API_KEY, OPENAI_API_KEY, or Ollama for a generated answer."
        ]
        for _score, source, chunk_id, sentence in selected:
            lines.append("- %s [%s chunk %s]" % (sentence, source, chunk_id))
        return "\n".join(lines)

    def _guardrail(self, question: str) -> Tuple[bool, Optional[str]]:
        q = (question or "").strip()
        q_lower = q.lower()
        if len(q_lower.split()) < 2:
            return False, "Please ask a complete HR policy question."

        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(q):
                return (
                    False,
                    "I cannot help reveal credentials, sensitive personal data, or another employee's private records.",
                )

        if any(term in q_lower for term in OBVIOUS_OUT_OF_SCOPE):
            return False, "I can only answer questions about Zyro Dynamics HR policies and employee processes."

        if any(term in q_lower for term in HR_KEYWORDS):
            return True, None

        # Let retrieval handle borderline short workplace questions, but block
        # broad non-HR questions that carry no workplace signal.
        workplace_signals = {"company", "office", "team", "supervisor", "approval", "claim", "request"}
        if any(term in q_lower for term in workplace_signals):
            return True, None

        return False, "I can only answer questions about Zyro Dynamics HR policies and employee processes."


def load_policy_documents(docs_path: str) -> List[Document]:
    root = Path(docs_path)
    if not root.exists():
        return []

    docs: List[Document] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")):
        suffix = path.suffix.lower()
        loaded: List[Document] = []
        try:
            if suffix in {".md", ".txt"}:
                loaded = TextLoader(str(path), encoding="utf-8").load()
            elif suffix == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            elif suffix == ".docx":
                loaded = load_docx_as_documents(path)
            elif suffix == ".csv":
                loaded = load_csv_as_documents(path)
            elif suffix == ".json":
                loaded = load_json_as_documents(path)
        except Exception:
            loaded = []

        for doc in loaded:
            doc.metadata.update(
                {
                    "source_file": path.name,
                    "source_path": str(path),
                    "file_type": suffix.lstrip("."),
                }
            )
            if clean_text(doc.page_content):
                docs.append(doc)
    return docs


def split_policy_documents(docs: Sequence[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(list(docs))
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["chunk_chars"] = len(chunk.page_content)
    return chunks


def build_embeddings(provider: str = "auto") -> Embeddings:
    selected = (provider or "auto").lower()
    env_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
    if selected == "auto" and env_provider and env_provider != "auto":
        selected = env_provider
    if selected == "auto":
        if os.getenv("OPENAI_API_KEY"):
            selected = "openai"
        elif os.getenv("OLLAMA_EMBEDDING_MODEL"):
            selected = "ollama"
        else:
            selected = "hash"

    if selected == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    if selected == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    if selected == "hash":
        return LocalHashEmbeddings(dim=int(os.getenv("HASH_EMBEDDING_DIM", "768")))
    raise ValueError("Unsupported embedding provider: %s" % provider)


def build_vectorstore(
    chunks: Sequence[Document],
    embeddings: Embeddings,
    db_path: Path,
    collection_name: str,
):
    if Chroma is not None:
        try:
            return Chroma.from_documents(
                chunks,
                embedding=embeddings,
                persist_directory=str(db_path),
                collection_name=collection_name,
            )
        except Exception:
            pass
    return InMemoryVectorStore.from_documents(chunks, embeddings)


def load_vectorstore_or_memory(
    chunks: Sequence[Document],
    embeddings: Embeddings,
    db_path: Path,
    collection_name: str,
):
    if Chroma is not None and db_path.exists():
        try:
            return Chroma(
                persist_directory=str(db_path),
                embedding_function=embeddings,
                collection_name=collection_name,
            )
        except Exception:
            pass
    return InMemoryVectorStore.from_documents(chunks, embeddings)


def build_chat_model(provider: str = "auto", temperature: float = 0.0):
    selected = (provider or "auto").lower()
    env_provider = os.getenv("LLM_PROVIDER", "").lower()
    if selected == "auto" and env_provider and env_provider != "auto":
        selected = env_provider
    if selected == "auto":
        if os.getenv("GROQ_API_KEY"):
            selected = "groq"
        elif os.getenv("OPENAI_API_KEY"):
            selected = "openai"
        elif os.getenv("OLLAMA_LLM_MODEL"):
            selected = "ollama"
        else:
            selected = "none"

    if selected == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=temperature,
        )
    if selected == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )
    if selected == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_LLM_MODEL", "llama3.1"),
            temperature=temperature,
        )
    if selected in {"none", "extractive", "offline"}:
        return None
    raise ValueError("Unsupported LLM provider: %s" % provider)


def load_csv_as_documents(path: Path) -> List[Document]:
    import csv

    docs = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            text = "\n".join("%s: %s" % (key, value) for key, value in row.items() if value)
            docs.append(Document(page_content=text, metadata={"row": idx}))
    return docs


def load_json_as_documents(path: Path) -> List[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    docs = []
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            text = json.dumps(item, ensure_ascii=True, indent=2)
        else:
            text = str(item)
        docs.append(Document(page_content=text, metadata={"item": idx}))
    return docs


def load_docx_as_documents(path: Path) -> List[Document]:
    paragraphs: List[str] = []
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_bytes)
    for paragraph in root.findall(".//w:p", namespace):
        runs = []
        for node in paragraph.findall(".//w:t", namespace):
            if node.text:
                runs.append(node.text)
        text = clean_text("".join(runs))
        if text:
            paragraphs.append(text)

    return [Document(page_content="\n".join(paragraphs), metadata={})]


def persist_chunks(chunks: Sequence[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            payload = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def load_persisted_chunks(path: Path) -> List[Document]:
    if not path.exists():
        return []
    docs = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            docs.append(Document(page_content=payload["page_content"], metadata=payload.get("metadata", {})))
    return docs


def source_dicts(docs: Sequence[Document]) -> List[Dict[str, str]]:
    sources = []
    seen = set()
    for doc in docs:
        source = str(doc.metadata.get("source_file", doc.metadata.get("source", "unknown")))
        chunk_id = str(doc.metadata.get("chunk_id", "n/a"))
        key = (source, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source_file": source,
                "chunk_id": chunk_id,
                "file_type": str(doc.metadata.get("file_type", "")),
                "preview": clean_text(doc.page_content)[:240],
            }
        )
    return sources


def doc_key(doc: Document) -> str:
    source = str(doc.metadata.get("source_file", doc.metadata.get("source", "")))
    chunk_id = str(doc.metadata.get("chunk_id", ""))
    if source or chunk_id:
        return "%s:%s" % (source, chunk_id)
    return hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def split_sentences(text: str) -> Iterable[str]:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        sentence = clean_text(sentence)
        if len(sentence) >= 30:
            yield sentence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    denom = (math.sqrt(sum(v * v for v in left)) or 1.0) * (math.sqrt(sum(v * v for v in right)) or 1.0)
    return sum(a * b for a, b in zip(left, right)) / denom
