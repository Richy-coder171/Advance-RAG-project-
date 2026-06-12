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
from langsmith import traceable

from .official_corpus import validate_official_corpus

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except Exception:  # pragma: no cover - optional in older/minimal installs
    try:
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    except Exception:  # pragma: no cover - optional in minimal installs
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
    if os.getenv("LANGCHAIN_PROJECT") and not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")
    if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")


load_environment()


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?%?")
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "which",
    "who",
    "with",
}

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
    "stock option",
    "esop",
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
    "apply for a job",
    "recruitment process",
    "hiring process",
    "product features",
    "revenue",
    "financial results",
    "performing financially",
    "profit",
    "ebitda",
}

SENSITIVE_PATTERNS = [
    re.compile(r"\b(ssn|social security|aadhaar|pan number|bank account|passport)\b", re.I),
    re.compile(r"\b(show|tell|give|share|reveal)\b.*\b(employee|coworker|colleague).*\b(salary|pay|address|phone|email|record)\b", re.I),
    re.compile(r"\b(private key|api key|password|secret token)\b", re.I),
]

EXTERNAL_ORGANIZATION_PATTERNS = [
    re.compile(r"\b(zoho|freshworks|salesforce)\b", re.I),
    re.compile(r"\b(other|another|different|competitor)\s+(company|companies|organization|employer)s?\b", re.I),
    re.compile(r"\bcompare\b.*\b(company|companies|employer|policy|policies)\b", re.I),
]

LEGAL_ADVICE_PATTERNS = [
    re.compile(r"\b(legal advice|legal opinion|act as (?:my|a) lawyer)\b", re.I),
    re.compile(r"\b(can|should)\s+i\s+(sue|file a lawsuit|take legal action)\b", re.I),
]

POLICY_SOURCE_ROUTES = [
    (("work from home", "wfh", "hybrid", "full remote", "ad-hoc wfh", "emergency wfh"), "03_Work_From_Home_Policy.pdf"),
    (("earned leave", "sick leave", "maternity leave", "paternity leave", "leave"), "02_Leave_Policy.pdf"),
    (("performance review", "annual performance review", "apr", "pip", "promotion", "rating"), "05_Performance_Review_Policy.pdf"),
    (("salary", "payroll", "ctc", "bonus", "insurance", "benefit", "provident fund", "gratuity", "esop"), "06_Compensation_and_Benefits_Policy.pdf"),
    (("travel", "expense", "reimbursement", "per diem"), "10_Travel_and_Expense_Policy.pdf"),
    (("onboarding", "probation", "notice period", "resignation", "separation", "full and final"), "09_Onboarding_and_Separation_Policy.pdf"),
    (("sexual harassment", "posh", "icc"), "08_Prevention_of_Sexual_Harassment_Policy.pdf"),
    (("data security", "password", "device", "laptop", "vpn"), "07_IT_and_Data_Security_Policy.pdf"),
    (("code of conduct", "disciplinary", "ethics", "conflict of interest"), "04_Code_of_Conduct.pdf"),
]


def answer_style_instruction(question: str) -> str:
    """Return concise answer-format guidance based on question intent."""
    q = clean_text(question).lower()
    is_multi_part = " and " in q or q.count("?") > 1
    completeness = (
        " Cover every requested part. Include the exact value, amount, "
        "limit, date, deadline, eligibility rule, or condition for each part."
        if is_multi_part
        else ""
    )

    if re.search(r"\b(timeline|schedule|stages?|steps?|process|procedure)\b", q):
        return (
            "Use numbered steps in chronological order. Include every stage, date, deadline, or owner "
            "requested by the question. Use plain text only." + completeness
        )

    if re.search(r"\b(how many|how much)\b", q) or re.search(r"\bdays?\b", q):
        return (
            "State the exact number, amount, or day count from the context with its condition "
            "in one or two sentences." + completeness
        )

    if re.search(r"\b(can i|can we|am i|are we|eligible|allowed|permitted|qualify|entitled)\b", q):
        return (
            "Start with Yes or No when the context supports it, then state the eligibility rule, "
            "condition, exception, or approval requirement directly in one or two sentences." + completeness
        )

    if re.search(r"\b(how to|process|procedure|apply|claim|request|submit|report|file)\b", q):
        return (
            "Use numbered steps. Keep each step short and include only actions stated in the context." + completeness
        )

    if re.search(r"\b(what is|what are|define|definition|meaning)\b", q):
        return (
            "Give a direct answer using the policy wording. Add only the scope, eligibility, or conditions "
            "that the question asks for." + completeness
        )

    return "Answer directly in one to three sentences using only the relevant policy facts." + completeness


@dataclass
class HRRagConfig:
    """Configuration for the HR Help Desk RAG pipeline."""

    docs_path: str = "hr_docs/official"
    db_path: str = "chroma_zyro_official_store"
    collection_name: str = "zyro_hr_policies"
    embedding_provider: str = "auto"
    llm_provider: str = "auto"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 8
    fetch_k: int = 60
    temperature: float = 0.0
    max_context_chars_per_chunk: int = 1800
    vector_weight: float = 0.65
    keyword_weight: float = 0.35
    rrf_k: int = 60
    min_confidence: float = 0.35
    min_retrieved_chunks: int = 2
    max_chunks_per_source: int = 2
    enable_hyde: bool = True
    enable_self_critique: bool = True
    critique_confidence_threshold: float = 0.55
    append_source_block: bool = True
    allow_extractive_fallback: bool = True

    def __post_init__(self) -> None:
        self.vector_weight = max(0.0, min(1.0, self.vector_weight))
        self.keyword_weight = max(0.0, min(1.0, self.keyword_weight))
        total_weight = self.vector_weight + self.keyword_weight
        if total_weight <= 0:
            self.vector_weight, self.keyword_weight = 0.6, 0.4
        else:
            self.vector_weight /= total_weight
            self.keyword_weight /= total_weight
        self.min_confidence = max(0.0, min(1.0, self.min_confidence))
        self.critique_confidence_threshold = max(0.0, min(1.0, self.critique_confidence_threshold))
        self.min_retrieved_chunks = max(1, self.min_retrieved_chunks)
        self.max_chunks_per_source = max(1, self.max_chunks_per_source)


@dataclass
class HRRagResponse:
    answer: str
    sources: List[Dict[str, str]]
    blocked: bool = False
    reason: Optional[str] = None
    retrieved_context: str = ""
    avg_confidence: float = 0.0
    used_hyde: bool = False
    refined: bool = False
    critique_rating: Optional[str] = None


def trace_answer_inputs(inputs: Dict[str, object]) -> Dict[str, object]:
    history = inputs.get("chat_history") or []
    return {
        "question": inputs.get("question", ""),
        "chat_history_turns": len(history) if isinstance(history, Sequence) else 0,
        "force_refine": bool(inputs.get("force_refine", False)),
    }


def trace_answer_output(response: HRRagResponse) -> Dict[str, object]:
    return {
        "answer": response.answer,
        "blocked": response.blocked,
        "reason": response.reason,
        "avg_confidence": response.avg_confidence,
        "used_hyde": response.used_hyde,
        "refined": response.refined,
        "sources": response.sources,
    }


def trace_retrieval_inputs(inputs: Dict[str, object]) -> Dict[str, object]:
    return {"question": inputs.get("question", "")}


def trace_retrieval_output(docs: Sequence[Document]) -> List[Dict[str, object]]:
    return [
        {
            "source_file": doc.metadata.get("source_file", doc.metadata.get("source", "unknown")),
            "chunk_id": doc.metadata.get("chunk_id", "n/a"),
            "retrieval_rank": doc.metadata.get("retrieval_rank", "n/a"),
            "retrieval_confidence": doc.metadata.get("retrieval_confidence", 0.0),
            "preview": clean_text(doc.page_content)[:400],
        }
        for doc in docs
    ]


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
        validate_official_corpus(cfg.docs_path)
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

    @traceable(
        name="zyro_hr_rag_answer",
        run_type="chain",
        process_inputs=trace_answer_inputs,
        process_outputs=trace_answer_output,
    )
    def answer(
        self,
        question: str,
        chat_history: Optional[Sequence[Tuple[str, str]]] = None,
        force_refine: bool = False,
    ) -> HRRagResponse:
        guard_ok, reason = self._guardrail(question)
        if not guard_ok:
            return HRRagResponse(
                answer=reason or "I can only answer Zyro Dynamics HR policy questions.",
                sources=[],
                blocked=True,
                reason=reason,
            )

        grounded_question = normalize_company_aliases(question)
        docs = self.retrieve(remove_company_aliases(question))
        if not docs:
            return HRRagResponse(
                answer="I could not find this information in the Zyro Dynamics HR policy documents.",
                sources=[],
                retrieved_context="",
            )

        context = self.format_context(docs)
        sources = source_dicts(docs)
        avg_confidence = average_confidence(docs)
        used_hyde = any(bool(doc.metadata.get("used_hyde")) for doc in docs)
        refined = False
        critique_rating = None

        if self.llm is None:
            answer = self._extractive_answer(grounded_question, docs)
        else:
            try:
                answer = self._llm_answer(grounded_question, docs, context, chat_history or [])
                should_refine = self.config.enable_self_critique and (
                    force_refine or avg_confidence < self.config.critique_confidence_threshold
                )
                if should_refine:
                    answer, critique_rating = self._self_critique(grounded_question, context, answer)
                    refined = True
            except Exception:
                if not self.config.allow_extractive_fallback:
                    raise
                answer = self._extractive_answer(grounded_question, docs)
                critique_rating = "EXTRACTIVE_FALLBACK"

        if self.config.append_source_block:
            answer = append_citation_block(answer, sources)

        return HRRagResponse(
            answer=answer.strip(),
            sources=sources,
            retrieved_context=context,
            avg_confidence=avg_confidence,
            used_hyde=used_hyde,
            refined=refined,
            critique_rating=critique_rating,
        )

    @traceable(
        name="zyro_hr_policy_retrieval",
        run_type="retriever",
        process_inputs=trace_retrieval_inputs,
        process_outputs=trace_retrieval_output,
    )
    def retrieve(self, question: str) -> List[Document]:
        fetch_k = max(self.config.fetch_k, self.config.retrieval_k)
        vector_query, used_hyde = self._retrieval_query(question)

        try:
            vector_docs = self.vectorstore.max_marginal_relevance_search(
                vector_query,
                k=min(fetch_k, max(self.config.retrieval_k, 12)),
                fetch_k=max(fetch_k, 24),
            )
        except Exception:
            try:
                vector_docs = self.vectorstore.similarity_search(vector_query, k=min(fetch_k, 24))
            except Exception:
                self.vectorstore = InMemoryVectorStore.from_documents(
                    self.chunks,
                    LocalHashEmbeddings(dim=int(os.getenv("HASH_EMBEDDING_DIM", "768"))),
                )
                vector_docs = self.vectorstore.max_marginal_relevance_search(
                    vector_query,
                    k=min(fetch_k, max(self.config.retrieval_k, 12)),
                    fetch_k=max(fetch_k, 24),
                )

        keyword_docs = [doc for doc, _score in self.keyword_index.search(question, fetch_k)]
        fused = weighted_reciprocal_rank_fusion(
            ranked_lists=[
                ("vector_mmr_hyde" if used_hyde else "vector_mmr", vector_docs, self.config.vector_weight),
                ("bm25", keyword_docs, self.config.keyword_weight),
            ],
            rrf_k=self.config.rrf_k,
        )
        source_hints = infer_policy_source_hints(question)
        if source_hints:
            fused.sort(
                key=lambda item: (
                    str(item[0].metadata.get("source_file", item[0].metadata.get("source", ""))) not in source_hints,
                    -query_doc_overlap(question, item[0]),
                    -item[1],
                )
            )
        if source_hints and needs_adjacent_context(question):
            fused = expand_with_adjacent_policy_chunks(fused, self.chunks, source_hints)
        if source_hints:
            routed = [
                item
                for item in fused
                if str(item[0].metadata.get("source_file", item[0].metadata.get("source", ""))) in source_hints
            ]
            if len(routed) >= self.config.min_retrieved_chunks:
                fused = routed

        selected: List[Document] = []
        source_counts: Dict[str, int] = defaultdict(int)
        source_chunk_limit = self.config.max_chunks_per_source
        if needs_adjacent_context(question):
            source_chunk_limit = max(source_chunk_limit, 3)
        for retrieval_rank, (doc, score, confidence, methods) in enumerate(fused, start=1):
            if confidence < self.config.min_confidence and len(selected) >= self.config.min_retrieved_chunks:
                continue
            source = str(doc.metadata.get("source_file", doc.metadata.get("source", "unknown")))
            if (
                source_counts[source] >= source_chunk_limit
                and len(selected) >= self.config.min_retrieved_chunks
            ):
                continue
            source_counts[source] += 1
            selected.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "retrieval_rank": retrieval_rank,
                        "retrieval_score": round(score, 6),
                        "retrieval_confidence": round(confidence, 4),
                        "retrieval_methods": ", ".join(methods),
                        "used_hyde": used_hyde,
                    },
                )
            )
            if len(selected) >= self.config.retrieval_k:
                break
        return selected

    def _retrieval_query(self, question: str) -> Tuple[str, bool]:
        if not self.config.enable_hyde or self.llm is None or not is_vague_query(question):
            return question, False

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Write a short hypothetical passage that could appear in an internal HR policy "
                        "and would help retrieve the answer to the employee question. Use likely policy "
                        "terminology, but do not invent numbers, dates, benefit amounts, or company-specific facts."
                    ),
                ),
                ("human", "{question}"),
            ]
        )
        try:
            rewritten = (prompt | self.llm | StrOutputParser()).invoke({"question": question}).strip()
            return (rewritten or question), bool(rewritten)
        except Exception:
            return question, False

    def format_context(self, docs: Sequence[Document]) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
            chunk_id = doc.metadata.get("chunk_id", "n/a")
            retrieval_rank = doc.metadata.get("retrieval_rank", "n/a")
            text = clean_text(doc.page_content)[: self.config.max_context_chars_per_chunk]
            parts.append(
                "Relevance rank: %s\nCitation: [%s from %s]\nSource file: %s\nChunk ID: %s\nPolicy text:\n%s"
                % (retrieval_rank, chunk_id, source, source, chunk_id, text)
            )
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
        style_instruction = answer_style_instruction(question)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an HR policy assistant for Zyro Dynamics. Answer only from the retrieved "
                        "policy excerpts supplied in the user message.\n"
                        "Rules:\n"
                        "- Copy key policy phrases, numbers, dates, percentages, amounts, and named terms "
                        "verbatim from the excerpts. Do not paraphrase them.\n"
                        "- Answer every part of the question, but include no unasked background or general HR knowledge.\n"
                        "- Use two to four concise sentences maximum, except when a complete requested timeline "
                        "requires a numbered list.\n"
                        "- If a question asks about a policy limit or carry-forward rule, include the directly stated "
                        "consequence of exceeding that limit when it appears in the excerpts.\n"
                        "- Follow the answer style instruction exactly.\n"
                        "- The challenge corpus uses Acrux Dynamics and Zyro Dynamics interchangeably. Treat "
                        "those two names as the same company only for answers grounded in this context.\n"
                        "- Company-wide policy facts such as salary bands, grade ranges, and benefit tables are "
                        "allowed. Do not reveal or infer any specific employee's private compensation or records.\n"
                        "- Return only the final answer in plain text. Do not use markdown, label prefixes, citations, "
                        "source lines, chunk IDs, reasoning, relevance ranks, or repeated facts.\n"
                        "- If the context does not contain the answer, return exactly: "
                        "I could not find this information in the Zyro Dynamics HR policy documents."
                    ),
                ),
                (
                    "human",
                    (
                        "Conversation history:\n{history}\n\n"
                        "Policy context:\n{context}\n\n"
                        "Employee question: {question}\n\n"
                        "Answer style instruction: {style_instruction}\n\n"
                        "Answer:"
                    ),
                ),
            ]
        )

        if create_stuff_documents_chain is not None:
            document_prompt = PromptTemplate.from_template(
                "Relevance rank: {retrieval_rank}\n"
                "Citation: [{chunk_id} from {source_file}]\n"
                "Policy text:\n{page_content}"
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
                    "style_instruction": style_instruction,
                }
            )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "history": history_text or "None",
                "context": context,
                "question": question,
                "style_instruction": style_instruction,
            }
        )

    def _extractive_answer(self, question: str, docs: Sequence[Document]) -> str:
        sentences = []
        query_terms = set(tokenize(question)) - QUERY_STOPWORDS
        for doc_rank, doc in enumerate(docs):
            for sentence_rank, sentence in enumerate(split_sentences(doc.page_content)):
                sentence_terms = set(tokenize(sentence))
                overlap = len(query_terms & sentence_terms)
                if overlap:
                    coverage = overlap / max(1, len(query_terms))
                    exact_number_bonus = 0.35 if any(token[0].isdigit() for token in sentence_terms) else 0.0
                    score = coverage * 4.0 + overlap + exact_number_bonus - (doc_rank * 0.05)
                    sentences.append((score, doc_rank, sentence_rank, clean_text(sentence)))
        sentences.sort(key=lambda item: item[0], reverse=True)
        selected: List[str] = []
        seen = set()
        normalized_question = clean_text(question).lower()
        max_sentences = 2 if (
            " and " in normalized_question
            or normalized_question.count("?") > 1
            or "maximum" in normalized_question
            or "carry-forward" in normalized_question
            or "carry forward" in normalized_question
        ) else 1
        for _score, _doc_rank, _sentence_rank, sentence in sentences:
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(sentence)
            if len(selected) >= max_sentences:
                break
        if not selected:
            return "I could not find this information in the Zyro Dynamics HR policy documents."
        return " ".join(selected)

    def _self_critique(self, question: str, context: str, draft: str) -> Tuple[str, Optional[str]]:
        style_instruction = answer_style_instruction(question)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a strict HR RAG answer reviewer. Check the draft only against the supplied "
                        "policy context. Preserve exact policy wording and answer only what the question asks. "
                        "The corpus uses Acrux Dynamics and Zyro Dynamics interchangeably. Company-wide policy "
                        "facts such as salary bands and benefit tables are allowed; private individual records are not. "
                        "Do not add unsupported facts. Return exactly two sections:\n"
                        "RATING: COMPLETE, PARTIAL, or MISSING\n"
                        "REFINED ANSWER: Return an improved answer following these rules:\n"
                        "- Ensure every requested number, date, percentage, amount, deadline, and condition present "
                        "in the context is included verbatim.\n"
                        "- Remove every statement that is not directly supported by the context or requested by the question.\n"
                        "- Keep two to four concise sentences maximum, except for a requested timeline.\n"
                        "- Use plain text with no markdown, labels, source lines, citations, chunk IDs, or commentary.\n"
                        "- If support is missing, use exactly: "
                        "I could not find this information in the Zyro Dynamics HR policy documents.\n"
                        "Return only those two sections."
                    ),
                ),
                (
                    "human",
                    (
                        "Question: {question}\n\n"
                        "Answer style instruction: {style_instruction}\n\n"
                        "Policy context:\n{context}\n\n"
                        "Draft answer:\n{draft}"
                    ),
                ),
            ]
        )
        try:
            output = (prompt | self.llm | StrOutputParser()).invoke(
                {
                    "question": question,
                    "style_instruction": style_instruction,
                    "context": context,
                    "draft": draft,
                }
            )
            rating = parse_section(output, "RATING:")
            refined = parse_section(output, "REFINED ANSWER:")
            return (refined or draft), (rating or None)
        except Exception:
            return draft, None

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

        if any(pattern.search(q) for pattern in EXTERNAL_ORGANIZATION_PATTERNS):
            return False, "I can only answer HR-related questions from Zyro Dynamics policy documents."

        if any(pattern.search(q) for pattern in LEGAL_ADVICE_PATTERNS):
            return False, "I can only answer HR-related questions from Zyro Dynamics policy documents."

        if any(term in q_lower for term in OBVIOUS_OUT_OF_SCOPE):
            return False, "I can only answer HR-related questions from Zyro Dynamics policy documents."

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
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
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
    if selected in {"huggingface", "sentence-transformers"}:
        model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        model_kwargs = {"device": os.getenv("HUGGINGFACE_EMBEDDING_DEVICE", "cpu")}
        encode_kwargs = {"normalize_embeddings": True}
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    if selected == "hash":
        return LocalHashEmbeddings(dim=int(os.getenv("HASH_EMBEDDING_DIM", "768")))
    raise ValueError("Unsupported embedding provider: %s" % provider)


def build_vectorstore(
    chunks: Sequence[Document],
    embeddings: Embeddings,
    db_path: Path,
    collection_name: str,
):
    if isinstance(embeddings, LocalHashEmbeddings):
        return InMemoryVectorStore.from_documents(chunks, embeddings)
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
    return build_memory_vectorstore_with_fallback(chunks, embeddings)


def load_vectorstore_or_memory(
    chunks: Sequence[Document],
    embeddings: Embeddings,
    db_path: Path,
    collection_name: str,
):
    if isinstance(embeddings, LocalHashEmbeddings):
        return InMemoryVectorStore.from_documents(chunks, embeddings)
    if Chroma is not None and db_path.exists():
        try:
            return Chroma(
                persist_directory=str(db_path),
                embedding_function=embeddings,
                collection_name=collection_name,
            )
        except Exception:
            pass
    return build_memory_vectorstore_with_fallback(chunks, embeddings)


def build_memory_vectorstore_with_fallback(
    chunks: Sequence[Document],
    embeddings: Embeddings,
) -> InMemoryVectorStore:
    try:
        return InMemoryVectorStore.from_documents(chunks, embeddings)
    except Exception:
        hash_embeddings = LocalHashEmbeddings(dim=int(os.getenv("HASH_EMBEDDING_DIM", "768")))
        return InMemoryVectorStore.from_documents(chunks, hash_embeddings)


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
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
            model_kwargs={"top_p": float(os.getenv("GROQ_TOP_P", "0.9"))},
            timeout=float(os.getenv("GROQ_REQUEST_TIMEOUT", "90")),
            max_retries=int(os.getenv("GROQ_MAX_RETRIES", "0")),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "500")),
        )
    if selected == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "500")),
            timeout=float(os.getenv("OPENAI_REQUEST_TIMEOUT", "90")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "0")),
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
                "confidence": "%.2f" % float(doc.metadata.get("retrieval_confidence", 0.0)),
                "retrieval_rank": str(doc.metadata.get("retrieval_rank", "")),
                "retrieval_methods": str(doc.metadata.get("retrieval_methods", "")),
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
    without_controls = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text or "")
    return re.sub(r"\s+", " ", without_controls.strip())


def normalize_company_aliases(text: str) -> str:
    """Normalize the competition's legacy company alias to the policy corpus name."""
    return re.sub(r"\bAcrux Dynamics\b", "Zyro Dynamics", text or "", flags=re.I)


def remove_company_aliases(text: str) -> str:
    """Remove company-name boilerplate from retrieval queries."""
    without_aliases = re.sub(r"\b(?:Acrux|Zyro)\s+Dynamics\b", " ", text or "", flags=re.I)
    return clean_text(without_aliases)


def infer_policy_source_hints(question: str) -> set[str]:
    normalized = clean_text(question).lower()
    return {
        source_file
        for terms, source_file in POLICY_SOURCE_ROUTES
        if any(term in normalized for term in terms)
    }


def query_doc_overlap(question: str, doc: Document) -> int:
    query_terms = set(tokenize(question)) - QUERY_STOPWORDS
    return len(query_terms & set(tokenize(doc.page_content)))


def needs_adjacent_context(question: str) -> bool:
    normalized = clean_text(question).lower()
    return " and " in normalized or bool(
        re.search(r"\b(timeline|schedule|stages?|steps?|process|procedure)\b", normalized)
    )


def expand_with_adjacent_policy_chunks(
    ranked: Sequence[Tuple[Document, float, float, List[str]]],
    all_chunks: Sequence[Document],
    source_hints: set[str],
) -> List[Tuple[Document, float, float, List[str]]]:
    if not ranked:
        return []

    primary = next(
        (
            item
            for item in ranked
            if str(item[0].metadata.get("source_file", item[0].metadata.get("source", ""))) in source_hints
        ),
        None,
    )
    if primary is None:
        return list(ranked)

    primary_doc, score, confidence, _methods = primary
    source = str(primary_doc.metadata.get("source_file", primary_doc.metadata.get("source", "")))
    try:
        chunk_id = int(primary_doc.metadata.get("chunk_id"))
    except (TypeError, ValueError):
        return list(ranked)

    chunk_lookup = {
        (
            str(doc.metadata.get("source_file", doc.metadata.get("source", ""))),
            int(doc.metadata.get("chunk_id")),
        ): doc
        for doc in all_chunks
        if str(doc.metadata.get("chunk_id", "")).isdigit()
    }
    adjacent = [
        chunk_lookup[(source, neighbor_id)]
        for neighbor_id in (chunk_id - 1, chunk_id + 1)
        if (source, neighbor_id) in chunk_lookup
    ]
    adjacent_keys = {doc_key(doc) for doc in adjacent}
    remainder = [item for item in ranked if doc_key(item[0]) not in adjacent_keys and item is not primary]
    expanded = [primary]
    expanded.extend((doc, score * 0.99, confidence, ["adjacent_context"]) for doc in adjacent)
    expanded.extend(remainder)
    return expanded


def split_sentences(text: str) -> Iterable[str]:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        sentence = clean_text(sentence)
        if len(sentence) >= 30:
            yield sentence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    denom = (math.sqrt(sum(v * v for v in left)) or 1.0) * (math.sqrt(sum(v * v for v in right)) or 1.0)
    return sum(a * b for a, b in zip(left, right)) / denom


def weighted_reciprocal_rank_fusion(
    ranked_lists: Sequence[Tuple[str, Sequence[Document], float]],
    rrf_k: int = 60,
) -> List[Tuple[Document, float, float, List[str]]]:
    candidates: Dict[str, Dict[str, object]] = {}
    active_weight = sum(weight for _name, docs, weight in ranked_lists if docs)
    ideal_score = active_weight / max(1, rrf_k + 1)

    for method, docs, weight in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc_key(doc)
            entry = candidates.setdefault(key, {"doc": doc, "score": 0.0, "methods": []})
            entry["score"] = float(entry["score"]) + weight / (rrf_k + rank)
            methods = entry["methods"]
            if isinstance(methods, list) and method not in methods:
                methods.append(method)

    ranked = sorted(candidates.values(), key=lambda entry: float(entry["score"]), reverse=True)
    return [
        (
            entry["doc"],
            float(entry["score"]),
            min(1.0, float(entry["score"]) / ideal_score) if ideal_score else 0.0,
            list(entry["methods"]),
        )
        for entry in ranked
    ]


def is_vague_query(question: str) -> bool:
    normalized = clean_text(question).lower()
    tokens = tokenize(normalized)
    vague_phrases = {
        "how do i start",
        "what do i do",
        "how does it work",
        "tell me about it",
        "help me with this",
        "what about this",
        "can i do this",
        "how can i do this",
        "what is the policy",
    }
    if any(phrase in normalized for phrase in vague_phrases):
        return True

    specific_terms = HR_KEYWORDS - {"hr", "employee", "employment", "policy", "manager", "department"}
    has_specific_term = any(term in normalized for term in specific_terms)
    return len(tokens) <= 5 and not has_specific_term


def average_confidence(docs: Sequence[Document]) -> float:
    if not docs:
        return 0.0
    values = [float(doc.metadata.get("retrieval_confidence", 0.0)) for doc in docs]
    return round(sum(values) / len(values), 4)


def append_citation_block(answer: str, sources: Sequence[Dict[str, str]]) -> str:
    if not sources or "\n\nSources:" in answer or answer.strip().startswith("Sources:"):
        return answer
    citations = [
        "[%s from %s]" % (source["chunk_id"], source["source_file"])
        for source in sources
    ]
    return answer.rstrip() + "\n\nSources: " + "; ".join(citations)


def parse_section(text: str, label: str) -> str:
    if label not in text:
        return ""
    content = text.split(label, 1)[1]
    if label == "RATING:" and "REFINED ANSWER:" in content:
        content = content.split("REFINED ANSWER:", 1)[0]
    return content.strip()
