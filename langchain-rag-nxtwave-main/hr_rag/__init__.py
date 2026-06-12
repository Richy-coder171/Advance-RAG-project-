"""Reusable HR Help Desk RAG pipeline."""

from .official_corpus import OFFICIAL_CORPUS_SHA256, validate_official_corpus
from .pipeline import HRRagConfig, HRRagPipeline, HRRagResponse, REFUSAL_TEXT

__all__ = [
    "HRRagConfig",
    "HRRagPipeline",
    "HRRagResponse",
    "REFUSAL_TEXT",
    "OFFICIAL_CORPUS_SHA256",
    "validate_official_corpus",
]
