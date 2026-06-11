"""Reusable HR Help Desk RAG pipeline."""

from .official_corpus import OFFICIAL_CORPUS_SHA256, validate_official_corpus
from .pipeline import HRRagConfig, HRRagPipeline, HRRagResponse

__all__ = [
    "HRRagConfig",
    "HRRagPipeline",
    "HRRagResponse",
    "OFFICIAL_CORPUS_SHA256",
    "validate_official_corpus",
]
