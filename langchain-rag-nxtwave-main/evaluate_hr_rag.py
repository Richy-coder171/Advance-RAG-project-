from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hr_rag import HRRagConfig, HRRagPipeline


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Zyro HR RAG pipeline on a validation set.")
    parser.add_argument("--docs-path", default="hr_docs/official", help="Folder containing the official HR policy PDFs.")
    parser.add_argument("--validation-file", default="eval/hr_validation_sample.jsonl", help="JSONL or CSV validation file.")
    parser.add_argument("--output-dir", default="eval/results", help="Directory for evaluation outputs.")
    parser.add_argument("--db-path", default="chroma_hr_eval_store", help="Vector DB folder for evaluation.")
    parser.add_argument(
        "--embedding-provider",
        default="hash",
        choices=["auto", "openai", "ollama", "huggingface", "hash"],
    )
    parser.add_argument("--llm-provider", default="extractive", choices=["auto", "groq", "openai", "ollama", "extractive"])
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--retrieval-k", type=int, default=8)
    parser.add_argument("--fetch-k", type=int, default=60)
    parser.add_argument("--vector-weight", type=float, default=0.65)
    parser.add_argument("--keyword-weight", type=float, default=None, help="BM25 weight. Defaults to 1 - vector_weight.")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--max-chunks-per-source", type=int, default=2)
    parser.add_argument("--critique-threshold", type=float, default=0.55)
    parser.add_argument("--disable-hyde", action="store_true")
    critique_group = parser.add_mutually_exclusive_group()
    critique_group.add_argument("--disable-self-critique", action="store_true")
    critique_group.add_argument("--force-self-critique", action="store_true")
    parser.add_argument("--no-source-block", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def load_validation_rows(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("Validation file not found: %s" % path)

    if file_path.suffix.lower() == ".jsonl":
        rows = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row.setdefault("id", line_no)
                rows.append(normalize_row(row))
        return rows

    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [normalize_row(row) for row in csv.DictReader(handle)]

    raise ValueError("Validation file must be .jsonl or .csv")


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    expected_sources = row.get("expected_sources", [])
    if isinstance(expected_sources, str):
        expected_sources = expected_sources.strip()
        if not expected_sources:
            expected_sources = []
        elif expected_sources.startswith("["):
            expected_sources = json.loads(expected_sources)
        else:
            expected_sources = [item.strip() for item in expected_sources.split(";") if item.strip()]

    should_block = row.get("should_block", False)
    if isinstance(should_block, str):
        should_block = should_block.lower() in {"1", "true", "yes", "y"}

    return {
        "id": row.get("id", ""),
        "question": row.get("question", row.get("query", "")),
        "reference_answer": row.get("reference_answer", row.get("answer", "")),
        "expected_sources": expected_sources,
        "should_block": should_block,
        "category": row.get("category", ""),
    }


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: Dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = longest_common_subsequence(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def longest_common_subsequence(left: List[str], right: List[str]) -> int:
    prev = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(prev[idx - 1] + 1)
            else:
                current.append(max(prev[idx], current[-1]))
        prev = current
    return prev[-1]


def source_recall(retrieved_sources: Iterable[str], expected_sources: List[str]) -> Optional[float]:
    if not expected_sources:
        return None
    retrieved = {normalize_source_name(source) for source in retrieved_sources}
    expected = {normalize_source_name(source) for source in expected_sources}
    return len(retrieved & expected) / len(expected)


def normalize_source_name(source: str) -> str:
    return Path(str(source)).name.lower()


def strip_sources(text: str) -> str:
    """Remove UI/source citations before answer-level metric calculation."""
    answer = re.split(r"(?:^|\n)\s*Sources\s*:", text or "", maxsplit=1, flags=re.I)[0]
    answer = re.sub(r"\s*\[\s*\d+\s+from\s+[^\]]+\]\s*", " ", answer)
    answer = re.sub(r"\s*\[\s*[^\]]+\s+chunk\s+\d+\s*\]\s*", " ", answer)
    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
    return re.sub(r"\s+", " ", answer).strip()


def clean_answer_for_scoring(answer: str) -> str:
    """Backward-compatible alias for older local notebooks."""
    return strip_sources(answer)


def evaluate_rows(pipeline: HRRagPipeline, rows: List[Dict[str, Any]], force_refine: bool) -> List[Dict[str, Any]]:
    results = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        error = ""
        answer = ""
        clean_answer = ""
        retrieved_sources: List[str] = []
        recall = None
        confidence = 0.0
        sources: List[Dict[str, str]] = []
        blocked = False
        used_hyde = False
        refined = False
        critique_rating = None
        blocked_correct = None
        reference = row["reference_answer"]
        try:
            response = pipeline.answer(row["question"], force_refine=force_refine)
            answer = response.answer
            clean_answer = strip_sources(answer)
            retrieved_sources = [source["source_file"] for source in response.sources]
            recall = source_recall(retrieved_sources, row["expected_sources"])
            confidence = response.avg_confidence
            sources = response.sources
            blocked = response.blocked
            used_hyde = response.used_hyde
            refined = response.refined
            critique_rating = response.critique_rating
            if row["should_block"]:
                blocked_correct = bool(response.blocked)
        except Exception as exc:
            error = "%s: %s" % (exc.__class__.__name__, exc)
            clean_answer = ""
            if row["should_block"]:
                blocked_correct = False

        result = {
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "blocked": blocked,
            "blocked_correct": blocked_correct,
            "expected_sources": row["expected_sources"],
            "retrieved_sources": retrieved_sources,
            "source_recall": recall,
            "confidence": confidence,
            "avg_confidence": confidence,
            "used_hyde": used_hyde,
            "refined": refined,
            "critique_rating": critique_rating,
            "token_f1": token_f1(clean_answer, reference) if reference else None,
            "rouge_l": rouge_l(clean_answer, reference) if reference else None,
            "clean_answer": clean_answer,
            "generated_answer": clean_answer,
            "answer_with_sources": answer,
            "answer": answer,
            "reference_answer": reference,
            "sources": sources,
            "error": error,
        }
        results.append(result)
        print("[%s/%s] %s confidence=%.2f%s" % (idx, total, row["id"], confidence, " ERROR" if error else ""))
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    in_scope = [result for result in results if not result["blocked"]]
    return {
        "num_questions": len(results),
        "avg_source_recall": average([r["source_recall"] for r in results if r["source_recall"] is not None]),
        "avg_token_f1": average([r["token_f1"] for r in results if r["token_f1"] is not None]),
        "avg_rouge_l": average([r["rouge_l"] for r in results if r["rouge_l"] is not None]),
        "avg_in_scope_token_f1": average([r["token_f1"] for r in in_scope if r["token_f1"] is not None]),
        "avg_in_scope_rouge_l": average([r["rouge_l"] for r in in_scope if r["rouge_l"] is not None]),
        "avg_confidence": average([r["confidence"] for r in results]),
        "hyde_rate": average([1.0 if r["used_hyde"] else 0.0 for r in results]),
        "refinement_rate": average([1.0 if r["refined"] else 0.0 for r in results]),
        "block_accuracy": average(
            [1.0 if r["blocked_correct"] else 0.0 for r in results if r["blocked_correct"] is not None]
        ),
        "unexpected_guardrail_blocks": sum(
            1 for result in results if result["blocked"] and not result["blocked_correct"]
        ),
        "num_errors": sum(1 for r in results if r["error"]),
    }


def average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def write_outputs(results: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    (path / "details.json").write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")

    csv_fields = [
        "question",
        "expected_source",
        "retrieved_sources",
        "source_recall",
        "confidence",
        "clean_answer",
        "answer_with_sources",
        "reference_answer",
        "token_f1",
        "rouge_l",
        "blocked",
        "error",
    ]
    with (path / "details.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field) for field in csv_fields}
            row["expected_source"] = ";".join(result["expected_sources"])
            row["retrieved_sources"] = ";".join(result["retrieved_sources"])
            writer.writerow(row)

    chunk_fields = [
        "question",
        "expected_source",
        "source_file",
        "chunk_id",
        "retrieval_rank",
        "confidence",
        "retrieval_methods",
        "preview",
    ]
    with (path / "retrieved_chunks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=chunk_fields)
        writer.writeheader()
        for result in results:
            for source in result.get("sources", []):
                writer.writerow(
                    {
                        "question": result["question"],
                        "expected_source": ";".join(result["expected_sources"]),
                        "source_file": source.get("source_file", ""),
                        "chunk_id": source.get("chunk_id", ""),
                        "retrieval_rank": source.get("retrieval_rank", ""),
                        "confidence": source.get("confidence", ""),
                        "retrieval_methods": source.get("retrieval_methods", ""),
                        "preview": source.get("preview", ""),
                    }
                )


def print_diagnostics(results: List[Dict[str, Any]], threshold: float = 0.75) -> None:
    bleeding = [
        result["id"]
        for result in results
        if result["token_f1"] is not None and result["token_f1"] < threshold
    ]
    unexpected_blocks = [
        result["id"]
        for result in results
        if result["blocked"] and not result["blocked_correct"]
    ]
    expected_blocks = [result for result in results if result["blocked_correct"] is not None]
    correctly_blocked = [result["id"] for result in expected_blocks if result["blocked_correct"]]
    print("Diagnostics:")
    print("- token_f1 < %.2f: %s" % (threshold, ", ".join(map(str, bleeding)) if bleeding else "NONE"))
    print("- unexpected guardrail blocks: %s" % (", ".join(map(str, unexpected_blocks)) if unexpected_blocks else "NONE"))
    print(
        "- expected blocked questions correct: %s/%s (%s)"
        % (len(correctly_blocked), len(expected_blocks), ", ".join(map(str, correctly_blocked)) if correctly_blocked else "NONE")
    )


def main() -> None:
    args = parse_args()
    rows = load_validation_rows(args.validation_file)
    config = HRRagConfig(
        docs_path=args.docs_path,
        db_path=args.db_path,
        embedding_provider=args.embedding_provider,
        llm_provider=args.llm_provider,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retrieval_k=args.retrieval_k,
        fetch_k=args.fetch_k,
        vector_weight=args.vector_weight,
        keyword_weight=args.keyword_weight if args.keyword_weight is not None else 1.0 - args.vector_weight,
        min_confidence=args.min_confidence,
        max_chunks_per_source=args.max_chunks_per_source,
        enable_hyde=not args.disable_hyde,
        enable_self_critique=not args.disable_self_critique,
        critique_confidence_threshold=args.critique_threshold,
        append_source_block=not args.no_source_block,
    )
    pipeline = HRRagPipeline.from_config(config, rebuild=args.rebuild)
    results = evaluate_rows(pipeline, rows, force_refine=args.force_self_critique)
    summary = summarize(results)
    write_outputs(results, summary, args.output_dir)
    print(json.dumps(summary, indent=2))
    print_diagnostics(results)
    print("Wrote evaluation files to %s" % args.output_dir)


if __name__ == "__main__":
    main()
