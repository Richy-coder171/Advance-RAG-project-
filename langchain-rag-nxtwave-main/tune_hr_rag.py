from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluate_hr_rag import evaluate_rows, load_validation_rows, summarize, write_outputs
from hr_rag import HRRagConfig, HRRagPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune HR RAG chunking and retrieval settings.")
    parser.add_argument("--docs-path", default="hr_docs/official", help="Folder containing the official HR policy PDFs.")
    parser.add_argument("--validation-file", default="eval/hr_validation_sample.jsonl", help="JSONL or CSV validation file.")
    parser.add_argument("--output-dir", default="eval/tuning", help="Directory for tuning outputs.")
    parser.add_argument(
        "--embedding-provider",
        default="hash",
        choices=["auto", "openai", "ollama", "huggingface", "hash"],
    )
    parser.add_argument("--llm-provider", default="extractive", choices=["auto", "groq", "openai", "ollama", "extractive"])
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--max-chunks-per-source", type=int, default=2)
    parser.add_argument("--critique-threshold", type=float, default=0.55)
    parser.add_argument("--disable-hyde", action="store_true")
    critique_group = parser.add_mutually_exclusive_group()
    critique_group.add_argument("--disable-self-critique", action="store_true")
    critique_group.add_argument("--force-self-critique", action="store_true")
    parser.add_argument("--no-source-block", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap for quick smoke tests.")
    parser.add_argument("--keep-all-details", action="store_true", help="Write details for every config, not only best.")
    parser.add_argument("--resume", action="store_true", help="Skip run IDs already present in the output leaderboard.")
    return parser.parse_args()


def config_grid() -> List[Dict[str, Any]]:
    chunk_sizes = [600, 700, 800, 900]
    chunk_overlaps = [150, 200, 250]
    retrieval_ks = [6, 8, 10]
    fetch_ks = [48, 60]
    vector_weights = [0.55, 0.65, 0.70]

    configs = []
    for chunk_size, chunk_overlap, retrieval_k, fetch_k, vector_weight in itertools.product(
        chunk_sizes,
        chunk_overlaps,
        retrieval_ks,
        fetch_ks,
        vector_weights,
    ):
        if fetch_k < retrieval_k:
            continue
        configs.append(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "retrieval_k": retrieval_k,
                "fetch_k": fetch_k,
                "vector_weight": vector_weight,
                "keyword_weight": round(1.0 - vector_weight, 2),
            }
        )
    return configs


def metric_value(value: Optional[float]) -> float:
    return -1.0 if value is None else float(value)


def leaderboard_key(row: Dict[str, Any]) -> tuple:
    return (
        metric_value(row.get("avg_rouge_l")),
        metric_value(row.get("avg_source_recall")),
        metric_value(row.get("avg_token_f1")),
        metric_value(row.get("block_accuracy")),
    )


def print_leaderboard(rows: List[Dict[str, Any]], limit: int = 12) -> None:
    sorted_rows = sorted(rows, key=leaderboard_key, reverse=True)
    fields = [
        "rank",
        "avg_rouge_l",
        "avg_source_recall",
        "avg_token_f1",
        "avg_confidence",
        "block_accuracy",
        "chunk_size",
        "chunk_overlap",
        "retrieval_k",
        "fetch_k",
        "vector_weight",
        "keyword_weight",
    ]
    widths = {field: max(len(field), 10) for field in fields}
    printable = []
    for rank, row in enumerate(sorted_rows[:limit], start=1):
        item = {"rank": rank, **row}
        printable.append(item)
        for field in fields:
            widths[field] = max(widths[field], len(format_cell(item.get(field))))

    header = "  ".join(field.ljust(widths[field]) for field in fields)
    print(header)
    print("-" * len(header))
    for row in printable:
        print("  ".join(format_cell(row.get(field)).ljust(widths[field]) for field in fields))


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.4f" % value
    return str(value)


def write_leaderboard(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=leaderboard_key, reverse=True)
    (output_dir / "leaderboard.json").write_text(json.dumps(sorted_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    if not sorted_rows:
        return
    fields = list(sorted_rows[0].keys())
    with (output_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = load_validation_rows(args.validation_file)
    grid = config_grid()
    if args.max_runs > 0:
        grid = grid[: args.max_runs]

    leaderboard_path = output_dir / "leaderboard.json"
    leaderboard: List[Dict[str, Any]] = []
    if args.resume and leaderboard_path.exists():
        leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
        print("Resuming with %s completed configurations." % len(leaderboard))
    completed_run_ids = {row.get("run_id") for row in leaderboard}
    best_results: Optional[List[Dict[str, Any]]] = None
    best_summary: Optional[Dict[str, Any]] = None
    best_score: Optional[tuple] = max((leaderboard_key(row) for row in leaderboard), default=None)

    for idx, cfg_values in enumerate(grid, start=1):
        run_id = (
            "cs{chunk_size}_co{chunk_overlap}_k{retrieval_k}_f{fetch_k}_vw{vector_weight:.2f}"
        ).format(**cfg_values).replace(".", "")
        if run_id in completed_run_ids:
            continue
        print("\n[%s/%s] %s" % (idx, len(grid), run_id))

        try:
            config = HRRagConfig(
                docs_path=args.docs_path,
                db_path=str(output_dir / "stores" / run_id),
                embedding_provider=args.embedding_provider,
                llm_provider=args.llm_provider,
                chunk_size=cfg_values["chunk_size"],
                chunk_overlap=cfg_values["chunk_overlap"],
                retrieval_k=cfg_values["retrieval_k"],
                fetch_k=cfg_values["fetch_k"],
                vector_weight=cfg_values["vector_weight"],
                keyword_weight=cfg_values["keyword_weight"],
                min_confidence=args.min_confidence,
                max_chunks_per_source=args.max_chunks_per_source,
                enable_hyde=not args.disable_hyde,
                enable_self_critique=not args.disable_self_critique,
                critique_confidence_threshold=args.critique_threshold,
                append_source_block=not args.no_source_block,
            )
            pipeline = HRRagPipeline.from_config(config, rebuild=True)
            results = evaluate_rows(pipeline, rows, force_refine=args.force_self_critique)
            summary = summarize(results)
            leaderboard_row = {"run_id": run_id, **cfg_values, **summary, "error": ""}

            score = leaderboard_key(leaderboard_row)
            if best_score is None or score > best_score:
                best_score = score
                best_results = results
                best_summary = summary

            if args.keep_all_details:
                write_outputs(results, summary, str(output_dir / "details" / run_id))
        except Exception as exc:
            leaderboard_row = {
                "run_id": run_id,
                **cfg_values,
                "avg_source_recall": None,
                "avg_token_f1": None,
                "avg_rouge_l": None,
                "avg_confidence": None,
                "block_accuracy": None,
                "error": "%s: %s" % (exc.__class__.__name__, exc),
            }
            print("ERROR %s" % leaderboard_row["error"])

        leaderboard.append(leaderboard_row)
        write_leaderboard(leaderboard, output_dir)
        print_leaderboard(leaderboard, limit=min(8, len(leaderboard)))

    if best_results is not None and best_summary is not None:
        write_outputs(best_results, best_summary, str(output_dir / "best"))

    print("\nFinal leaderboard sorted by avg_rouge_l, avg_source_recall:")
    print_leaderboard(leaderboard, limit=min(20, len(leaderboard)))
    print("Wrote tuning files to %s" % output_dir)


if __name__ == "__main__":
    main()
