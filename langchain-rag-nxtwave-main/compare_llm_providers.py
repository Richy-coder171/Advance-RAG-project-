from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluate_hr_rag import evaluate_rows, load_validation_rows, summarize, write_outputs
from hr_rag import HRRagConfig, HRRagPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple LLM providers on the HR validation set.")
    parser.add_argument("--docs-path", default="hr_docs/official")
    parser.add_argument("--validation-file", default="eval/hr_validation_sample.jsonl")
    parser.add_argument("--output-dir", default="eval/provider_comparison")
    parser.add_argument("--providers", default="groq,openai,anthropic,google", help="Comma-separated llm providers to evaluate.")
    parser.add_argument("--embedding-provider", default="huggingface", choices=["auto", "openai", "ollama", "huggingface", "hash"])
    parser.add_argument("--chunking-strategy", default="semantic", choices=["recursive", "semantic"])
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--retrieval-k", type=int, default=8)
    parser.add_argument("--fetch-k", type=int, default=60)
    parser.add_argument("--vector-weight", type=float, default=0.65)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--max-chunks-per-source", type=int, default=2)
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-top-n", type=int, default=12)
    parser.add_argument("--critique-threshold", type=float, default=0.55)
    parser.add_argument("--disable-hyde", action="store_true")
    parser.add_argument("--disable-self-critique", action="store_true")
    parser.add_argument("--disable-few-shot", action="store_true")
    parser.add_argument("--no-source-block", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def metric_value(value: Any) -> float:
    return -1.0 if value is None else float(value)


def leaderboard_key(row: Dict[str, Any]) -> tuple:
    return (
        metric_value(row.get("avg_rouge_l")),
        metric_value(row.get("avg_token_f1")),
        metric_value(row.get("avg_source_recall")),
        metric_value(row.get("block_accuracy")),
    )


def write_leaderboard(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=leaderboard_key, reverse=True)
    (output_dir / "leaderboard.json").write_text(json.dumps(sorted_rows, ensure_ascii=True, indent=2), encoding="utf-8")
    if not sorted_rows:
        return
    with (output_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sorted_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = load_validation_rows(args.validation_file)
    providers = [provider.strip().lower() for provider in args.providers.split(",") if provider.strip()]
    leaderboard: List[Dict[str, Any]] = []

    for provider in providers:
        run_id = f"{provider}_{args.embedding_provider}_{args.chunking_strategy}"
        print(f"\nEvaluating provider: {provider}", flush=True)
        try:
            config = HRRagConfig(
                docs_path=args.docs_path,
                db_path=str(output_dir / "stores" / run_id),
                embedding_provider=args.embedding_provider,
                llm_provider=provider,
                chunking_strategy=args.chunking_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                retrieval_k=args.retrieval_k,
                fetch_k=args.fetch_k,
                vector_weight=args.vector_weight,
                keyword_weight=1.0 - args.vector_weight,
                min_confidence=args.min_confidence,
                max_chunks_per_source=args.max_chunks_per_source,
                reranker_model=args.reranker_model,
                reranker_top_n=args.reranker_top_n,
                enable_hyde=not args.disable_hyde,
                enable_self_critique=not args.disable_self_critique,
                critique_confidence_threshold=args.critique_threshold,
                use_few_shot_examples=not args.disable_few_shot,
                append_source_block=not args.no_source_block,
            )
            pipeline = HRRagPipeline.from_config(config, rebuild=args.rebuild)
            results = evaluate_rows(pipeline, rows, force_refine=False)
            summary = summarize(results)
            write_outputs(results, summary, str(output_dir / provider))
            leaderboard.append({"provider": provider, **summary, "error": ""})
        except Exception as exc:
            leaderboard.append(
                {
                    "provider": provider,
                    "avg_source_recall": None,
                    "avg_token_f1": None,
                    "avg_rouge_l": None,
                    "avg_confidence": None,
                    "block_accuracy": None,
                    "num_errors": None,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            print(f"ERROR {provider}: {exc}", flush=True)

        write_leaderboard(leaderboard, output_dir)

    print("\nProvider comparison complete.")
    print("Wrote comparison files to %s" % output_dir)


if __name__ == "__main__":
    main()
