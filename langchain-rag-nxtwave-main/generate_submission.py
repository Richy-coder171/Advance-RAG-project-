from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from hr_rag import HRRagConfig, HRRagPipeline


QUESTION_COLUMNS = ["question", "query", "prompt", "employee_question", "Question", "Query"]
ID_COLUMNS = ["id", "ID", "qid", "question_id", "QuestionId"]


def infer_column(columns, preferred, fallback: Optional[str] = None):
    for name in preferred:
        if name in columns:
            return name
    return fallback


def infer_question_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    found = infer_column(df.columns, QUESTION_COLUMNS)
    if found:
        return found
    for column in df.columns:
        if df[column].dtype == object:
            return column
    raise ValueError("Could not infer question column. Pass --question-column.")


def infer_id_column(df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return infer_column(df.columns, ID_COLUMNS)


def build_output_frame(
    questions_df: pd.DataFrame,
    answers,
    id_column: Optional[str],
    answer_column: str,
    sample_submission: Optional[str],
) -> pd.DataFrame:
    if sample_submission:
        output = pd.read_csv(sample_submission)
        if len(output) != len(answers):
            raise ValueError("Sample submission row count does not match questions row count.")
        target_col = answer_column if answer_column in output.columns else output.columns[-1]
        output[target_col] = answers
        return output

    output = pd.DataFrame()
    if id_column:
        output[id_column] = questions_df[id_column]
    else:
        output["id"] = questions_df.index
    output[answer_column] = answers
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission answers with the HR RAG pipeline.")
    parser.add_argument("--docs-path", default="hr_docs", help="Folder containing Zyro HR policy documents.")
    parser.add_argument("--questions", required=True, help="CSV file containing evaluation questions.")
    parser.add_argument("--output", default="submission.csv", help="Output submission CSV path.")
    parser.add_argument("--sample-submission", default=None, help="Optional sample_submission.csv to preserve column order.")
    parser.add_argument("--question-column", default=None, help="Question column name. Inferred if omitted.")
    parser.add_argument("--id-column", default=None, help="ID column name. Inferred if omitted.")
    parser.add_argument("--answer-column", default="answer", help="Submission answer column name.")
    parser.add_argument("--db-path", default="chroma_hr_store", help="Vector DB folder.")
    parser.add_argument("--embedding-provider", default="auto", choices=["auto", "openai", "ollama", "hash"])
    parser.add_argument("--llm-provider", default="auto", choices=["auto", "groq", "openai", "ollama", "extractive"])
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    parser.add_argument("--retrieval-k", type=int, default=6)
    parser.add_argument("--fetch-k", type=int, default=24)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector index before answering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_df = pd.read_csv(args.questions)
    question_column = infer_question_column(questions_df, args.question_column)
    id_column = infer_id_column(questions_df, args.id_column)

    config = HRRagConfig(
        docs_path=args.docs_path,
        db_path=args.db_path,
        embedding_provider=args.embedding_provider,
        llm_provider=args.llm_provider,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retrieval_k=args.retrieval_k,
        fetch_k=args.fetch_k,
    )
    pipeline = HRRagPipeline.from_config(config, rebuild=args.rebuild)

    answers = []
    source_logs = []
    for idx, question in enumerate(questions_df[question_column].fillna("").astype(str), start=1):
        response = pipeline.answer(question)
        answers.append(response.answer)
        source_logs.append({"row": idx, "question": question, "sources": response.sources, "blocked": response.blocked})
        print("[%s/%s] answered" % (idx, len(questions_df)))

    output_df = build_output_frame(
        questions_df=questions_df,
        answers=answers,
        id_column=id_column,
        answer_column=args.answer_column,
        sample_submission=args.sample_submission,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    log_path = output_path.with_suffix(".sources.json")
    log_path.write_text(json.dumps(source_logs, ensure_ascii=True, indent=2), encoding="utf-8")
    print("Wrote %s" % output_path)
    print("Wrote %s" % log_path)


if __name__ == "__main__":
    main()
