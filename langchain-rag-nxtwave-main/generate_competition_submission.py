from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import List, Sequence, Tuple

from cryptography.fernet import Fernet

from evaluate_hr_rag import strip_sources
from hr_rag import HRRagConfig, HRRagPipeline, validate_official_corpus


STREAMLIT_PATTERN = re.compile(r"^https://.+\.streamlit\.app(/.*)?$", re.IGNORECASE)
LANGSMITH_PATTERN = re.compile(r"^https://smith\.langchain\.com/.+", re.IGNORECASE)
PLACEHOLDER_LINK_MARKERS = ("your-", "your_", "placeholder", "example", "test-trace", "replace-me")
REFUSAL_MARKERS = (
    "i can only answer",
    "i could not find this information",
    "i cannot answer",
    "not available in the",
)
REQUIRED_COLUMNS = [
    "question_id",
    "question_enc",
    "answer_enc",
    "streamlit_link",
    "langsmith_link",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the official Zyro RAG competition submission.")
    parser.add_argument("--docs-path", default="hr_docs/official", help="Folder containing the official 11 PDFs.")
    parser.add_argument("--starter-notebook", default="competition/Starter_Notebook.ipynb")
    parser.add_argument("--output", default="submissions/submission.csv")
    parser.add_argument("--streamlit-link", required=True)
    parser.add_argument("--langsmith-link", required=True, help="Publicly shared LangSmith project or trace URL.")
    parser.add_argument("--db-path", default="chroma_zyro_official_store")
    parser.add_argument("--embedding-provider", default="auto", choices=["auto", "openai", "ollama", "hash"])
    parser.add_argument("--llm-provider", default="auto", choices=["auto", "groq", "openai", "ollama", "extractive"])
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between questions to reduce rate-limit risk.")
    parser.add_argument("--disable-self-critique", action="store_true")
    parser.add_argument("--disable-tracing", action="store_true", help="Disable LangSmith only for local smoke tests.")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def extract_competition_questions(starter_notebook: str) -> Tuple[Fernet, List[Tuple[str, str]]]:
    notebook_path = Path(starter_notebook)
    if not notebook_path.exists():
        raise FileNotFoundError("Starter notebook not found: %s" % notebook_path)

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    secret = None
    encrypted_questions: Sequence[Tuple[str, str]] | None = None

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        secret_match = re.search(r"SUBMISSION_SECRET\s*=\s*b[\"']([^\"']+)", source)
        if secret_match:
            secret = secret_match.group(1).encode("ascii")

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "_Q" for target in node.targets):
                encrypted_questions = ast.literal_eval(node.value)

    if secret is None or encrypted_questions is None:
        raise ValueError("Could not find SUBMISSION_SECRET and _Q in the official starter notebook.")

    fernet = Fernet(secret)
    questions = [
        (question_id, fernet.decrypt(encrypted.encode("ascii")).decode("utf-8"))
        for question_id, encrypted in encrypted_questions
    ]
    expected_ids = ["Q%02d" % index for index in range(1, 16)]
    if [question_id for question_id, _question in questions] != expected_ids:
        raise ValueError("Expected official questions Q01-Q15 in order.")
    return fernet, questions


def validate_links(streamlit_link: str, langsmith_link: str) -> None:
    errors = []
    normalized_streamlit = streamlit_link.strip().lower()
    normalized_langsmith = langsmith_link.strip().lower()
    if not STREAMLIT_PATTERN.match(streamlit_link.strip()):
        errors.append("Invalid Streamlit URL. Expected https://<app>.streamlit.app")
    if not LANGSMITH_PATTERN.match(langsmith_link.strip()):
        errors.append("Invalid LangSmith URL. Expected https://smith.langchain.com/...")
    if any(marker in normalized_streamlit for marker in PLACEHOLDER_LINK_MARKERS):
        errors.append("Streamlit URL still contains a placeholder value.")
    if any(marker in normalized_langsmith for marker in PLACEHOLDER_LINK_MARKERS):
        errors.append("LangSmith URL still contains a placeholder value.")
    if errors:
        raise ValueError("\n".join(errors))


def is_refusal(answer: str) -> bool:
    normalized = answer.strip().lower()
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def validate_submission(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 15:
        raise ValueError("Submission must contain exactly 15 rows; found %s." % len(rows))
    if list(rows[0].keys()) != REQUIRED_COLUMNS:
        raise ValueError("Submission columns must be exactly: %s" % ", ".join(REQUIRED_COLUMNS))
    for index, row in enumerate(rows, start=1):
        if row["question_id"] != "Q%02d" % index:
            raise ValueError("Submission question IDs must be Q01-Q15 in order.")
        if not all(row.get(column, "").strip() for column in REQUIRED_COLUMNS):
            raise ValueError("Submission contains an empty required field in row %s." % index)
        validate_links(row["streamlit_link"], row["langsmith_link"])


def main() -> None:
    args = parse_args()
    validate_links(args.streamlit_link, args.langsmith_link)
    validate_official_corpus(args.docs_path)
    fernet, questions = extract_competition_questions(args.starter_notebook)

    if args.disable_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    else:
        os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
        os.environ["LANGSMITH_PROJECT"] = "zyro-rag-challenge"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"

    config = HRRagConfig(
        docs_path=args.docs_path,
        db_path=args.db_path,
        embedding_provider=args.embedding_provider,
        llm_provider=args.llm_provider,
        enable_self_critique=not args.disable_self_critique,
        append_source_block=True,
    )
    pipeline = HRRagPipeline.from_config(config, rebuild=args.rebuild)
    if pipeline.llm is None and args.llm_provider != "extractive":
        raise ValueError(
            "No answer LLM is configured. Add GROQ_API_KEY or explicitly use --llm-provider extractive for smoke tests."
        )

    rows = []
    debug_rows = []
    for index, (question_id, question) in enumerate(questions, start=1):
        response = pipeline.answer(question, force_refine=not args.disable_self_critique)
        clean_answer = strip_sources(response.answer)
        if not clean_answer:
            raise ValueError("%s produced an empty answer." % question_id)
        if index <= 10 and (response.blocked or is_refusal(clean_answer)):
            raise ValueError(
                "%s is in scope but produced a refusal. Review retrieval and generation. Answer: %s"
                % (question_id, clean_answer)
            )
        if index >= 11 and not response.blocked:
            raise ValueError("%s is out of scope but was not blocked by the guardrail." % question_id)

        rows.append(
            {
                "question_id": question_id,
                "question_enc": fernet.encrypt(question.encode("utf-8")).decode("ascii"),
                "answer_enc": fernet.encrypt(clean_answer.encode("utf-8")).decode("ascii"),
                "streamlit_link": args.streamlit_link.strip(),
                "langsmith_link": args.langsmith_link.strip(),
            }
        )
        debug_rows.append(
            {
                "question_id": question_id,
                "question": question,
                "clean_answer": clean_answer,
                "answer_with_sources": response.answer,
                "blocked": response.blocked,
                "confidence": response.avg_confidence,
                "sources": response.sources,
            }
        )
        print("[%02d/15] %s answered%s" % (index, question_id, " (blocked)" if response.blocked else ""))
        if index < len(questions) and args.delay > 0:
            time.sleep(args.delay)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    validate_submission(output_path)
    debug_path = output_path.with_suffix(".sources.json")
    debug_path.write_text(json.dumps(debug_rows, ensure_ascii=True, indent=2), encoding="utf-8")
    print("Validated official 15-row submission: %s" % output_path)
    print("Wrote answer/source debug log: %s" % debug_path)


if __name__ == "__main__":
    main()
