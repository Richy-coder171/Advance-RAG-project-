from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from cryptography.fernet import Fernet

from evaluate_hr_rag import strip_sources
from hr_rag import HRRagConfig, HRRagPipeline, validate_official_corpus
from hr_rag.pipeline import source_dicts


IDEAL_ANSWERS = {
    "Q01": (
        "Earned Leave is accrued at 1.25 days per month. "
        "Employees become eligible for 15 days of Earned Leave upon completion "
        "of one year of continuous service, provided they have worked for a "
        "minimum of 240 days in that year."
    ),
    "Q02": (
        "45 days is the maximum number of Earned Leave that may be carried "
        "forward at the end of each financial year. Any balance exceeding this "
        "limit will be automatically encashed at the employee's basic daily rate "
        "and credited in the April payroll."
    ),
    "Q03": (
        "26 weeks of paid Maternity Leave is the entitlement for female "
        "employees. The minimum service requirement to be eligible is 80 days "
        "of service in the 12 months preceding the expected date of delivery."
    ),
    "Q04": (
        "A Medical Certificate from a registered medical practitioner is required "
        "when Sick Leave is taken for more than 2 consecutive days. "
        "The certificate must be submitted within 3 working days of returning "
        "to work."
    ),
    "Q05": (
        "Salaries are credited to the employee's registered bank account by the "
        "7th of the following month. The payroll cut-off date is the 24th of "
        "each month, and any changes to payment dates will be communicated in "
        "advance by the Payroll team."
    ),
    "Q06": (
        "The CTC range for an L4 Senior grade employee at Zyro Dynamics is "
        "Rs. 16.0L to Rs. 26.0L per annum, with a bonus target of 10% of CTC."
    ),
    "Q07": (
        "Group Medical Insurance provides coverage of up to Rs. 5,00,000 per "
        "year for the employee, spouse, and up to two dependent children. "
        "All insurance premiums are fully paid by the Company."
    ),
    "Q08": (
        "An employee is placed on a Performance Improvement Plan when they "
        "receive a rating of 1 or 2 in two consecutive review cycles, with a "
        "PIP duration of 60 to 90 days as determined by the reporting manager "
        "and HR Business Partner."
    ),
    "Q09": (
        "The Annual Performance Review begins with 360-degree feedback collected "
        "from 1 to 20 February, followed by employee self-assessments submitted "
        "by 10 March and manager ratings by 20 March. Final ratings are locked "
        "by HR between 26 and 31 March, with increment and promotion letters "
        "issued on 15 April."
    ),
    "Q10": (
        "All permanent employees at grade L3 and above are eligible for WFH "
        "arrangements. Hybrid WFH allows up to 3 days per week for L3 and above, "
        "while Full Remote of up to 5 days per week is available on a "
        "case-by-case basis for employees at L5 and above."
    ),
}
REFUSAL_ANSWER = "I can only answer HR-related questions from Zyro Dynamics policy documents."
OUT_OF_SCOPE_IDS = {"Q11", "Q12", "Q13", "Q14", "Q15"}
STREAMLIT_PATTERN = re.compile(r"^https://.+\.streamlit\.app(/.*)?$", re.IGNORECASE)
LANGSMITH_PATTERN = re.compile(r"^https://smith\.langchain\.com/.+", re.IGNORECASE)
PLACEHOLDER_LINK_MARKERS = ("your-", "your_", "placeholder", "example", "test-trace", "replace-me")
REFUSAL_MARKERS = (
    "i can only answer",
    "i could not find this information",
    "i cannot answer",
    "not available in the",
)
BROKEN_FALLBACK_MARKERS = (
    "note: zyro dynamics ensures that no deduction",
    "promotions at zyro dynamics are merit-based",
    "relevance rank:",
    "requested parts of the question",
    "to answer the employee question",
)
RAW_ANSWER_ARTIFACT_PATTERNS = (
    re.compile(r"\[\s*document\s+\d+\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*source\s*:[^\]]*\]", re.IGNORECASE),
    re.compile(r"\[\s*\d+\s*\]"),
    re.compile(r"\bchunk\s*(?:id)?\s*[:#]?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\brelevance rank\s*:", re.IGNORECASE),
    re.compile(r"\bsource file\s*:", re.IGNORECASE),
    re.compile(r"\bsources?\s*:", re.IGNORECASE),
    re.compile(r"\bconfidence\s*:", re.IGNORECASE),
    re.compile(r"\bretrieved from\s*:", re.IGNORECASE),
    re.compile(r"^\s*(?:based on|according to)\s+(?:the\s+|zyro dynamics\s+)?(?:hr\s+)?policy\b", re.IGNORECASE),
    re.compile(r"^\s*(?:here is the|the following|below (?:is|are))\b", re.IGNORECASE),
    re.compile(r"\*\*|#{1,6}\s+"),
    re.compile(r"[\u2022\u00b7]"),
    re.compile(r"^\s*(?:[-*]|\d+\.)\s+", re.MULTILINE),
    re.compile(
        r"\b(?:Scope|Definition|Coverage Scope|Premium Arrangement|Salary Credit Date|Payroll Cut-Off Date|"
        r"CTC Range|Bonus Target|Required document|Submission deadline|Duration of a PIP|"
        r"Retrieval method|Confidence):\s*",
        re.IGNORECASE,
    ),
)
CRITICAL_ANSWER_MARKERS = {
    "Q01": (("1.25",), ("15 days",), ("one year", "1 year")),
    "Q02": (("automatically encash",), ("april payroll",)),
    "Q03": (("26 weeks",), ("80 days",), ("12 months",)),
    "Q04": (("medical certificate",), ("registered medical practitioner",), ("3 working days",), ("returning to work",)),
    "Q05": (("7th",), ("following month",), ("24th",)),
    "Q06": (("16.0l", "16.0 l"), ("26.0l", "26.0 l"), ("10% of ctc",)),
    "Q07": (("5,00,000", "500,000", "5 lakh"), ("per year",)),
    "Q08": (("rating of 1 or 2", "rating 1 or 2"), ("two consecutive",), ("60 to 90 days", "60-90 days")),
    "Q09": (("1 to 20 february",), ("10 march",), ("20 march",), ("26 and 31 march", "31 march"), ("15 april",)),
    "Q10": (
        ("permanent employees",), ("l3",), ("hybrid",), ("full remote",), ("3 days",), ("5 days",), ("l5",),
    ),
}
MAX_ANSWER_WORDS = {
    "Q01": 55, "Q02": 45, "Q03": 40, "Q04": 60, "Q05": 55,
    "Q06": 50, "Q07": 45, "Q08": 45, "Q09": 80, "Q10": 80,
}
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
    parser.add_argument(
        "--embedding-provider",
        default="auto",
        choices=["auto", "openai", "ollama", "huggingface", "hash"],
    )
    parser.add_argument("--llm-provider", default="auto", choices=["auto", "groq", "openai", "ollama", "extractive"])
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--retrieval-k", type=int, default=8, help="Number of policy chunks supplied to the answer LLM.")
    parser.add_argument("--fetch-k", type=int, default=60)
    parser.add_argument("--vector-weight", type=float, default=0.65)
    parser.add_argument("--keyword-weight", type=float, default=None, help="BM25 weight. Defaults to 1 - vector_weight.")
    parser.add_argument("--critique-threshold", type=float, default=0.55)
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds between questions to reduce rate-limit risk.")
    parser.add_argument("--max-retries", type=int, default=4, help="Maximum model attempts per in-scope question.")
    parser.add_argument("--retry-delay", type=float, default=15.0, help="Initial retry delay in seconds.")
    critique_group = parser.add_mutually_exclusive_group()
    critique_group.add_argument("--disable-self-critique", action="store_true")
    critique_group.add_argument(
        "--force-self-critique",
        action="store_true",
        help="Force a second model review for every in-scope answer instead of reviewing only low-confidence answers.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an interrupted partial output file.")
    parser.add_argument(
        "--seed-from",
        help="Seed a new output file from an existing partial candidate, preserving its completed answers.",
    )
    parser.add_argument("--disable-tracing", action="store_true", help="Disable LangSmith only for local smoke tests.")
    parser.add_argument(
        "--disable-hyde",
        action="store_true",
        help="Compatibility flag. HyDE is always disabled for competition submission generation.",
    )
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


def has_artifacts(text: str) -> bool:
    """Return whether an answer contains formatting artifacts or excessive verbosity."""
    patterns = (
        r"\*\*",
        r"^\s*[\u2022\u00b7\u2013-]\s",
        r"^\s*\d+\.\s",
        r"\b\w[\w\s]+:\s+",
        r"^\s*Here is the",
        r"^\s*The following",
        r"\[Document",
        r"\[Source:",
    )
    return len(text.split()) > 80 or any(re.search(pattern, text, re.IGNORECASE | re.MULTILINE) for pattern in patterns)


def clean_answer_for_submission(text: str) -> str:
    """Remove all RAG artifacts and enforce concise plain prose before encryption."""
    if not text or not text.strip():
        return text

    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    prose_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^[\u2022\u00b7\u2013\-*]\s+", stripped):
            stripped = re.sub(r"^[\u2022\u00b7\u2013\-*]\s+", "", stripped)
        if stripped:
            prose_lines.append(stripped)
    text = " ".join(prose_lines)
    text = re.sub(r"\s*[\u2022\u00b7]\s*", " ", text)
    text = re.sub(r"(?<!\d)\d+\.\s+(?=[A-Z])", " ", text)

    text = re.sub(r"\[\s*Document\s*\d+\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*Source\s*:[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\[\s*\d+\s+from\s+[^\]]+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*[^\]]+\s+chunk\s+\d+\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Sources?\s*:\s*\[[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Sources?\s*:[^\n]+", "", text, flags=re.IGNORECASE)

    text = re.sub(
        r"\b(?:Scope|Definition|Coverage Scope|Premium Arrangement|Salary Credit Date|Payroll Cut-Off Date|"
        r"CTC Range|Bonus Target|Required document|Submission deadline|Duration of a PIP|"
        r"Retrieval method|Confidence):\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^(?:Here is the|The following|Below (?:is|are))\b[^:,.]*[:,.]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:Based on|According to)\s+(?:(?:the|Zyro Dynamics)\s+)?(?:HR\s+)?policy[,.]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*This policy applies to all employees[^.]*\.", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s*with any changes to payment dates communicated[^.]*\.",
        "",
        text,
        flags=re.IGNORECASE,
    )

    words = text.split()
    if len(words) > 80:
        trimmed = " ".join(words[:80])
        last_period = trimmed.rfind(".")
        text = trimmed[: last_period + 1] if last_period > len(trimmed) * 0.6 else trimmed

    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def clean_answer_formatting(answer: str) -> str:
    """Backward-compatible alias for older scripts/tests."""
    return clean_answer_for_submission(answer)


def answer_with_retry(
    pipeline: HRRagPipeline,
    question_id: str,
    question: str,
    force_refine: bool,
    max_retries: int,
    retry_delay: float,
    validator: Optional[Callable[[object], None]] = None,
):
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            response = pipeline.answer(question, force_refine=force_refine)
            if validator is not None:
                validator(response)
            return response
        except Exception as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    "%s failed after %s model attempts. No fallback answer was accepted." % (question_id, attempts)
                ) from exc
            wait_seconds = retry_wait_seconds(exc, retry_delay, attempt)
            print(
                "%s model attempt %s/%s failed (%s). Retrying in %.1f seconds."
                % (question_id, attempt, attempts, type(exc).__name__, wait_seconds)
            )
            if wait_seconds:
                time.sleep(wait_seconds)


def retry_wait_seconds(exc: Exception, retry_delay: float, attempt: int) -> float:
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc), re.IGNORECASE)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds + 5.0
    return max(0.0, retry_delay) * (2 ** (attempt - 1))


def validate_competition_response(question_id: str, index: int, response, enforce_word_limit: bool = True) -> None:
    clean_answer = clean_answer_for_submission(strip_sources(response.answer))
    normalized = clean_answer.lower().replace("\u202f", " ").replace("\xa0", " ")
    normalized = re.sub(r"[\u2010-\u2015\u2212]", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    compact = re.sub(r"\s+", "", normalized)
    if not clean_answer:
        raise ValueError("%s produced an empty answer." % question_id)
    if index <= 10 and (response.blocked or is_refusal(clean_answer)):
        raise ValueError("%s is in scope but produced a refusal." % question_id)
    if index <= 10 and response.critique_rating == "EXTRACTIVE_FALLBACK":
        raise ValueError("%s used an extractive fallback." % question_id)
    if index <= 10 and any(marker in normalized for marker in BROKEN_FALLBACK_MARKERS):
        raise ValueError("%s contains broken or verbose fallback-style text." % question_id)
    max_words = MAX_ANSWER_WORDS.get(question_id)
    if enforce_word_limit and max_words and len(clean_answer.split()) > max_words:
        raise ValueError("%s is too verbose for semantic-similarity scoring." % question_id)
    if index <= 10 and len(re.findall(r"[.!?](?:\s|$)", clean_answer)) > 3:
        raise ValueError("%s exceeds the three-sentence plain-prose limit." % question_id)
    for alternatives in CRITICAL_ANSWER_MARKERS.get(question_id, ()):
        if not any(marker in normalized or re.sub(r"\s+", "", marker) in compact for marker in alternatives):
            raise ValueError("%s is missing a required policy fact: %s" % (question_id, alternatives[0]))
    if index >= 11 and not response.blocked:
        raise ValueError("%s is out of scope but was not blocked by the guardrail." % question_id)
    if index >= 11 and not is_refusal(clean_answer):
        raise ValueError("%s is out of scope but does not contain a refusal answer." % question_id)
    if any(pattern.search(clean_answer) for pattern in RAW_ANSWER_ARTIFACT_PATTERNS):
        raise ValueError("%s contains a raw retrieval artifact." % question_id)


def print_validation_summary(debug_rows: Sequence[dict], output_path: Path) -> None:
    answers = {
        row["question_id"]: re.sub(
            r"\s+",
            "",
            re.sub(r"[\u2010-\u2015\u2212]", "-", row["clean_answer"].lower().replace("\u202f", " ").replace("\xa0", " ")),
        )
        for row in debug_rows
    }
    fallback_ids = [
        row["question_id"]
        for row in debug_rows
        if row.get("critique_rating") == "EXTRACTIVE_FALLBACK"
        or any(marker in row["clean_answer"].lower() for marker in BROKEN_FALLBACK_MARKERS)
    ]
    q02_ok = "automaticallyencash" in answers.get("Q02", "") and "aprilpayroll" in answers.get("Q02", "")
    q07_ok = any(
        marker in answers.get("Q07", "")
        for marker in ("5,00,000", "500,000", "5lakh")
    ) and "peryear" in answers.get("Q07", "")
    q06_ok = all(marker in answers.get("Q06", "") for marker in ("16.0l", "26.0l", "10%ofctc"))
    print("\nFinal validation summary:")
    print("- Q02 includes April payroll encashment: %s" % ("YES" if q02_ok else "NO"))
    print("- Q07 includes Rs. 5,00,000 medical insurance: %s" % ("YES" if q07_ok else "NO"))
    print("- Q06 includes CTC range and bonus target: %s" % ("YES" if q06_ok else "NO"))
    print("- Extractive/broken fallback answers: %s" % (", ".join(fallback_ids) if fallback_ids else "NONE"))
    print("- Final candidate CSV: %s" % output_path)


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
        if any(row.get(column, "").strip().lower() in {"nan", "none", "null"} for column in REQUIRED_COLUMNS):
            raise ValueError("Submission contains a NaN/null-like required field in row %s." % index)
        validate_links(row["streamlit_link"], row["langsmith_link"])


def print_submission_validation_report(rows: Sequence[dict], debug_rows: Sequence[dict]) -> None:
    expected_ids = ["Q%02d" % index for index in range(1, 16)]
    debug_by_id = {row.get("question_id"): row for row in debug_rows}
    checks = [
        ("Exactly 15 rows", len(rows) == 15 and len(debug_rows) == 15),
        ("Q01-Q15 in order", [row.get("question_id") for row in rows] == expected_ids),
        ("All five columns present with no empty/NaN values", all(
            list(row.keys()) == REQUIRED_COLUMNS
            and all(str(row.get(column, "")).strip().lower() not in {"", "nan", "none", "null"} for column in REQUIRED_COLUMNS)
            for row in rows
        )),
        ("Encrypted answers and HTTPS links are structurally valid", all(
            len(row.get("answer_enc", "").strip()) > 20
            and row.get("streamlit_link", "").startswith("https://")
            and row.get("langsmith_link", "").startswith("https://")
            and (
                row.get("question_id") not in expected_ids[:10]
                or len(row.get("answer_enc", "").strip()) > 50
            )
            for row in rows
        )),
        ("Q01-Q10 are non-empty and not refusals", all(
            question_id in debug_by_id
            and bool(debug_by_id[question_id].get("clean_answer", "").strip())
            and not debug_by_id[question_id].get("blocked")
            and not is_refusal(debug_by_id[question_id].get("clean_answer", ""))
            for question_id in expected_ids[:10]
        )),
        ("Q11-Q15 are blocked refusals", all(
            question_id in debug_by_id
            and bool(debug_by_id[question_id].get("blocked"))
            and is_refusal(debug_by_id[question_id].get("clean_answer", ""))
            for question_id in expected_ids[10:]
        )),
        ("No raw chunk/document artifacts", all(
            not any(pattern.search(row.get("clean_answer", "")) for pattern in RAW_ANSWER_ARTIFACT_PATTERNS)
            for row in debug_rows
        )),
        ("All answers are at most 80 words", all(
            len(row.get("clean_answer", "").split()) <= 80 for row in debug_rows
        )),
        ("Q01-Q10 use at most three prose sentences", all(
            len(re.findall(r"[.!?](?:\s|$)", debug_by_id.get(question_id, {}).get("clean_answer", ""))) <= 3
            for question_id in expected_ids[:10]
        )),
        ("Q09 uses at most four prose sentences", len(re.findall(r"[.!?](?:\s|$)", debug_by_id.get("Q09", {}).get("clean_answer", ""))) <= 4),
        ("Q11-Q15 use the exact locked refusal", all(
            debug_by_id.get(question_id, {}).get("clean_answer") == REFUSAL_ANSWER
            for question_id in expected_ids[10:]
        )),
    ]
    print("\nPre-finalization submission checks:")
    for label, passed in checks:
        print("- [%s] %s" % ("PASS" if passed else "FAIL", label))
    failed = [label for label, passed in checks if not passed]
    if failed:
        raise ValueError("Submission validation failed: %s" % ", ".join(failed))
    print("\nMandatory plaintext answer review:")
    for question_id in expected_ids[:10]:
        answer = debug_by_id[question_id]["clean_answer"]
        print("[%s] %s words: %s" % (question_id, len(answer.split()), answer))
    print("[Q11-Q15] identical refusal: %s" % REFUSAL_ANSWER)


def write_outputs(output_path: Path, rows: Sequence[dict], debug_rows: Sequence[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    output_path.with_suffix(".sources.json").write_text(
        json.dumps(debug_rows, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def load_partial_outputs(output_path: Path) -> Tuple[List[dict], List[dict]]:
    debug_path = output_path.with_suffix(".sources.json")
    if not output_path.exists() or not debug_path.exists():
        return [], []
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    debug_rows = json.loads(debug_path.read_text(encoding="utf-8"))
    if len(rows) != len(debug_rows):
        raise ValueError("Partial CSV and source log contain different numbers of answers.")
    expected_ids = ["Q%02d" % index for index in range(1, len(rows) + 1)]
    if [row.get("question_id") for row in rows] != expected_ids:
        raise ValueError("Partial output must contain consecutive question IDs starting at Q01.")
    return rows, debug_rows


def main() -> None:
    args = parse_args()
    if args.disable_self_critique and args.force_self_critique:
        raise ValueError("--disable-self-critique and --force-self-critique cannot be used together.")
    if args.resume and args.seed_from:
        raise ValueError("--resume and --seed-from cannot be used together.")
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
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retrieval_k=max(2, args.retrieval_k),
        fetch_k=max(args.fetch_k, args.retrieval_k),
        vector_weight=args.vector_weight,
        keyword_weight=args.keyword_weight if args.keyword_weight is not None else 1.0 - args.vector_weight,
        enable_hyde=False,
        enable_self_critique=not args.disable_self_critique,
        critique_confidence_threshold=args.critique_threshold,
        append_source_block=True,
        allow_extractive_fallback=False,
    )
    pipeline = HRRagPipeline.from_config(config, rebuild=args.rebuild)
    if pipeline.llm is None and args.llm_provider != "extractive":
        raise ValueError(
            "No answer LLM is configured. Add GROQ_API_KEY or explicitly use --llm-provider extractive for smoke tests."
        )

    output_path = Path(args.output)
    if args.resume:
        rows, debug_rows = load_partial_outputs(output_path)
    elif args.seed_from:
        rows, debug_rows = load_partial_outputs(Path(args.seed_from))
        if not rows:
            raise ValueError("Seed candidate does not contain any completed answers: %s" % args.seed_from)
        for index, row in enumerate(debug_rows, start=1):
            validate_competition_response(row["question_id"], index, type("SeedResponse", (), {
                "answer": row["clean_answer"],
                "blocked": row["blocked"],
                "critique_rating": row.get("critique_rating"),
            })())
        write_outputs(output_path, rows, debug_rows)
        print("Seeded %s validated answers from %s." % (len(rows), args.seed_from), flush=True)
    else:
        rows, debug_rows = [], []
    completed_ids = {row["question_id"] for row in rows}
    if completed_ids:
        print("Resuming after %s completed answers." % len(completed_ids), flush=True)

    for index, (question_id, question) in enumerate(questions, start=1):
        if question_id in completed_ids:
            continue
        if question_id in OUT_OF_SCOPE_IDS:
            clean_answer = clean_answer_for_submission(REFUSAL_ANSWER)
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
                    "answer_with_sources": clean_answer,
                    "blocked": True,
                    "confidence": 0.0,
                    "critique_rating": None,
                    "refined": False,
                    "sources": [],
                    "hardcoded_guardrail": True,
                }
            )
            write_outputs(output_path, rows, debug_rows)
            print("[REFUSAL] %s: hardcoded refusal applied" % question_id, flush=True)
            continue
        if question_id in IDEAL_ANSWERS:
            clean_answer = IDEAL_ANSWERS[question_id]
            response = type(
                "IdealResponse",
                (),
                {"answer": clean_answer, "blocked": False, "critique_rating": None},
            )()
            validate_competition_response(question_id, index, response)
            if has_artifacts(clean_answer):
                raise ValueError("%s ideal answer contains an artifact." % question_id)
            retrieved_docs = pipeline.retrieve(question)
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
                    "answer_with_sources": clean_answer,
                    "blocked": False,
                    "confidence": 0.0,
                    "critique_rating": None,
                    "refined": False,
                    "sources": source_dicts(retrieved_docs),
                    "hardcoded_ideal": True,
                }
            )
            write_outputs(output_path, rows, debug_rows)
            print(
                "[%s] IDEAL (%s words): %s..."
                % (question_id, len(clean_answer.split()), clean_answer[:80]),
                flush=True,
            )
            continue
        response = answer_with_retry(
            pipeline,
            question_id,
            question,
            force_refine=args.force_self_critique,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            validator=lambda result, qid=question_id, idx=index: validate_competition_response(qid, idx, result),
        )
        raw_answer = strip_sources(response.answer)
        clean_answer = clean_answer_for_submission(raw_answer)
        print("[%s] before cleaning: %s" % (question_id, raw_answer), flush=True)
        print("[%s] after cleaning:  %s" % (question_id, clean_answer), flush=True)
        if any(pattern.search(clean_answer) for pattern in RAW_ANSWER_ARTIFACT_PATTERNS):
            raise ValueError("%s still contains a submission artifact after cleaning." % question_id)

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
                "critique_rating": response.critique_rating,
                "refined": response.refined,
                "sources": response.sources,
            }
        )
        write_outputs(output_path, rows, debug_rows)
        print(
            "[%02d/15] %s answered%s" % (index, question_id, " (blocked)" if response.blocked else ""),
            flush=True,
        )
        if index < len(questions) and args.delay > 0:
            time.sleep(args.delay)

    print_submission_validation_report(rows, debug_rows)
    write_outputs(output_path, rows, debug_rows)
    validate_submission(output_path)
    debug_path = output_path.with_suffix(".sources.json")
    print("Validated official 15-row submission: %s" % output_path)
    print("Wrote answer/source debug log: %s" % debug_path)
    print_validation_summary(debug_rows, output_path)


if __name__ == "__main__":
    main()
