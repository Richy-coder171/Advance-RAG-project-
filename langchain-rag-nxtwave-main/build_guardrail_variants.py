from __future__ import annotations

import csv
import json
import re
from copy import deepcopy
from pathlib import Path

from generate_competition_submission import (
    REQUIRED_COLUMNS,
    extract_competition_questions,
    validate_submission,
)


ROOT = Path(__file__).resolve().parent
SUBMISSIONS = ROOT / "submissions"
BASELINE = SUBMISSIONS / "score_90_49_submission.csv"
STARTER_NOTEBOOK = ROOT / "competition" / "Starter_Notebook.ipynb"

Q11_ANSWER = (
    "The provided policies do not specify how to apply for a job or describe recruitment, interviews, "
    "or candidate selection. After an offer is accepted, HR issues the offer and appointment letter, "
    "collects education, identity, and experience proof 7 days before joining, initiates background and "
    "reference verification 7 days before joining, and sends the Day 1 schedule and reporting instructions "
    "the day before joining."
)
Q12_ANSWER = (
    "Employee Stock Options (ESOP) are offered to employees at grade L5 and above with a 4-year vesting "
    "schedule on a 1-year cliff basis. ESOP eligibility begins when probation is confirmed. The policies "
    "do not specify how many stock options a new joiner will receive."
)
Q15_ANSWER = (
    "I can only provide Zyro Dynamics leave policy information from the supplied documents. The documents "
    "do not contain the leave policies of Zoho or Freshworks, so I cannot compare them."
)

EVIDENCE = {
    "Q11": [
        {
            "source_file": "09_Onboarding_and_Separation_Policy.pdf",
            "page": 2,
            "preview": (
                "Upon offer acceptance, HR issues the offer and appointment letter. HR collects education, "
                "identity, and experience proof and initiates background and reference verification 7 days "
                "before joining, then sends the Day 1 schedule and reporting instructions the day before joining."
            ),
        }
    ],
    "Q12": [
        {
            "source_file": "06_Compensation_and_Benefits_Policy.pdf",
            "page": 3,
            "preview": (
                "Employee Stock Options (ESOP): Offered to employees at grade L5 and above, with a 4-year "
                "vesting schedule on a 1-year cliff basis."
            ),
        },
        {
            "source_file": "09_Onboarding_and_Separation_Policy.pdf",
            "page": 2,
            "preview": "Benefits including ESOP eligibility and full leave accrual commence from the date probation is confirmed.",
        },
    ],
    "Q15": [
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 1,
            "preview": "This policy applies to all Zyro Dynamics employees.",
        }
    ],
}

VARIANTS = {
    "variant_a_q12_partial.csv": {"Q12": Q12_ANSWER},
    "variant_b_q11_q12_partial.csv": {"Q11": Q11_ANSWER, "Q12": Q12_ANSWER},
    "variant_c_q12_partial_q15_specific_refusal.csv": {"Q12": Q12_ANSWER, "Q15": Q15_ANSWER},
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_debug(path: Path) -> list[dict]:
    return json.loads(path.with_suffix(".sources.json").read_text(encoding="utf-8"))


def validate_variant(
    csv_path: Path,
    debug_path: Path,
    baseline_rows: list[dict[str, str]],
    expected_answers: dict[str, str],
    fernet,
    official_questions: dict[str, str],
) -> None:
    validate_submission(csv_path)
    rows = load_csv(csv_path)
    debug_rows = load_debug(debug_path.with_suffix(""))
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    debug_by_id = {row["question_id"]: row for row in debug_rows}

    assert len(debug_rows) == 15
    assert [row["question_id"] for row in rows] == [f"Q{i:02d}" for i in range(1, 16)]
    assert list(rows[0]) == REQUIRED_COLUMNS

    for row in rows:
        question_id = row["question_id"]
        baseline = baseline_by_id[question_id]
        assert row["question_enc"] == baseline["question_enc"]
        assert row["streamlit_link"] == baseline["streamlit_link"]
        assert row["langsmith_link"] == baseline["langsmith_link"]
        assert fernet.decrypt(row["question_enc"].encode("ascii")).decode("utf-8") == official_questions[question_id]

        answer = fernet.decrypt(row["answer_enc"].encode("ascii")).decode("utf-8")
        if question_id in expected_answers:
            assert answer == expected_answers[question_id]
            assert debug_by_id[question_id]["clean_answer"] == answer
            assert debug_by_id[question_id]["answer_with_sources"] == answer
            assert debug_by_id[question_id]["sources"] == EVIDENCE[question_id]
            assert not re.search(r"\*\*|(?:^|\n)\s*(?:[-*]|\d+\.)\s+", answer)
        else:
            assert row == baseline

    q12 = expected_answers.get("Q12")
    if q12:
        normalized = q12.lower()
        assert all(term in normalized for term in ("l5", "4-year", "1-year cliff", "not specify"))
    q11 = expected_answers.get("Q11")
    if q11:
        normalized = q11.lower()
        assert all(term in normalized for term in ("do not specify", "offer", "background", "reference verification"))
    q15 = expected_answers.get("Q15")
    if q15:
        normalized = q15.lower()
        assert all(term in normalized for term in ("zyro dynamics", "zoho", "freshworks", "cannot compare"))


def main() -> None:
    baseline_rows = load_csv(BASELINE)
    baseline_debug = load_debug(BASELINE)
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    baseline_debug_by_id = {row["question_id"]: row for row in baseline_debug}
    fernet, question_pairs = extract_competition_questions(str(STARTER_NOTEBOOK))
    official_questions = dict(question_pairs)

    assert len(baseline_rows) == 15
    assert len(baseline_debug) == 15
    assert baseline_by_id.keys() == baseline_debug_by_id.keys() == official_questions.keys()

    for filename, replacements in VARIANTS.items():
        output_path = SUBMISSIONS / filename
        temp_path = SUBMISSIONS / f".{filename}.tmp"
        temp_debug_path = temp_path.with_suffix(".sources.json")
        rows = deepcopy(baseline_rows)
        debug_rows = deepcopy(baseline_debug)

        for row in rows:
            question_id = row["question_id"]
            if question_id in replacements:
                row["answer_enc"] = fernet.encrypt(replacements[question_id].encode("utf-8")).decode("ascii")

        for debug in debug_rows:
            question_id = debug["question_id"]
            if question_id in replacements:
                debug.update(
                    {
                        "clean_answer": replacements[question_id],
                        "answer_with_sources": replacements[question_id],
                        "blocked": question_id == "Q15",
                        "confidence": {"Q11": 0.8, "Q12": 1.0, "Q15": 0.0}[question_id],
                        "critique_rating": None,
                        "refined": False,
                        "sources": EVIDENCE[question_id],
                        "hardcoded_guardrail": question_id == "Q15",
                        "guardrail_experiment": True,
                    }
                )

        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temp_debug_path.write_text(json.dumps(debug_rows, ensure_ascii=True, indent=2), encoding="utf-8")

        # Only publish final filenames after the temporary outputs pass all checks.
        validate_variant(temp_path, temp_debug_path, baseline_rows, replacements, fernet, official_questions)
        temp_path.replace(output_path)
        temp_debug_path.replace(output_path.with_suffix(".sources.json"))
        print(f"PASS {output_path.name}: changed {', '.join(replacements)}")


if __name__ == "__main__":
    main()
