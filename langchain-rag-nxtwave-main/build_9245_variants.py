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
BASELINE = SUBMISSIONS / "variant_q11_q12_refusal_fix.csv"
STARTER_NOTEBOOK = ROOT / "competition" / "Starter_Notebook.ipynb"

Q01_ANSWER = (
    "Earned Leave accrues at 1.25 days per month after completion of one year of continuous service. "
    "Employees become eligible for 15 days of Earned Leave upon completion of one year of continuous service, "
    "provided they have worked for a minimum of 240 days in that year."
)
Q02_ANSWER = (
    "A maximum of 45 days of Earned Leave may be carried forward at the end of each financial year. Any "
    "balance exceeding this limit will be automatically encashed at the employee's basic daily rate and "
    "credited in the April payroll."
)
Q03_ANSWER = (
    "Female employees are entitled to 26 weeks of paid Maternity Leave, with a minimum service requirement "
    "of 80 days of service in the 12 months preceding the expected date of delivery."
)
Q04_ANSWER = (
    "A Medical Certificate from a registered medical practitioner is required when Sick Leave is taken for "
    "more than 2 consecutive days, and must be submitted within 3 working days of returning to work."
)
Q05_ANSWER = (
    "Salaries are credited to the employee's registered bank account by the 7th of the following month. "
    "The payroll cut-off date is the 24th of each month."
)

EVIDENCE = {
    "Q01": [
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 2,
            "preview": (
                "Employees become eligible for 15 days of Earned Leave upon completion of one year of "
                "continuous service, provided they have worked for a minimum of 240 days. Thereafter, "
                "Earned Leave accrues at 1.25 days per month."
            ),
        }
    ],
    "Q02": [
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 3,
            "preview": (
                "A maximum of 45 days of Earned Leave may be carried forward at the end of each financial "
                "year. Any balance exceeding this limit is automatically encashed and credited in April payroll."
            ),
        }
    ],
    "Q03": [
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 3,
            "preview": (
                "Female employees who have completed 80 days of service in the 12 months preceding the "
                "expected date of delivery are entitled to 26 weeks of paid Maternity Leave."
            ),
        }
    ],
    "Q04": [
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 2,
            "preview": (
                "Sick Leave taken for more than 2 consecutive days requires a Medical Certificate from a "
                "registered medical practitioner, submitted within 3 working days of returning to work."
            ),
        }
    ],
    "Q05": [
        {
            "source_file": "06_Compensation_and_Benefits_Policy.pdf",
            "page": 1,
            "preview": (
                "Salaries and professional fees are processed and credited to the employee's registered bank "
                "account by the 7th of the following month. The payroll cut-off date is the 24th of each month."
            ),
        }
    ],
}

VARIANTS = {
    "variant_q05_no_professional_fees.csv": {"Q05": Q05_ANSWER},
    "variant_q01_no_rate_of.csv": {"Q01": Q01_ANSWER},
    "variant_q03_no_births.csv": {"Q03": Q03_ANSWER},
    "variant_q02_no_parenthetical.csv": {"Q02": Q02_ANSWER},
    "variant_q04_reordered.csv": {"Q04": Q04_ANSWER},
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_debug(path: Path) -> list[dict]:
    return json.loads(path.with_suffix(".sources.json").read_text(encoding="utf-8"))


def validate_variant(
    csv_path: Path,
    baseline_rows: list[dict[str, str]],
    expected_answers: dict[str, str],
    fernet,
    official_questions: dict[str, str],
) -> None:
    validate_submission(csv_path)
    rows = load_csv(csv_path)
    debug_rows = load_debug(csv_path)
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    rows_by_id = {row["question_id"]: row for row in rows}
    debug_by_id = {row["question_id"]: row for row in debug_rows}

    assert len(rows) == len(debug_rows) == 15
    assert [row["question_id"] for row in rows] == [f"Q{i:02d}" for i in range(1, 16)]
    assert list(rows[0]) == REQUIRED_COLUMNS

    for question_id, row in rows_by_id.items():
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
            assert debug_by_id[question_id].get("unsupported_claims", []) == []
            assert not re.search(r"\*\*|(?:^|\n)\s*(?:[-*]|\d+\.)\s+", answer)
        else:
            assert row == baseline

    # Locked rows are checked relative to the actual scored base file.
    locked_ids = {"Q06", "Q08", "Q09", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"}
    for question_id in locked_ids - set(expected_answers):
        assert rows_by_id[question_id] == baseline_by_id[question_id]


def main() -> None:
    baseline_rows = load_csv(BASELINE)
    baseline_debug = load_debug(BASELINE)
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    baseline_debug_by_id = {row["question_id"]: row for row in baseline_debug}
    fernet, question_pairs = extract_competition_questions(str(STARTER_NOTEBOOK))
    official_questions = dict(question_pairs)

    assert len(baseline_rows) == len(baseline_debug) == 15
    assert baseline_by_id.keys() == baseline_debug_by_id.keys() == official_questions.keys()

    for filename, answers in VARIANTS.items():
        output_path = SUBMISSIONS / filename
        temp_path = SUBMISSIONS / f".{filename}.tmp"
        temp_debug_path = temp_path.with_suffix(".sources.json")
        rows = deepcopy(baseline_rows)
        debug_rows = deepcopy(baseline_debug)

        for row in rows:
            question_id = row["question_id"]
            if question_id in answers:
                row["answer_enc"] = fernet.encrypt(answers[question_id].encode("utf-8")).decode("ascii")

        for debug in debug_rows:
            question_id = debug["question_id"]
            if question_id not in answers:
                continue
            debug.update(
                {
                    "clean_answer": answers[question_id],
                    "answer_with_sources": answers[question_id],
                    "blocked": False,
                    "confidence": 1.0,
                    "critique_rating": None,
                    "refined": False,
                    "sources": EVIDENCE[question_id],
                    "hardcoded_guardrail": False,
                    "controlled_override": True,
                    "unsupported_claims": [],
                }
            )

        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temp_debug_path.write_text(json.dumps(debug_rows, ensure_ascii=True, indent=2), encoding="utf-8")

        validate_variant(temp_path, baseline_rows, answers, fernet, official_questions)
        temp_path.replace(output_path)
        temp_debug_path.replace(output_path.with_suffix(".sources.json"))
        print(f"PASS {output_path.name}: changed {', '.join(answers)}")


if __name__ == "__main__":
    main()
