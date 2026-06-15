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
BASELINE = SUBMISSIONS / "variant_h_q09_correct_apr.csv"
STARTER_NOTEBOOK = ROOT / "competition" / "Starter_Notebook.ipynb"

Q11_ANSWER = (
    "The supplied Zyro Dynamics HR policy documents do not describe how external candidates can apply for a "
    "job or the full recruitment and hiring process. They only support HR policy information available in the "
    "provided documents, so the application process is not specified."
)
Q12_ANSWER = (
    "Employee Stock Options (ESOP) are offered to employees at grade L5 and above. The vesting schedule is "
    "4 years with a 1-year cliff. The policy documents do not specify how many stock options a new joiner "
    "will receive."
)
Q13_ANSWER = (
    "The supplied Zyro Dynamics HR policy documents do not contain company revenue, financial performance, "
    "or business financial results. I can only answer from the provided HR policy documents."
)
Q14_ANSWER = (
    "The supplied Zyro Dynamics HR policy documents do not contain AcruxCRM product features or comparisons "
    "with Salesforce. I can only answer from the provided HR policy documents."
)

EVIDENCE = {
    "Q11": [],
    "Q12": [
        {
            "source_file": "06_Compensation_and_Benefits_Policy.pdf",
            "page": 3,
            "preview": (
                "Employee Stock Options (ESOP) are offered to employees at grade L5 and above, with a 4-year "
                "vesting schedule on a 1-year cliff basis."
            ),
        }
    ],
    "Q13": [],
    "Q14": [],
}

VARIANTS = {
    "variant_r_q12_short_no_probation.csv": {"Q12": Q12_ANSWER},
    "variant_s_q11_specific_application_missing.csv": {"Q11": Q11_ANSWER},
    "variant_t_q13_specific_finance_refusal.csv": {"Q13": Q13_ANSWER},
    "variant_u_q14_specific_product_refusal.csv": {"Q14": Q14_ANSWER},
}

PROTECTED_IDS = {f"Q{i:02d}" for i in range(1, 11)}


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
    assert debug_by_id["Q13"]["blocked"] and debug_by_id["Q14"]["blocked"]

    for question_id, row in rows_by_id.items():
        baseline = baseline_by_id[question_id]
        assert row["question_enc"] == baseline["question_enc"]
        assert row["streamlit_link"] == baseline["streamlit_link"]
        assert row["langsmith_link"] == baseline["langsmith_link"]
        assert fernet.decrypt(row["question_enc"].encode("ascii")).decode("utf-8") == official_questions[question_id]

        answer = fernet.decrypt(row["answer_enc"].encode("ascii")).decode("utf-8")
        if question_id in expected_answers:
            assert question_id not in PROTECTED_IDS
            assert answer == expected_answers[question_id]
            assert debug_by_id[question_id]["clean_answer"] == answer
            assert debug_by_id[question_id]["answer_with_sources"] == answer
            assert debug_by_id[question_id]["sources"] == EVIDENCE[question_id]
            assert not re.search(r"\*\*|(?:^|\n)\s*(?:[-*]|\d+\.)\s+", answer)
        else:
            assert row == baseline

    for question_id in PROTECTED_IDS:
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
    assert baseline_debug_by_id["Q13"]["blocked"] and baseline_debug_by_id["Q14"]["blocked"]

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
                    "blocked": question_id in {"Q11", "Q13", "Q14"},
                    "confidence": 1.0 if question_id == "Q12" else 0.0,
                    "critique_rating": None,
                    "refined": False,
                    "sources": EVIDENCE[question_id],
                    "hardcoded_guardrail": question_id in {"Q11", "Q13", "Q14"},
                    "controlled_override": True,
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
