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
BASELINE = SUBMISSIONS / "variant_w_q07_full_insurance.csv"
STARTER_NOTEBOOK = ROOT / "competition" / "Starter_Notebook.ipynb"

Q03_ANSWER = (
    "Female employees are entitled to 26 weeks of paid Maternity Leave for the first two live births, with a "
    "minimum service requirement of 80 days of service in the 12 months preceding the expected date of "
    "delivery. Up to 8 weeks may be taken before the expected date of delivery."
)
Q07_ANSWER = (
    "Group Medical Insurance provides coverage of up to Rs. 5,00,000 per year for the employee, spouse, and "
    "up to two dependent children, with all premiums fully paid by the Company. Employees also receive "
    "Personal Accident Insurance coverage of 5 times annual CTC and Term Life Insurance coverage of 3 times "
    "annual CTC for all permanent employees."
)
Q09_REQUIRED = (
    "The APR timeline includes 360-degree feedback collected from peers and subordinates from 1 to 20 "
    "February, self-assessment from 1 to 10 March, manager assessment from 11 to 20 March, calibration from "
    "21 to 25 March, and final ratings locked from 26 to 31 March. Increment and promotion letters are issued "
    "on 15 April."
)
Q10_REQUIRED = (
    "All permanent employees at grade L3 and above are eligible for WFH. Hybrid WFH allows up to 3 days per "
    "week, Full Remote allows up to 5 days per week for L5 and above on a case-by-case basis, and Ad-hoc WFH "
    "allows up to 2 days for unplanned requests. Emergency WFH is available to all employees as directed by HR."
)
Q11_ANSWER = (
    "The supplied Zyro Dynamics HR policy documents do not describe how external candidates can apply for a "
    "job or the full recruitment and hiring process. I can only answer questions covered in the provided HR "
    "policy documents."
)
Q12_REQUIRED = (
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

AC_ANSWERS = {"Q03": Q03_ANSWER, "Q07": Q07_ANSWER}
AE_ANSWERS = {**AC_ANSWERS, "Q11": Q11_ANSWER, "Q13": Q13_ANSWER, "Q14": Q14_ANSWER}

EVIDENCE = {
    "Q03": [
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 3,
            "preview": (
                "Female employees with at least 80 days of service in the 12 months preceding the expected "
                "delivery date receive 26 weeks of paid Maternity Leave for the first two live births. Up to "
                "8 weeks of pre-natal leave may be taken before the expected delivery date."
            ),
        }
    ],
    "Q07": [
        {
            "source_file": "06_Compensation_and_Benefits_Policy.pdf",
            "page": 3,
            "preview": (
                "Group Medical Insurance covers up to Rs. 5,00,000 per year for the employee, spouse, and up "
                "to two dependent children, with premiums fully paid by the Company. Personal Accident "
                "Insurance is 5 times annual CTC and Term Life Insurance is 3 times annual CTC for permanent "
                "employees."
            ),
        }
    ],
    "Q11": [],
    "Q13": [],
    "Q14": [],
}

VARIANTS = {
    "variant_ac_clean_93_push.csv": AC_ANSWERS,
    "variant_ae_ac_plus_specific_refusals.csv": AE_ANSWERS,
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
    assert debug_by_id["Q09"]["clean_answer"] == Q09_REQUIRED
    assert debug_by_id["Q10"]["clean_answer"] == Q10_REQUIRED
    assert debug_by_id["Q12"]["clean_answer"] == Q12_REQUIRED
    assert debug_by_id["Q13"]["blocked"] and debug_by_id["Q14"]["blocked"]

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

    # These strong/base answers must remain byte-for-byte unchanged.
    for question_id in ("Q09", "Q10", "Q12", "Q15"):
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
    assert baseline_debug_by_id["Q09"]["clean_answer"] == Q09_REQUIRED
    assert baseline_debug_by_id["Q10"]["clean_answer"] == Q10_REQUIRED
    assert baseline_debug_by_id["Q12"]["clean_answer"] == Q12_REQUIRED

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
                    "confidence": 0.0 if question_id in {"Q11", "Q13", "Q14"} else 1.0,
                    "critique_rating": None,
                    "refined": False,
                    "sources": EVIDENCE[question_id],
                    "hardcoded_guardrail": question_id in {"Q11", "Q13", "Q14"},
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
