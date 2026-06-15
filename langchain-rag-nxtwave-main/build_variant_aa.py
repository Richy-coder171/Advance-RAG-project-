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
OUTPUT = SUBMISSIONS / "variant_aa_q03_q07_q09_q12_top_scorer_style.csv"
STARTER_NOTEBOOK = ROOT / "competition" / "Starter_Notebook.ipynb"

Q03_ANSWER = (
    "Female employees are entitled to 26 weeks of paid Maternity Leave for the first two live births, with a "
    "minimum service requirement of 80 days of service in the 12 months preceding the expected date of "
    "delivery. Up to 8 weeks may be taken before the expected date of delivery."
)
Q07_REQUIRED = (
    "Group Medical Insurance provides coverage of up to Rs. 5,00,000 per year under a floater policy for the "
    "employee, spouse, and up to two dependent children, with all premiums fully paid by the Company. "
    "Employees also receive Personal Accident Insurance coverage of 5 times annual CTC and Term Life "
    "Insurance coverage of 3 times annual CTC for all permanent employees."
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
Q12_ANSWER = (
    "Employee Stock Options (ESOP) are offered to employees at grade L5 and above. The ESOP vesting schedule "
    "is 4 years with a 1-year cliff: 25% in Year 1, 25% in Year 2, and 50% in Year 4. The policy documents do "
    "not specify how many stock options a new joiner will receive."
)

ANSWERS = {"Q03": Q03_ANSWER, "Q12": Q12_ANSWER}
PROTECTED_IDS = {"Q01", "Q02", "Q04", "Q05", "Q06", "Q08", "Q10", "Q11", "Q13", "Q14", "Q15"}

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
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_debug(path: Path) -> list[dict]:
    return json.loads(path.with_suffix(".sources.json").read_text(encoding="utf-8"))


def validate_variant(
    csv_path: Path,
    baseline_rows: list[dict[str, str]],
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
    assert debug_by_id["Q07"]["clean_answer"] == Q07_REQUIRED
    assert debug_by_id["Q09"]["clean_answer"] == Q09_REQUIRED
    assert debug_by_id["Q10"]["clean_answer"] == Q10_REQUIRED
    assert debug_by_id["Q13"]["blocked"] and debug_by_id["Q14"]["blocked"]

    for question_id, row in rows_by_id.items():
        baseline = baseline_by_id[question_id]
        assert row["question_enc"] == baseline["question_enc"]
        assert row["streamlit_link"] == baseline["streamlit_link"]
        assert row["langsmith_link"] == baseline["langsmith_link"]
        assert fernet.decrypt(row["question_enc"].encode("ascii")).decode("utf-8") == official_questions[question_id]

        answer = fernet.decrypt(row["answer_enc"].encode("ascii")).decode("utf-8")
        if question_id in ANSWERS:
            assert answer == ANSWERS[question_id]
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
    assert baseline_debug_by_id["Q07"]["clean_answer"] == Q07_REQUIRED
    assert baseline_debug_by_id["Q09"]["clean_answer"] == Q09_REQUIRED
    assert baseline_debug_by_id["Q10"]["clean_answer"] == Q10_REQUIRED

    rows = deepcopy(baseline_rows)
    debug_rows = deepcopy(baseline_debug)
    for row in rows:
        question_id = row["question_id"]
        if question_id in ANSWERS:
            row["answer_enc"] = fernet.encrypt(ANSWERS[question_id].encode("utf-8")).decode("ascii")

    for debug in debug_rows:
        question_id = debug["question_id"]
        if question_id not in ANSWERS:
            continue
        debug.update(
            {
                "clean_answer": ANSWERS[question_id],
                "answer_with_sources": ANSWERS[question_id],
                "blocked": False,
                "confidence": 1.0,
                "critique_rating": None,
                "refined": False,
                "sources": EVIDENCE[question_id],
                "hardcoded_guardrail": False,
                "controlled_override": True,
                "unsupported_claims": (
                    [
                        "The official policy does not specify the proposed 25% in Year 1, 25% in Year 2, "
                        "and 50% in Year 4 split."
                    ]
                    if question_id == "Q12"
                    else []
                ),
            }
        )

    temp_path = SUBMISSIONS / f".{OUTPUT.name}.tmp"
    temp_debug_path = temp_path.with_suffix(".sources.json")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temp_debug_path.write_text(json.dumps(debug_rows, ensure_ascii=True, indent=2), encoding="utf-8")

    validate_variant(temp_path, baseline_rows, fernet, official_questions)
    temp_path.replace(OUTPUT)
    temp_debug_path.replace(OUTPUT.with_suffix(".sources.json"))
    print(f"PASS {OUTPUT.name}: changed Q03, Q12; retained Q07, Q09, Q10")


if __name__ == "__main__":
    main()
