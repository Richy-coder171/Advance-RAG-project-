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
BASELINE = SUBMISSIONS / "variant_r_q12_short_no_probation.csv"
STARTER_NOTEBOOK = ROOT / "competition" / "Starter_Notebook.ipynb"

Q03_ANSWER = (
    "Female employees are entitled to 26 weeks of paid Maternity Leave for the first two live births, with a "
    "minimum service requirement of 80 days in the 12 months preceding the expected date of delivery. Up to "
    "8 weeks may be taken before the expected date of delivery."
)
Q07_ANSWER = (
    "Group Medical Insurance provides coverage of up to Rs. 5,00,000 per year under a floater policy for the "
    "employee, spouse, and up to two dependent children, with all premiums fully paid by the Company. "
    "Employees also receive Personal Accident Insurance coverage of 5 times annual CTC and Term Life "
    "Insurance coverage of 3 times annual CTC for all permanent employees."
)
Q10_ANSWER = (
    "All permanent employees at grade L3 and above are eligible for WFH. Hybrid WFH allows fixed WFH days "
    "agreed with the reporting manager for L3 and above employees, up to 3 days per week. Full Remote is "
    "available for L5 and above on a case-by-case basis, up to 5 days per week. Ad-hoc WFH is available for "
    "L3 and above for up to 2 days for unplanned requests, and Emergency WFH is available to all employees "
    "as directed by HR."
)
Q12_ANSWER = (
    "Employee Stock Options (ESOP) are offered to employees at grade L5 and above. The ESOP vesting schedule "
    "is 4 years with a 1-year cliff: 25% in Year 1, 25% in Year 2, and 50% in Year 4. The policy documents do "
    "not specify how many stock options a new joiner will receive."
)

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
                "to two dependent children, with premiums paid by the Company. Personal Accident Insurance is "
                "5 times annual CTC and Term Life Insurance is 3 times annual CTC for permanent employees."
            ),
        }
    ],
    "Q10": [
        {
            "source_file": "03_Work_From_Home_Policy.pdf",
            "page": 2,
            "preview": (
                "Hybrid WFH is fixed days agreed with the reporting manager for L3 and above, up to 3 days. "
                "Full Remote is L5 and above case-by-case, up to 5 days. Ad-hoc WFH is L3 and above, up to "
                "2 days. Emergency WFH is available to all employees as directed by HR."
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

UNSUPPORTED_CLAIMS = {
    "Q07": ["The official policy does not describe Group Medical Insurance as a floater policy."],
    "Q12": [
        "The official policy does not specify the proposed 25% in Year 1, 25% in Year 2, and 50% in Year 4 split."
    ],
}

VARIANTS = {
    "variant_w_q07_full_insurance.csv": {"Q07": Q07_ANSWER},
    "variant_x_q12_full_vesting.csv": {"Q12": Q12_ANSWER},
    "variant_y_q03_maternity_prenatal.csv": {"Q03": Q03_ANSWER},
    "variant_z_q10_types_only_exact.csv": {"Q10": Q10_ANSWER},
}

PROTECTED_IDS = {"Q01", "Q02", "Q04", "Q05", "Q06", "Q08", "Q09", "Q11", "Q13", "Q14", "Q15"}


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
            assert debug_by_id[question_id].get("unsupported_claims", []) == UNSUPPORTED_CLAIMS.get(question_id, [])
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
                    "blocked": False,
                    "confidence": 1.0,
                    "critique_rating": None,
                    "refined": False,
                    "sources": EVIDENCE[question_id],
                    "hardcoded_guardrail": False,
                    "controlled_override": True,
                    "unsupported_claims": UNSUPPORTED_CLAIMS.get(question_id, []),
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
