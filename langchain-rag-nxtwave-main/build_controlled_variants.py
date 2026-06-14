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
BASELINE = SUBMISSIONS / "variant_c_q12_partial_q15_specific_refusal.csv"
STARTER_NOTEBOOK = ROOT / "competition" / "Starter_Notebook.ipynb"

Q12_REQUIRED = (
    "Employee Stock Options (ESOP) are offered to employees at grade L5 and above with a 4-year vesting "
    "schedule on a 1-year cliff basis. ESOP eligibility begins when probation is confirmed. The policies "
    "do not specify how many stock options a new joiner will receive."
)
Q10_ANSWER = (
    "Permanent employees at grade L3 and above are eligible for WFH if they have completed at least 6 months "
    "of continuous service, have a most recent performance rating of Meets Expectations or above, and have no "
    "active Performance Improvement Plan. WFH arrangements include Hybrid WFH up to 3 days per week, Full "
    "Remote up to 5 days per week for L5 and above on a case-by-case basis, Ad-hoc WFH up to 2 days for "
    "unplanned requests, and Emergency WFH as directed by HR."
)
Q11_ANSWER = (
    "The supplied Zyro Dynamics HR policy documents do not specify how external candidates can apply for a "
    "job or describe the full recruitment and hiring process. I can only answer this if the recruitment policy "
    "is included in the provided documents."
)
Q15_ANSWER = (
    "From the Zyro Dynamics leave policy, Earned Leave is 15 days after one year of service and accrues at "
    "1.25 days per month. Up to 45 days of Earned Leave can be carried forward at the end of the financial "
    "year, and excess balance is automatically encashed and credited in April payroll. Sick Leave taken for "
    "more than 2 consecutive days requires a medical certificate within 3 working days of returning to work. "
    "The supplied documents do not contain Zoho or Freshworks leave policies, so I cannot compare them."
)
Q09_PROPOSED = (
    "The APR timeline includes 360-degree feedback from 1 to 20 February, self-assessment from 21 February "
    "to 10 March, manager assessment from 11 to 20 March, calibration from 21 to 25 March, and final ratings "
    "locked from 26 to 31 March. Increment and promotion letters are issued on 15 April."
)

EVIDENCE = {
    "Q10": [
        {
            "source_file": "03_Work_From_Home_Policy.pdf",
            "page": 1,
            "preview": "The WFH policy applies to all permanent employees at grade L3 and above.",
        },
        {
            "source_file": "03_Work_From_Home_Policy.pdf",
            "page": 2,
            "preview": (
                "Hybrid WFH: L3 and above, maximum 3 days/week. Full Remote: L5 and above case-by-case, "
                "maximum 5 days/week. Ad-hoc WFH: L3 and above, maximum 2 days. Emergency WFH: all employees, "
                "as directed by HR. Eligibility requires 6 months of service, Meets Expectations or above, and "
                "no active Performance Improvement Plan."
            ),
        },
    ],
    "Q11": [],
    "Q15": [
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 2,
            "preview": (
                "Earned Leave is 15 days after one year and accrues at 1.25 days/month. Sick Leave longer than "
                "2 consecutive days requires a medical certificate within 3 working days of returning to work."
            ),
        },
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 3,
            "preview": (
                "A maximum of 45 days of Earned Leave may be carried forward; excess is automatically encashed "
                "and credited in the April payroll."
            ),
        },
    ],
}

VARIANTS = {
    "variant_d_q10_complete_wfh.csv": {
        "answers": {"Q10": Q10_ANSWER},
    },
    "variant_e_q10_q15_partial.csv": {
        "answers": {"Q10": Q10_ANSWER, "Q15": Q15_ANSWER},
    },
    "variant_f_q10_q11_q15.csv": {
        "answers": {"Q10": Q10_ANSWER, "Q11": Q11_ANSWER, "Q15": Q15_ANSWER},
    },
    "variant_g_q09_q10_q11_q15.csv": {
        "answers": {"Q10": Q10_ANSWER, "Q11": Q11_ANSWER, "Q15": Q15_ANSWER},
        "conditional_skips": {
            "Q09": (
                "Not overridden: the proposed answer says self-assessment runs from 21 February to 10 March, "
                "but 05_Performance_Review_Policy.pdf page 3 confirms 1 to 10 March."
            )
        },
    },
}

PROTECTED_IDS = {f"Q{i:02d}" for i in range(1, 9)} | {"Q12", "Q13", "Q14"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_debug(path: Path) -> list[dict]:
    return json.loads(path.with_suffix(".sources.json").read_text(encoding="utf-8"))


def validate_variant(
    csv_path: Path,
    baseline_rows: list[dict[str, str]],
    expected_answers: dict[str, str],
    conditional_skips: dict[str, str],
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

    for question_id, reason in conditional_skips.items():
        assert rows_by_id[question_id] == baseline_by_id[question_id]
        assert debug_by_id[question_id]["conditional_override_applied"] is False
        assert debug_by_id[question_id]["conditional_override_reason"] == reason


def main() -> None:
    baseline_rows = load_csv(BASELINE)
    baseline_debug = load_debug(BASELINE)
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    baseline_debug_by_id = {row["question_id"]: row for row in baseline_debug}
    fernet, question_pairs = extract_competition_questions(str(STARTER_NOTEBOOK))
    official_questions = dict(question_pairs)

    assert len(baseline_rows) == len(baseline_debug) == 15
    assert baseline_by_id.keys() == baseline_debug_by_id.keys() == official_questions.keys()
    assert baseline_debug_by_id["Q12"]["clean_answer"] == Q12_REQUIRED

    for filename, config in VARIANTS.items():
        answers = config["answers"]
        conditional_skips = config.get("conditional_skips", {})
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
            if question_id in answers:
                debug.update(
                    {
                        "clean_answer": answers[question_id],
                        "answer_with_sources": answers[question_id],
                        "blocked": question_id == "Q11",
                        "confidence": {"Q10": 1.0, "Q11": 0.0, "Q15": 1.0}[question_id],
                        "critique_rating": None,
                        "refined": False,
                        "sources": EVIDENCE[question_id],
                        "hardcoded_guardrail": question_id == "Q11",
                        "controlled_override": True,
                    }
                )
            if question_id in conditional_skips:
                debug["conditional_override_applied"] = False
                debug["conditional_override_reason"] = conditional_skips[question_id]
                debug["proposed_answer_not_used"] = Q09_PROPOSED

        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temp_debug_path.write_text(json.dumps(debug_rows, ensure_ascii=True, indent=2), encoding="utf-8")

        validate_variant(temp_path, baseline_rows, answers, conditional_skips, fernet, official_questions)
        temp_path.replace(output_path)
        temp_debug_path.replace(output_path.with_suffix(".sources.json"))
        changed = ", ".join(answers)
        skipped = ", ".join(conditional_skips)
        print(f"PASS {output_path.name}: changed {changed}" + (f"; skipped {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
