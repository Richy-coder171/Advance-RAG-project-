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
BASELINE = SUBMISSIONS / "variant_ae_ac_plus_specific_refusals.csv"
STARTER_NOTEBOOK = ROOT / "competition" / "Starter_Notebook.ipynb"

Q01_ANSWER = (
    "Earned Leave accrues at 1.25 days per month after completion of one year of continuous service. "
    "Employees are entitled to 15 days of Earned Leave after completing one year of service."
)
Q06_ANSWER = "CTC Range: Rs. 16.0L to Rs. 26.0L. Bonus Target: 10% of CTC."
Q08_ANSWER = (
    "An employee who receives a rating of 1 or 2 in two consecutive review cycles is placed on a formal "
    "Performance Improvement Plan. The PIP is a structured and time-bound programme. The PIP duration is "
    "60 to 90 days, as determined by the reporting manager and HR Business Partner."
)
Q09_ANSWER = (
    "360 degree feedback is collected from peers and subordinates from 1 to 20 February; employee "
    "self-assessment is submitted from 1 to 10 March; manager assessment runs from 11 to 20 March; "
    "calibration is held from 21 to 25 March; final ratings are locked from 26 to 31 March. Increment and "
    "promotion letters are issued on 15 April."
)
Q10_REQUIRED = (
    "All permanent employees at grade L3 and above are eligible for WFH. Hybrid WFH allows up to 3 days per "
    "week, Full Remote allows up to 5 days per week for L5 and above on a case-by-case basis, and Ad-hoc WFH "
    "allows up to 2 days for unplanned requests. Emergency WFH is available to all employees as directed by HR."
)
Q11_ANSWER = (
    "The provided Zyro Dynamics HR policy documents do not include an external job application process, "
    "recruitment process, or hiring workflow. Therefore, the application and hiring process is not specified "
    "in the supplied documents."
)
Q13_ANSWER = (
    "Company revenue, financial performance, and business results are not covered in the supplied Zyro "
    "Dynamics HR policy documents."
)
Q14_ANSWER = (
    "AcruxCRM product features and Salesforce comparisons are not covered in the supplied Zyro Dynamics HR "
    "policy documents."
)

BG_ANSWERS = {"Q06": Q06_ANSWER, "Q09": Q09_ANSWER, "Q13": Q13_ANSWER, "Q14": Q14_ANSWER}
BH_ANSWERS = {**BG_ANSWERS, "Q01": Q01_ANSWER}
BI_ANSWERS = {**BG_ANSWERS, "Q08": Q08_ANSWER, "Q11": Q11_ANSWER}
BJ_ANSWERS = {**BG_ANSWERS, "Q01": Q01_ANSWER, "Q08": Q08_ANSWER, "Q11": Q11_ANSWER}

EVIDENCE = {
    "Q01": [
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 2,
            "preview": (
                "Employees become eligible for 15 days of Earned Leave upon completion of one year of "
                "continuous service. Thereafter, Earned Leave accrues at 1.25 days per month."
            ),
        }
    ],
    "Q06": [
        {
            "source_file": "06_Compensation_and_Benefits_Policy.pdf",
            "page": 3,
            "preview": "L4 Senior salary band: CTC Range Rs. 16.0L to Rs. 26.0L; Bonus Target 10% of CTC.",
        }
    ],
    "Q08": [
        {
            "source_file": "05_Performance_Review_Policy.pdf",
            "page": 3,
            "preview": (
                "A rating of 1 or 2 in two consecutive review cycles results in a formal PIP. The PIP is a "
                "structured and time-bound programme with a duration of 60 to 90 days."
            ),
        }
    ],
    "Q09": [
        {
            "source_file": "05_Performance_Review_Policy.pdf",
            "page": 3,
            "preview": (
                "360 degree feedback: 1 to 20 February; self-assessment: 1 to 10 March; manager assessment: "
                "11 to 20 March; calibration: 21 to 25 March; final ratings: 26 to 31 March; increment and "
                "promotion letters: 15 April."
            ),
        }
    ],
    "Q11": [],
    "Q13": [],
    "Q14": [],
}

VARIANTS = {
    "variant_bg_source_exact_attack.csv": BG_ANSWERS,
    "variant_bh_bg_plus_q01_short.csv": BH_ANSWERS,
    "variant_bi_bg_plus_q08_q11.csv": BI_ANSWERS,
    "variant_bj_full_attack_combo.csv": BJ_ANSWERS,
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
    assert debug_by_id["Q10"]["clean_answer"] == Q10_REQUIRED

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

    # Q10 and Q15 are known-sensitive and must stay byte-for-byte unchanged.
    assert rows_by_id["Q10"] == baseline_by_id["Q10"]
    assert rows_by_id["Q15"] == baseline_by_id["Q15"]


def main() -> None:
    baseline_rows = load_csv(BASELINE)
    baseline_debug = load_debug(BASELINE)
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    baseline_debug_by_id = {row["question_id"]: row for row in baseline_debug}
    fernet, question_pairs = extract_competition_questions(str(STARTER_NOTEBOOK))
    official_questions = dict(question_pairs)

    assert len(baseline_rows) == len(baseline_debug) == 15
    assert baseline_by_id.keys() == baseline_debug_by_id.keys() == official_questions.keys()
    assert baseline_debug_by_id["Q10"]["clean_answer"] == Q10_REQUIRED

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
