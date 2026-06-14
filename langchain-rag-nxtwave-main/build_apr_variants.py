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

Q09_ANSWER = (
    "The APR timeline includes 360-degree feedback collected from peers and subordinates from 1 to 20 "
    "February, self-assessment from 1 to 10 March, manager assessment from 11 to 20 March, calibration from "
    "21 to 25 March, and final ratings locked from 26 to 31 March. Increment and promotion letters are issued "
    "on 15 April."
)
Q15_ANSWER = (
    "I can only provide Zyro Dynamics leave policy information from the supplied documents. The documents "
    "do not contain the leave policies of Zoho or Freshworks, so I cannot compare them."
)
Q12_REQUIRED = (
    "Employee Stock Options (ESOP) are offered to employees at grade L5 and above with a 4-year vesting "
    "schedule on a 1-year cliff basis. ESOP eligibility begins when probation is confirmed. The policies "
    "do not specify how many stock options a new joiner will receive."
)

Q09_EVIDENCE = [
    {
        "source_file": "05_Performance_Review_Policy.pdf",
        "page": 3,
        "preview": (
            "360 degree feedback is collected from peers and subordinates from 1 to 20 February; employee "
            "self-assessment is submitted from 1 to 10 March; manager assessment runs from 11 to 20 March; "
            "calibration is held from 21 to 25 March; final ratings are locked from 26 to 31 March; increment "
            "and promotion letters are issued on 15 April."
        ),
    }
]

VARIANTS = {
    "variant_h_q09_correct_apr.csv": {"Q09": Q09_ANSWER},
    "variant_i_q09_correct_q15_short_refusal.csv": {"Q09": Q09_ANSWER, "Q15": Q15_ANSWER},
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
            assert not re.search(r"\*\*|(?:^|\n)\s*(?:[-*]|\d+\.)\s+", answer)
        else:
            assert row == baseline

    for question_id in PROTECTED_IDS:
        assert rows_by_id[question_id] == baseline_by_id[question_id]

    assert debug_by_id["Q09"]["sources"] == Q09_EVIDENCE
    assert all(
        marker in debug_by_id["Q09"]["clean_answer"]
        for marker in (
            "1 to 20 February",
            "1 to 10 March",
            "11 to 20 March",
            "21 to 25 March",
            "26 to 31 March",
            "15 April",
        )
    )


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
    assert baseline_debug_by_id["Q15"]["clean_answer"] == Q15_ANSWER

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
                    "blocked": question_id == "Q15",
                    "confidence": 1.0 if question_id == "Q09" else 0.0,
                    "critique_rating": None,
                    "refined": False,
                    "hardcoded_guardrail": question_id == "Q15",
                    "controlled_override": True,
                }
            )
            if question_id == "Q09":
                debug["sources"] = Q09_EVIDENCE

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
