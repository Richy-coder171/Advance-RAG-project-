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

Q11_ANSWER = (
    "The provided Zyro Dynamics HR policy documents do not include an external job application, recruitment, "
    "or hiring process policy. Therefore, the application and hiring process is not specified in the supplied "
    "documents."
)
Q13_ANSWER = (
    "Company revenue, financial performance, and business results are not covered in the supplied Zyro "
    "Dynamics HR policy documents."
)
Q14_ANSWER = (
    "AcruxCRM product features and Salesforce comparisons are not covered in the supplied Zyro Dynamics HR "
    "policy documents."
)
Q15_ANSWER = (
    "Zyro Dynamics Earned Leave is 15 days after one year of service and accrues at 1.25 days per month. Up "
    "to 45 days can be carried forward at the end of the financial year, and excess Earned Leave is "
    "automatically encashed and credited in April payroll. The supplied documents do not contain Zoho or "
    "Freshworks leave policies, so no comparison can be made."
)

EVIDENCE = {
    "Q11": [],
    "Q13": [],
    "Q14": [],
    "Q15": [
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 2,
            "preview": "Earned Leave is 15 days after one year of service and accrues at 1.25 days per month.",
        },
        {
            "source_file": "02_Leave_Policy.pdf",
            "page": 3,
            "preview": (
                "A maximum of 45 days of Earned Leave may be carried forward at the end of the financial year; "
                "excess is automatically encashed and credited in April payroll."
            ),
        },
    ],
}

VARIANTS = {
    "variant_af_ae_plus_q15_partial_leave.csv": {"Q15": Q15_ANSWER},
    "variant_ag_ae_plus_q11_clean_refusal.csv": {"Q11": Q11_ANSWER},
    "variant_ah_ae_plus_q13_q14_short_refusals.csv": {"Q13": Q13_ANSWER, "Q14": Q14_ANSWER},
    "variant_ai_combined_refusal_optimized.csv": {
        "Q11": Q11_ANSWER,
        "Q13": Q13_ANSWER,
        "Q14": Q14_ANSWER,
        "Q15": Q15_ANSWER,
    },
}

PROTECTED_IDS = {f"Q{i:02d}" for i in range(1, 11)} | {"Q12"}


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
            assert question_id not in PROTECTED_IDS
            assert answer == expected_answers[question_id]
            assert debug_by_id[question_id]["clean_answer"] == answer
            assert debug_by_id[question_id]["answer_with_sources"] == answer
            assert debug_by_id[question_id]["sources"] == EVIDENCE[question_id]
            assert debug_by_id[question_id].get("unsupported_claims", []) == []
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
                    "blocked": question_id != "Q15",
                    "confidence": 1.0 if question_id == "Q15" else 0.0,
                    "critique_rating": None,
                    "refined": False,
                    "sources": EVIDENCE[question_id],
                    "hardcoded_guardrail": question_id != "Q15",
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
