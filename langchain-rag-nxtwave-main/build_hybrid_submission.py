from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from generate_competition_submission import (
    REQUIRED_COLUMNS,
    print_submission_validation_report,
    print_validation_summary,
    validate_competition_response,
    validate_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validated hybrid from generated RAG candidates.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--alternate", required=True)
    parser.add_argument("--replace", required=True, help="Comma-separated question IDs to take from alternate.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trust-baseline",
        action="store_true",
        help="Strictly validate replacement rows while preserving unchanged rows from a known-scored baseline.",
    )
    return parser.parse_args()


def load_candidate(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["question_id"]: row for row in csv.DictReader(handle)}
    debug_path = path.with_suffix(".sources.json")
    debug_rows = {
        row["question_id"]: row
        for row in json.loads(debug_path.read_text(encoding="utf-8"))
    }
    if rows.keys() != debug_rows.keys():
        raise ValueError("CSV and source log question IDs do not match for %s." % path)
    return rows, debug_rows


def main() -> None:
    args = parse_args()
    replace_ids = {item.strip().upper() for item in args.replace.split(",") if item.strip()}
    expected_ids = ["Q%02d" % index for index in range(1, 16)]
    if not replace_ids.issubset(set(expected_ids)):
        raise ValueError("Replacement IDs must be within Q01-Q15.")

    baseline_rows, baseline_debug = load_candidate(Path(args.baseline))
    alternate_rows, alternate_debug = load_candidate(Path(args.alternate))
    rows = []
    debug_rows = []
    for index, question_id in enumerate(expected_ids, start=1):
        use_alternate = question_id in replace_ids
        row = alternate_rows[question_id] if use_alternate else baseline_rows[question_id]
        debug = alternate_debug[question_id] if use_alternate else baseline_debug[question_id]
        response = type(
            "CandidateResponse",
            (),
            {
                "answer": debug["clean_answer"],
                "blocked": debug["blocked"],
                "critique_rating": debug.get("critique_rating"),
            },
        )()
        if use_alternate or not args.trust_baseline:
            validate_competition_response(question_id, index, response, enforce_word_limit=False)
        rows.append(row)
        debug_rows.append({**debug, "hybrid_source": "alternate" if use_alternate else "baseline"})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    output_path.with_suffix(".sources.json").write_text(
        json.dumps(debug_rows, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print_submission_validation_report(rows, debug_rows)
    validate_submission(output_path)
    print("Replaced from alternate: %s" % ", ".join(sorted(replace_ids)))
    print_validation_summary(debug_rows, output_path)


if __name__ == "__main__":
    main()
