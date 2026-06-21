from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from generate_competition_submission import REQUIRED_COLUMNS, extract_competition_questions, validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a controlled single-question variant from an existing scored Kaggle submission."
    )
    parser.add_argument("--baseline", required=True, help="Base scored CSV to preserve everywhere except replaced IDs.")
    parser.add_argument("--alternate", required=True, help="Alternate CSV to copy selected question answers from.")
    parser.add_argument("--replace", required=True, help="Comma-separated question IDs to replace, e.g. Q03 or Q03,Q04.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if list(rows[0].keys()) != REQUIRED_COLUMNS:
        raise ValueError(f"{path} does not match the required Kaggle columns.")
    return rows


def load_sources(path: Path) -> list[dict]:
    return json.loads(path.with_suffix(".sources.json").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    alternate_path = Path(args.alternate)
    output_path = Path(args.output)
    replace_ids = {item.strip().upper() for item in args.replace.split(",") if item.strip()}
    expected_ids = [f"Q{i:02d}" for i in range(1, 16)]

    if not replace_ids:
        raise ValueError("At least one question ID must be supplied in --replace.")
    if not replace_ids.issubset(set(expected_ids)):
        raise ValueError("Replacement IDs must stay within Q01-Q15.")

    baseline_rows = load_csv(baseline_path)
    alternate_rows = load_csv(alternate_path)
    baseline_sources = load_sources(baseline_path)
    alternate_sources = load_sources(alternate_path)

    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    alternate_by_id = {row["question_id"]: row for row in alternate_rows}
    baseline_sources_by_id = {row["question_id"]: row for row in baseline_sources}
    alternate_sources_by_id = {row["question_id"]: row for row in alternate_sources}

    output_rows: list[dict[str, str]] = []
    output_sources: list[dict] = []
    for question_id in expected_ids:
        use_alternate = question_id in replace_ids
        output_rows.append((alternate_by_id if use_alternate else baseline_by_id)[question_id])
        output_sources.append((alternate_sources_by_id if use_alternate else baseline_sources_by_id)[question_id])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    output_path.with_suffix(".sources.json").write_text(
        json.dumps(output_sources, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    validate_submission(output_path)

    fernet, _ = extract_competition_questions(str(Path(__file__).resolve().parent / "competition" / "Starter_Notebook.ipynb"))
    changed_ids = [
        question_id
        for question_id in expected_ids
        if baseline_by_id[question_id]["answer_enc"] != {row["question_id"]: row for row in output_rows}[question_id]["answer_enc"]
    ]
    changed_answers = {
        question_id: fernet.decrypt({row["question_id"]: row for row in output_rows}[question_id]["answer_enc"].encode("ascii")).decode("utf-8")
        for question_id in changed_ids
    }

    print(
        json.dumps(
            {
                "filename": str(output_path),
                "base_file": str(baseline_path),
                "alternate_file": str(alternate_path),
                "changed_questions": changed_ids,
                "changed_count": len(changed_ids),
                "validation_status": "passed",
                "decrypted_changed_answers": changed_answers,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
