from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


OFFICIAL_CORPUS_SHA256: Dict[str, str] = {
    "00_Company_Profile.pdf": "BE4CE9C10C3D4068606EFB9B880F68DD1D2F390207E4EC2BB0074AE0CE45BD94",
    "01_Employee_Handbook.pdf": "6FACFB851D1072A2E80742A82B0714BC08F93CEDB82B28E30C39A81B6653B787",
    "02_Leave_Policy.pdf": "91C0ECFED798EBED90D08A089BAE1BD9A1831F9312253D64E5997AF52A1835E1",
    "03_Work_From_Home_Policy.pdf": "FD69ACDD5D43220B898AC1860E084F1A8C14C156EC518D38A0D2657713C49DDB",
    "04_Code_of_Conduct.pdf": "FAB0291873C79060F673702CAE8B90109071961F9EDE7190F82CDCB3B57B12D2",
    "05_Performance_Review_Policy.pdf": "2A4DF520B5C9313EFF0B5C140078772D36147EA024480DB3FDEC6BD04C048D1F",
    "06_Compensation_and_Benefits_Policy.pdf": "7DD59132A7DD3E524F4FA99B0597D9F3A733376629460310BF9EEC419E97EF3D",
    "07_IT_and_Data_Security_Policy.pdf": "35E886D81888B66A53B1AA2FBA6989FF9631AF57551EB6674E9A2B0CE12FBB55",
    "08_Prevention_of_Sexual_Harassment_Policy.pdf": "CF3AB02393E98CB949C21A46508BB5306EAE74F849B18B93D5861BD9C9FFE7D9",
    "09_Onboarding_and_Separation_Policy.pdf": "31F9E951C33F4161062B23BF1325BFA701FE5934D810D860520C61D25330E4CC",
    "10_Travel_and_Expense_Policy.pdf": "986C139C6298A66AEFE257DE58FA747F83EF2FDE4B56FCE29B0CAF8E6178FE22",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_official_corpus(docs_path: str) -> None:
    root = Path(docs_path)
    if not root.exists():
        raise ValueError("Official HR corpus folder not found: %s" % root)

    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    expected_files = set(OFFICIAL_CORPUS_SHA256)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            "Official HR corpus must contain only the exact 11 competition PDFs. "
            "Missing: %s. Unexpected: %s." % (missing or "none", unexpected or "none")
        )

    mismatched = [
        name
        for name, expected_hash in OFFICIAL_CORPUS_SHA256.items()
        if file_sha256(root / name) != expected_hash
    ]
    if mismatched:
        raise ValueError("Official HR corpus file hash mismatch: %s." % ", ".join(sorted(mismatched)))
