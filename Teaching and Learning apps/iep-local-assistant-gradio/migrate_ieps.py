from __future__ import annotations

import shutil
from pathlib import Path

from iep_store import parse_iep_markdown


ROOT = Path("/Users/jrostan/HuggingFace")
SOURCE_DIR = ROOT / "MockIEPs"
TARGET_DIR = ROOT / "data" / "ieps"


def normalize_grade(grade_level: str) -> str:
    value = grade_level.strip().upper().replace(" ", "")
    if value.startswith("GRADE"):
        value = value.replace("GRADE", "G", 1)
    if value.isdigit():
        value = f"G{value}"
    if not value.startswith("G"):
        return "UNKNOWN"
    number = value[1:]
    if not number.isdigit():
        return "UNKNOWN"
    grade_num = int(number)
    if grade_num < 6 or grade_num > 12:
        return "UNKNOWN"
    return f"G{grade_num}"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(SOURCE_DIR.glob("*.md")):
        record = parse_iep_markdown(src)
        grade = normalize_grade(record.grade_level)
        grade_dir = TARGET_DIR / grade
        grade_dir.mkdir(parents=True, exist_ok=True)
        destination = grade_dir / src.name
        shutil.copy2(src, destination)
        print(f"Copied {src.name} -> {destination}")


if __name__ == "__main__":
    main()
