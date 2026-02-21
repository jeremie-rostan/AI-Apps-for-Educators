from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class IEPRecord:
    path: Path
    student_id: str
    grade_level: str
    diagnosis: str
    case_manager: str
    content: str

    @property
    def label(self) -> str:
        return (
            f"{self.grade_level} | {pseudonym_for_student_id(self.student_id)}"
            f" | {self.student_id} | {self.diagnosis}"
        )


FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Riley", "Casey", "Morgan", "Avery", "Cameron",
    "Skyler", "Parker", "Reese", "Quinn", "Rowan", "Sage", "Emerson", "Harper",
]

LAST_NAMES = [
    "Brooks", "Hayes", "Perry", "Sawyer", "Bennett", "Reed", "Bailey", "Foster",
    "Shaw", "Miller", "Turner", "Parker", "Lane", "Sullivan", "Carter", "Hayden",
]


def pseudonym_for_student_id(student_id: str) -> str:
    digest = hashlib.sha256(student_id.encode("utf-8")).digest()
    first = FIRST_NAMES[digest[0] % len(FIRST_NAMES)]
    last = LAST_NAMES[digest[1] % len(LAST_NAMES)]
    return f"{first} {last}"


FIELD_PATTERNS = {
    "student_id": re.compile(r"\*\*Student ID:\*\*\s*(.+)"),
    "grade_level": re.compile(r"\*\*Grade Level:\*\*\s*(.+)"),
    "diagnosis": re.compile(r"\*\*Diagnosis:\*\*\s*(.+)"),
    "case_manager": re.compile(r"\*\*Case Manager:\*\*\s*(.+)"),
}


def _extract_field(pattern: re.Pattern[str], text: str, fallback: str) -> str:
    match = pattern.search(text)
    if not match:
        return fallback
    value = match.group(1).strip()
    return value if value else fallback


def parse_iep_markdown(path: Path) -> IEPRecord:
    content = path.read_text(encoding="utf-8")
    student_id = _extract_field(FIELD_PATTERNS["student_id"], content, "UNKNOWN")
    grade_level = _extract_field(FIELD_PATTERNS["grade_level"], content, "UNKNOWN")
    diagnosis = _extract_field(FIELD_PATTERNS["diagnosis"], content, "UNKNOWN")
    case_manager = _extract_field(FIELD_PATTERNS["case_manager"], content, "UNKNOWN")
    return IEPRecord(
        path=path,
        student_id=student_id,
        grade_level=grade_level,
        diagnosis=diagnosis,
        case_manager=case_manager,
        content=content,
    )


def discover_iep_files(roots: Iterable[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(path)
    return discovered


def load_ieps(roots: Iterable[Path]) -> list[IEPRecord]:
    records = [parse_iep_markdown(path) for path in discover_iep_files(roots)]
    return sorted(records, key=lambda r: (r.grade_level, r.student_id, str(r.path)))
