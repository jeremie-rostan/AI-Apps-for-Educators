from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from iep_store import IEPRecord


@dataclass(frozen=True)
class PromptContext:
    ieps: list[IEPRecord]
    isp_way_text: str


def load_isp_way(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_shared_context(context: PromptContext) -> str:
    iep_sections = []
    for iep in context.ieps:
        iep_sections.append(
            "\n".join(
                [
                    f"### IEP Record: {iep.student_id}",
                    f"- Grade Level: {iep.grade_level}",
                    f"- Diagnosis: {iep.diagnosis}",
                    f"- Case Manager: {iep.case_manager}",
                    "",
                    iep.content,
                ]
            )
        )
    iep_block = "\n\n".join(iep_sections)
    return (
        "## ISP Way (Required Policy Context)\n"
        f"{context.isp_way_text}\n\n"
        "## Selected IEP Records (Authoritative)\n"
        f"{iep_block}"
    )


def build_chat_system_prompt(shared_context: str) -> str:
    return (
        "You are an IEP specialist at ISP (International School of Panama). "
        "Use ONLY the context below. Do not invent student details. "
        "If context is missing, say what is missing.\n\n"
        f"{shared_context}"
    )


def build_adapt_system_prompt(shared_context: str) -> str:
    return (
        "You redesign assessments, units or lessons for selected students using UDL (Universal Design for Learning) and ISP Way guidance. "
        "Keep standards-aligned rigor, preserve core intent, and embed access supports. "
        "Return markdown only.\n\n"
        f"{shared_context}"
    )


def build_adapt_user_prompt(assessment_text: str, user_instruction: str) -> str:
    extra = user_instruction.strip()
    if not extra:
        extra = "No additional instructions."
    return (
        "Task: Adapt the assessment, unit or lesson below for the selected IEP students.\n"
        "Output format requirements:\n"
        "- Markdown only\n"
        "- Include title, directions, sections, and checklist supports\n"
        "- Include adaptations relevant to the IEP, including UDL strategies that can benefits all students.\n"
        "- Include a section explaining alignment with the ISP Way.\n"
        "- Keep academic rigor and original learning goals\n"
        "- Mark allowed supports clearly\n\n"
        f"Additional teacher instruction:\n{extra}\n\n"
        "Source assessment, unit or lesson text:\n"
        f"{assessment_text}"
    )
