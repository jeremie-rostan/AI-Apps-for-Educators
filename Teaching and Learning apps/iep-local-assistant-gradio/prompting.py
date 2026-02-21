from __future__ import annotations

from dataclasses import dataclass

from iep_store import IEPRecord


@dataclass(frozen=True)
class PromptContext:
    ieps: list[IEPRecord]


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
        "## Selected IEP Records (Authoritative)\n"
        f"{iep_block}"
    )


def build_chat_system_prompt(shared_context: str) -> str:
    return (
        "You are an IEP specialist supporting teachers. "
        "Use ONLY the context below. Do not invent student details. "
        "If context is missing, say what is missing. "
        "When the teacher specifies a setting or subject (for example: PE, art, science lab), "
        "anchor recommendations to that exact setting unless the teacher explicitly asks for broader classroom guidance. "
        "Do not drift to generic classroom advice when a specific setting is requested.\n\n"
        f"{shared_context}"
    )


def build_adapt_system_prompt(shared_context: str) -> str:
    return (
        "You redesign assessments, units or lessons for selected students using UDL (Universal Design for Learning). "
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
        "- Keep academic rigor and original learning goals\n"
        "- Mark allowed supports clearly\n\n"
        f"Additional teacher instruction:\n{extra}\n\n"
        "Source assessment, unit or lesson text:\n"
        f"{assessment_text}"
    )
