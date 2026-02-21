from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

from document_ingest import extract_google_doc_text, extract_uploaded_text
from iep_store import IEPRecord, load_ieps
from llama_inference import load_llama_cpp_llm
from prompting import (
    PromptContext,
    build_adapt_system_prompt,
    build_adapt_user_prompt,
    build_chat_system_prompt,
    build_shared_context,
)

ROOT = Path("/Users/jrostan/HuggingFace")
DEFAULT_IEP_ROOT = ROOT / "data" / "ieps"
OUTPUT_DIR = ROOT / "outputs"


@lru_cache(maxsize=1)
def get_model():
    return load_llama_cpp_llm()


@lru_cache(maxsize=1)
def get_records() -> list[IEPRecord]:
    return load_ieps([DEFAULT_IEP_ROOT])


def _grade_sort_key(grade: str) -> tuple[int, str]:
    g = grade.strip().upper()
    if g.startswith("G") and g[1:].isdigit():
        return (0, int(g[1:]))
    return (1, g)


def grade_levels() -> list[str]:
    levels = {r.grade_level for r in get_records()}
    return sorted(levels, key=_grade_sort_key)


def labels_for_grade(grade: str) -> list[str]:
    return [r.label for r in get_records() if r.grade_level == grade]


def on_grade_change(grade: str) -> gr.CheckboxGroup:
    return gr.CheckboxGroup(choices=labels_for_grade(grade), value=[])


def _selected_records(labels: list[str]) -> list[IEPRecord]:
    by_label = {r.label: r for r in get_records()}
    return [by_label[label] for label in labels if label in by_label]


def _build_shared_context(selected_labels: list[str]) -> str:
    selected = _selected_records(selected_labels)
    if not selected:
        raise ValueError("Select at least one student IEP.")
    context = PromptContext(ieps=selected)
    return build_shared_context(context)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _chat_user_prompt_from_history(history: list[Any], message: str) -> str:
    if not history:
        return message

    lines: list[str] = []
    for turn in history:
        if isinstance(turn, dict):
            role = turn.get("role", "")
            text = _content_to_text(turn.get("content"))
            if text:
                if role == "user":
                    lines.append(f"Teacher: {text}")
                elif role == "assistant":
                    lines.append(f"Assistant: {text}")
            continue

        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, assistant_msg = turn
            if isinstance(user_msg, str) and user_msg.strip():
                lines.append(f"Teacher: {user_msg.strip()}")
            if isinstance(assistant_msg, str) and assistant_msg.strip():
                lines.append(f"Assistant: {assistant_msg.strip()}")

    transcript = "\n".join(lines)
    return (
        "Conversation history:\n"
        f"{transcript}\n\n"
        "Latest teacher question:\n"
        f"{message}\n\n"
        "Respond to the latest question while staying consistent with prior turns."
    )


def chat_turn(
    grade: str,
    selected_labels: list[str],
    history: list[Any],
    message: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, str]], str]:
    msg = (message or "").strip()
    if not msg:
        return history, ""

    shared_context = _build_shared_context(selected_labels)
    system_prompt = build_chat_system_prompt(shared_context)
    user_prompt = _chat_user_prompt_from_history(history, msg)

    model = get_model()
    answer = model.generate(
        mode="chat",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
    )

    new_history: list[dict[str, str]] = []
    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            text = _content_to_text(turn.get("content"))
            if role in {"user", "assistant"} and text:
                new_history.append({"role": role, "content": text})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, assistant_msg = turn
            if isinstance(user_msg, str) and user_msg.strip():
                new_history.append({"role": "user", "content": user_msg.strip()})
            if isinstance(assistant_msg, str) and assistant_msg.strip():
                new_history.append({"role": "assistant", "content": assistant_msg.strip()})

    new_history.append({"role": "user", "content": msg})
    new_history.append({"role": "assistant", "content": answer})
    return new_history, ""


def clear_chat() -> tuple[list[dict[str, str]], str]:
    return [], ""


def _extract_assessment_text(file_path: str | None, gdoc_url: str) -> str:
    uploaded_text = ""
    if file_path:
        path = Path(file_path)
        uploaded_text = extract_uploaded_text(path.name, path.read_bytes())

    gdoc_text = ""
    if gdoc_url.strip():
        gdoc_text = extract_google_doc_text(gdoc_url.strip())

    if uploaded_text and gdoc_text:
        return uploaded_text + "\n\n---\n\n" + gdoc_text
    if uploaded_text:
        return uploaded_text
    if gdoc_text:
        return gdoc_text
    return ""


def run_adapt(
    grade: str,
    selected_labels: list[str],
    file_path: str | None,
    gdoc_url: str,
    user_instruction: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, str | None]:
    _ = grade
    shared_context = _build_shared_context(selected_labels)
    assessment_text = _extract_assessment_text(file_path, gdoc_url)

    if not assessment_text.strip():
        raise ValueError("Upload an assessment file or provide a Google Doc URL.")

    system_prompt = build_adapt_system_prompt(shared_context)
    user_prompt = build_adapt_user_prompt(assessment_text, user_instruction)

    model = get_model()
    adapted = model.generate(
        mode="adapt",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"adapted_assessment_{ts}.md"
    out_path.write_text(adapted, encoding="utf-8")
    return adapted, str(out_path)


def build_ui() -> gr.Blocks:
    if not get_records():
        raise RuntimeError(f"No IEP markdown files found in {DEFAULT_IEP_ROOT}")

    levels = grade_levels()
    default_grade = levels[0]
    with gr.Blocks(title="IEP Local Assistant (Gradio)") as demo:
        gr.Markdown("# IEP Local Assistant")
        gr.Markdown("Local-first multi-turn chat and assessment adaptation using local GGUF models.")

        with gr.Row():
            grade = gr.Dropdown(
                label="Grade level",
                choices=levels,
                value=default_grade,
            )
            students = gr.CheckboxGroup(
                label="Students",
                choices=labels_for_grade(default_grade),
                value=[],
            )

        with gr.Tab("Chat with IEP (Multi-turn)"):
            chatbot = gr.Chatbot(height=460)
            with gr.Row():
                message = gr.Textbox(label="Message", lines=3, scale=5)
                with gr.Column(scale=1):
                    chat_max_tokens = gr.Slider(256, 4096, value=1800, step=64, label="Max new tokens")
                    chat_temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear chat")

        with gr.Tab("Adapt assessment"):
            file_input = gr.File(label="Upload assessment (.pdf, .doc, .docx)", file_count="single", type="filepath")
            gdoc_url = gr.Textbox(label="Google Doc URL", placeholder="https://docs.google.com/document/d/...", lines=1)
            instruction = gr.Textbox(
                label="Optional adaptation instructions",
                placeholder="Example: keep printable in one class period and include UDL supports.",
                lines=4,
            )
            with gr.Row():
                adapt_max_tokens = gr.Slider(512, 8192, value=3600, step=64, label="Max new tokens")
                adapt_temp = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Temperature")
            adapt_btn = gr.Button("Adapt now", variant="primary")
            adapt_md = gr.Markdown(label="Adapted assessment")
            adapt_file = gr.File(label="Download Markdown")

        grade.change(on_grade_change, inputs=[grade], outputs=[students])
        send_btn.click(
            chat_turn,
            inputs=[grade, students, chatbot, message, chat_max_tokens, chat_temp],
            outputs=[chatbot, message],
        )
        message.submit(
            chat_turn,
            inputs=[grade, students, chatbot, message, chat_max_tokens, chat_temp],
            outputs=[chatbot, message],
        )
        clear_btn.click(clear_chat, inputs=[], outputs=[chatbot, message])

        adapt_btn.click(
            run_adapt,
            inputs=[grade, students, file_input, gdoc_url, instruction, adapt_max_tokens, adapt_temp],
            outputs=[adapt_md, adapt_file],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[str(OUTPUT_DIR)],
    )


if __name__ == "__main__":
    main()
