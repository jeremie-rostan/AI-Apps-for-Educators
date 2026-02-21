from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

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
LEGACY_IEP_ROOT = ROOT / "MockIEPs"
OUTPUT_DIR = ROOT / "outputs"


@st.cache_resource
def get_model():
    return load_llama_cpp_llm()


def get_iep_records() -> list[IEPRecord]:
    roots = [DEFAULT_IEP_ROOT]
    return load_ieps(roots)


def _grade_sort_key(grade: str) -> tuple[int, str]:
    g = grade.strip().upper()
    if g.startswith("G") and g[1:].isdigit():
        return (0, int(g[1:]))
    return (1, g)


def _selected_records(records: list[IEPRecord], labels: list[str]) -> list[IEPRecord]:
    by_label = {r.label: r for r in records}
    return [by_label[label] for label in labels if label in by_label]


def _extract_assessment_text() -> str:
    st.subheader("Assessment Source")
    uploaded = st.file_uploader(
        "Upload assessment (.pdf, .doc, .docx)", type=["pdf", "doc", "docx"]
    )
    gdoc_url = st.text_input("Or paste Google Doc URL")

    uploaded_text = ""
    if uploaded is not None:
        uploaded_text = extract_uploaded_text(uploaded.name, uploaded.getvalue())

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


def _build_shared_context(selected: list[IEPRecord]) -> str:
    context = PromptContext(ieps=selected)
    return build_shared_context(context)


def main() -> None:
    st.set_page_config(page_title="IEP Local Assistant", layout="wide")
    st.title("IEP Local Assistant")
    st.caption("Local-first IEP chat and assessment adaptation (no cloud inference).")

    records = get_iep_records()
    if not records:
        st.error(
            f"No IEP markdown files found in {DEFAULT_IEP_ROOT}."
        )
        st.stop()

    st.sidebar.header("IEP Selection")
    st.sidebar.write(f"Found {len(records)} IEP file(s).")
    grade_levels = sorted({r.grade_level for r in records}, key=_grade_sort_key)
    selected_grade = st.sidebar.selectbox("Select grade level", options=grade_levels)
    grade_records = [r for r in records if r.grade_level == selected_grade]
    st.sidebar.write(f"{len(grade_records)} student IEP(s) in {selected_grade}.")
    selected_labels = st.sidebar.multiselect(
        "Select one or more students",
        options=[r.label for r in grade_records],
    )

    mode = st.radio("Mode", options=["Chat with IEP", "Adapt assessment"], horizontal=True)
    if not selected_labels:
        st.info("Select at least one IEP in the sidebar to continue.")
        st.stop()

    selected = _selected_records(records, selected_labels)
    shared_context = _build_shared_context(selected)

    with st.expander("Selected context preview"):
        st.text(shared_context[:4000] + ("..." if len(shared_context) > 4000 else ""))

    model = get_model()

    if mode == "Chat with IEP":
        st.subheader("Chat")
        user_question = st.text_area("Ask a question", height=120)
        if st.button("Run chat", type="primary"):
            if not user_question.strip():
                st.warning("Enter a question first.")
                st.stop()
            with st.spinner("Generating response locally..."):
                system_prompt = build_chat_system_prompt(shared_context)
                response = model.generate(
                    mode="chat",
                    system_prompt=system_prompt,
                    user_prompt=user_question.strip(),
                    max_new_tokens=1800,
                    temperature=0.2,
                )
            st.markdown("### Response")
            st.write(response)

    else:
        st.subheader("Adapt assessment")
        try:
            assessment_text = _extract_assessment_text()
        except Exception as err:  # noqa: BLE001
            st.error(str(err))
            st.stop()

        user_instruction = st.text_area(
            "Optional adaptation instructions",
            placeholder="Example: Keep this printable in one class period and include UDL supports.",
            height=120,
        )

        if st.button("Adapt now", type="primary"):
            if not assessment_text.strip():
                st.warning("Upload an assessment file or provide a Google Doc URL.")
                st.stop()
            with st.spinner("Generating adapted assessment locally..."):
                system_prompt = build_adapt_system_prompt(shared_context)
                user_prompt = build_adapt_user_prompt(assessment_text, user_instruction)
                adapted = model.generate(
                    mode="adapt",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=3600,
                    temperature=0.15,
                )

            st.markdown("### Adapted Assessment")
            st.markdown(adapted)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"adapted_assessment_{ts}.md"
            out_path.write_text(adapted, encoding="utf-8")

            st.download_button(
                label="Download as Markdown",
                data=adapted.encode("utf-8"),
                file_name=out_path.name,
                mime="text/markdown",
            )
            st.caption(f"Saved locally to {out_path}")


if __name__ == "__main__":
    main()
