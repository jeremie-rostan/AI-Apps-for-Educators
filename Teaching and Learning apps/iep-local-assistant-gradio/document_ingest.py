from __future__ import annotations

import io
import re
import subprocess
import tempfile
from pathlib import Path

import requests
from docx import Document
from pypdf import PdfReader


def _read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def _read_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs).strip()


def _read_doc_via_textutil(file_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError("Unable to parse .doc file with textutil.")
        return result.stdout.strip()
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_uploaded_text(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _read_pdf(file_bytes)
    if lower.endswith(".docx"):
        return _read_docx(file_bytes)
    if lower.endswith(".doc"):
        return _read_doc_via_textutil(file_bytes, ".doc")
    raise ValueError("Unsupported file type. Use .pdf, .doc, or .docx.")


def _extract_google_doc_id(url: str) -> str | None:
    match = re.search(r"docs\.google\.com/document/d/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def extract_google_doc_text(url: str, timeout_sec: int = 20) -> str:
    doc_id = _extract_google_doc_id(url)
    if not doc_id:
        raise ValueError("Not a valid Google Doc URL.")

    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(export_url, timeout=timeout_sec)
    if response.status_code != 200:
        raise ValueError(
            "Could not fetch Google Doc text. Ensure the doc is shared or upload a local file."
        )
    text = response.text.strip()
    if not text:
        raise ValueError("Google Doc text export returned empty content.")
    return text
