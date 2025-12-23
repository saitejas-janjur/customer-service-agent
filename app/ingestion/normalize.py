"""
Text normalization stage.

Responsibilities:
- Clean up whitespace, null bytes, and repeated blank lines.
- Ensure stable metadata fields.
- Keep logic deterministic and testable.

Rule: normalization is a pure function (Document -> Document).
"""

from __future__ import annotations

import re
from langchain_core.documents import Document


_NULL_BYTES = re.compile(r"\x00+")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_documents(docs: list[Document]) -> list[Document]:
    """
    Normalize a list of Documents.
    """
    return [normalize_document(d) for d in docs]


def normalize_document(doc: Document) -> Document:
    """
    Normalize a single Document.

    - Replaces Windows newlines with '\n'
    - Removes null bytes
    - Trims trailing spaces
    - Collapses excessive blank lines

    Returns:
        New Document with cleaned text.
    """
    text = doc.page_content or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _NULL_BYTES.sub("", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _MANY_BLANK_LINES.sub("\n\n", text)
    text = text.strip()

    metadata = dict(doc.metadata or {})
    # Ensure we always have a "source" for citations and debugging.
    metadata.setdefault("source", "unknown")

    return Document(page_content=text, metadata=metadata)