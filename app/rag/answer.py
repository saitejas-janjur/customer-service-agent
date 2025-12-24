"""
RAG answer generation (grounded + cited).

Inputs:
- user question
- retrieved chunks (Documents + citations)

Outputs:
- answer text that cites sources like [1], [2]
- sources list for UI/debugging

Key safety rule:
- If context is insufficient, say you don't know and ask clarifying questions.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.retrieval.hybrid import RetrievedChunk


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[str]


class RAGAnswerer:
    def __init__(self, settings: Settings) -> None:
        self._llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.response_model,
            temperature=0.2,
        )

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> AnswerResult:
        context, sources = _format_context(chunks)

        prompt = "\n".join(
            [
                "You are a customer service assistant.",
                "Answer the question using ONLY the provided context.",
                "",
                "CITATION RULES:",
                "- When you use a fact from a snippet, cite it with [n].",
                "- If the answer is not in the context, say you don't have "
                "enough information and ask a clarifying question.",
                "- Do not invent policies, numbers, timelines, or steps.",
                "",
                "CONTEXT SNIPPETS:",
                context,
                "",
                f"QUESTION: {question}",
                "",
                "Write a helpful, concise answer with citations.",
            ]
        )

        msg = self._llm.invoke(prompt)
        text = (msg.content or "").strip()

        return AnswerResult(answer=text, sources=sources)


def _format_context(chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
    lines: list[str] = []
    sources: list[str] = []

    for i, r in enumerate(chunks, start=1):
        citation = str(r.doc.metadata.get("citation", "unknown"))
        sources.append(citation)

        snippet = (r.doc.page_content or "").strip()
        snippet = snippet[:1200]  # keep prompt bounded

        lines.append(f"[{i}] SOURCE: {citation}")
        lines.append(snippet)
        lines.append("")

    if not lines:
        return "[no context retrieved]", []

    return "\n".join(lines).strip(), sources