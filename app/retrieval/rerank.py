"""
Optional LLM reranker.

This is *optional* because it costs tokens/latency. It can improve quality
when vector + BM25 produce a noisy top-N set.

Implementation approach:
- Provide the query and short snippets with ids.
- Ask the model to output a JSON list of ids in best-to-worst order.
- Validate and reorder deterministically.

If reranking fails, we fall back to the original order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import Settings


class _RerankResponse(BaseModel):
    ranked_ids: list[int] = Field(
        ...,
        description="List of candidate ids in best-to-worst order",
    )


@dataclass(frozen=True)
class RerankCandidate:
    candidate_id: int
    doc: Document
    score: float


class LLMReranker:
    def __init__(self, settings: Settings) -> None:
        self._llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.reasoning_model,
            temperature=0.0,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankCandidate]:
        if not candidates:
            return []

        prompt = self._build_prompt(query, candidates)
        msg = self._llm.invoke(prompt)
        text = (msg.content or "").strip()

        try:
            data = json.loads(text)
            parsed = _RerankResponse.model_validate(data)
        except Exception:
            # Fall back if the model didn't follow the contract.
            return list(candidates)

        id_to_candidate = {c.candidate_id: c for c in candidates}
        out: list[RerankCandidate] = []
        for cid in parsed.ranked_ids:
            if cid in id_to_candidate:
                out.append(id_to_candidate[cid])

        # Append any missing candidates in original order.
        missing = [c for c in candidates if c.candidate_id not in parsed.ranked_ids]
        out.extend(missing)

        return out

    def _build_prompt(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> str:
        lines = [
            "You are reranking retrieval results for a customer-support QA system.",
            "Return ONLY valid JSON with the following schema:",
            '{"ranked_ids":[0,1,2]}',
            "",
            f"Query: {query}",
            "",
            "Candidates (id, citation, snippet):",
        ]

        for c in candidates:
            citation = str(c.doc.metadata.get("citation", "unknown"))
            snippet = (c.doc.page_content or "").strip().replace("\n", " ")
            snippet = snippet[:280]
            lines.append(f"- id={c.candidate_id} | {citation} | {snippet}")

        lines.append("")
        lines.append("Rules:")
        lines.append("- Rank by relevance to the query.")
        lines.append("- Prefer grounded policy text over vague statements.")
        lines.append("- Output ONLY JSON, no extra text.")

        return "\n".join(lines)