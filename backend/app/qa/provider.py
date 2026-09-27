from __future__ import annotations

import asyncio
import json
import re

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings
from app.schemas import QAEvidence


class AnswerProposal(BaseModel):
    answer: str = Field(min_length=1)
    citation_numbers: list[int] = Field(default_factory=list)
    insufficient_evidence: bool = False


class DeepSeekChatProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.deepseek_api_key
        self.base_url = self.settings.deepseek_base_url.rstrip("/")
        self.model = self.settings.deepseek_chat_model
        self.timeout = self.settings.table_repair_timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    async def answer(
        self, question: str, evidence: list[QAEvidence]
    ) -> AnswerProposal:
        if not self.available:
            raise ValueError("DEEPSEEK_API_KEY is required for answer generation.")
        evidence_text = "\n\n".join(
            (
                f"[E{item.evidence_number}]\n"
                f"Filename: {item.filename}\n"
                f"Page: {item.page}\n"
                f"Report fiscal year: {item.fiscal_year or 'unknown'}\n"
                f"Document type: {item.document_type or 'unknown'}\n"
                f"Table: {item.table_title or 'not applicable'}\n"
                f"Row: {item.row_label or 'not applicable'}\n"
                f"Chunk ID: {item.chunk_id}\n"
                f"Content:\n{item.content}"
            )
            for item in evidence
        ) or "No evidence was retrieved."
        system_prompt = """You answer questions about financial reports using only the supplied evidence chunks.
Rules:
1. Never use outside knowledge and never claim access to a PDF or any source beyond the evidence.
2. Answer in the same language as the user's question.
3. Preserve exact values, units, dates, currencies, fiscal periods, and distinctions such as FY versus H1.
4. You may perform transparent arithmetic only when every input value is present in the evidence.
5. If the evidence does not fully support the requested answer, clearly say that the evidence is insufficient and set insufficient_evidence to true.
6. Cite supporting evidence inline as [E1], [E2], etc. Use the minimum exact row-level evidence needed for each claim; do not cite summaries or related chunks when an exact row is available, and do not cite evidence that does not support the statement.
7. Treat a report's fiscal year and a row/category period as distinct fields. Use the period requested by the question; do not relabel a historical row as the report fiscal year.
8. Values in different reports are not a conflict merely because they differ. Answer each requested report or period independently when each value is explicit.
9. Return JSON only with keys: answer, citation_numbers, insufficient_evidence. citation_numbers contains the cited evidence numbers as integers."""
        user_prompt = f"Question:\n{question}\n\nFinal reranked evidence only:\n{evidence_text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        last_error: Exception | None = None
        valid_numbers = {item.evidence_number for item in evidence}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(3):
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    if response.status_code in {429, 500, 502, 503, 504}:
                        last_error = RuntimeError(
                            f"DeepSeek request failed ({response.status_code}): {response.text[:500]}"
                        )
                    elif response.is_error:
                        raise RuntimeError(
                            f"DeepSeek request failed ({response.status_code}): {response.text[:500]}"
                        )
                    else:
                        content = ""
                        repair_reason = ""
                        try:
                            content = response.json()["choices"][0]["message"]["content"]
                            proposal = AnswerProposal.model_validate(json.loads(content))
                        except (KeyError, TypeError, json.JSONDecodeError, ValidationError):
                            repair_reason = (
                                "Return one valid JSON object with exactly answer, "
                                "citation_numbers, and insufficient_evidence."
                            )
                            last_error = RuntimeError("DeepSeek returned an invalid structured answer.")
                        else:
                            inline_numbers = [
                                int(value)
                                for value in re.findall(r"\[E(\d+)\]", proposal.answer)
                            ]
                            if any(number not in valid_numbers for number in inline_numbers):
                                repair_reason = (
                                    "Use only evidence citation numbers supplied in the prompt."
                                )
                                last_error = RuntimeError(
                                    "DeepSeek answer cited evidence outside the supplied context."
                                )
                            else:
                                proposal.citation_numbers = list(
                                    dict.fromkeys(
                                        number
                                        for number in proposal.citation_numbers
                                        if number in valid_numbers
                                    )
                                )
                                if not proposal.citation_numbers:
                                    proposal.citation_numbers = list(
                                        dict.fromkeys(
                                            number
                                            for number in inline_numbers
                                            if number in valid_numbers
                                        )
                                    )
                                if proposal.insufficient_evidence or proposal.citation_numbers:
                                    return proposal
                                repair_reason = (
                                    "A sufficient answer must cite at least one supplied evidence number."
                                )
                                last_error = RuntimeError(
                                    "DeepSeek answer did not cite any supplied evidence."
                                )
                        if repair_reason and attempt < 2:
                            messages.extend(
                                [
                                    {"role": "assistant", "content": content[:4000]},
                                    {
                                        "role": "user",
                                        "content": f"Repair the response. {repair_reason}",
                                    },
                                ]
                            )
                except httpx.HTTPError as exc:
                    last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"DeepSeek answer failed after 3 attempts: {last_error}")
