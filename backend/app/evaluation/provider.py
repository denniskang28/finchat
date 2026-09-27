from __future__ import annotations

import asyncio
import json

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings


class GeneratedQuestion(BaseModel):
    source_index: int = Field(ge=0)
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    language: str
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)


class GeneratedQuestionBatch(BaseModel):
    cases: list[GeneratedQuestion]


class JudgeResult(BaseModel):
    correctness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)
    language_match: bool
    failure_reasons: list[str] = Field(default_factory=list)
    rationale: str = ""


class AlibabaEvaluationProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.dashscope_api_key
        self.base_url = self.settings.alibaba_base_url.rstrip("/")
        self.model = self.settings.alibaba_evaluation_model
        self.timeout = self.settings.table_repair_timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    async def _json_completion(self, system: str, user: str) -> dict:
        if not self.available:
            raise ValueError("DASHSCOPE_API_KEY is required for evaluation generation and judging.")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 4000,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_error: Exception | None = None
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
                            f"Alibaba evaluation request failed ({response.status_code}): "
                            f"{response.text[:500]}"
                        )
                    elif response.is_error:
                        raise RuntimeError(
                            f"Alibaba evaluation request failed ({response.status_code}): "
                            f"{response.text[:500]}"
                        )
                    else:
                        content = response.json()["choices"][0]["message"]["content"]
                        return json.loads(content)
                except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Alibaba evaluation request failed after 3 attempts: {last_error}")

    async def generate_questions(
        self,
        sources: list[dict],
        *,
        language: str,
        difficulty: str,
    ) -> list[GeneratedQuestion]:
        system = """Generate grounded financial-report evaluation questions from the supplied evidence.
Return JSON only: {"cases":[{"source_index":0,"question":"...","expected_answer":"...","language":"en","difficulty":"medium","tags":["table","numeric"]}]}.
Rules:
1. Generate exactly one question for each source and use only that source.
2. The expected answer must be directly and completely supported by the source.
3. Preserve exact numbers, currency, units, dates, and FY/H1 distinctions.
4. Do not mention source index, chunk, page, evidence, or retrieval in the question.
5. Prefer useful finance questions over trivial document-location questions.
6. Do not invent or calculate values unless the arithmetic and all inputs are explicit."""
        payload = {
            "requested_language": language,
            "requested_difficulty": difficulty,
            "sources": sources,
        }
        try:
            parsed = GeneratedQuestionBatch.model_validate(
                await self._json_completion(system, json.dumps(payload, ensure_ascii=False))
            )
        except ValidationError as exc:
            raise RuntimeError("Qwen returned invalid generated evaluation cases.") from exc
        return parsed.cases

    async def judge(
        self,
        *,
        question: str,
        expected_answer: str,
        expected_evidence: list[dict],
        actual_answer: str,
        citations: list[dict],
    ) -> JudgeResult:
        system = """You are a strict evaluator of financial-report QA.
Use only the supplied expected answer/evidence to assess the actual answer.
Exact values, currencies, units, dates, and FY/H1 distinctions are mandatory.
Do not reward unsupported statements. Return JSON only with correctness, completeness, faithfulness (0 to 1), language_match, failure_reasons, and a concise rationale."""
        payload = {
            "question": question,
            "expected_answer": expected_answer,
            "expected_evidence": expected_evidence,
            "actual_answer": actual_answer,
            "actual_citations": citations,
        }
        try:
            return JudgeResult.model_validate(
                await self._json_completion(system, json.dumps(payload, ensure_ascii=False))
            )
        except ValidationError as exc:
            raise RuntimeError("Qwen returned an invalid evaluation judgment.") from exc
