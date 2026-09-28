from __future__ import annotations

import asyncio
import json

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings


GENERATION_PROMPT_VERSION = "evidence-bundle-v2"


class GeneratedCalculation(BaseModel):
    expression: str = Field(min_length=1)
    result: str = Field(min_length=1)
    unit: str | None = None


class GeneratedQuestion(BaseModel):
    bundle_index: int = Field(ge=0)
    scenario_type: str
    source_ids_used: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    language: str
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)
    calculations: list[GeneratedCalculation] = Field(default_factory=list)


class GeneratedQuestionBatch(BaseModel):
    cases: list[GeneratedQuestion]


class JudgeResult(BaseModel):
    correctness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)
    language_match: bool
    failure_reasons: list[str] = Field(default_factory=list)
    rationale: str = ""


class OpenAICompatibleEvaluationProvider:
    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    async def _json_completion(self, system: str, user: str) -> dict:
        if not self.available:
            raise ValueError(f"{self.name} API key is required for evaluation.")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 5000,
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
                            f"{self.name} evaluation request failed ({response.status_code}): "
                            f"{response.text[:500]}"
                        )
                    elif response.is_error:
                        raise RuntimeError(
                            f"{self.name} evaluation request failed ({response.status_code}): "
                            f"{response.text[:500]}"
                        )
                    else:
                        content = response.json()["choices"][0]["message"]["content"]
                        return json.loads(content)
                except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"{self.name} evaluation request failed after 3 attempts: {last_error}")

    async def generate_questions(
        self,
        bundles: list[dict],
        *,
        language: str,
        difficulty: str,
        allow_calculations: bool,
        question_style: str,
    ) -> list[GeneratedQuestion]:
        question_style_rule = (
            "7. Write questions as a finance user would naturally ask them. Do not "
            "mention report titles, presentation names, table titles, chart titles, "
            "pages, chunks, evidence, or retrieval. Keep business metrics, entities, "
            "and naturally useful periods, but do not add locator details solely to "
            "disambiguate sources."
            if question_style == "USER_REALISTIC"
            else "7. Include each exact table_title in the question when present. This "
            "is a diagnostic retrieval test, so explicit table and chart locators are required."
        )
        system = """Generate grounded financial-report evaluation questions from evidence bundles.
Return JSON only:
{"cases":[{"bundle_index":0,"scenario_type":"CROSS_YEAR","source_ids_used":["S1","S2"],"question":"...","expected_answer":"...","language":"en","difficulty":"medium","tags":["cross-year","numeric"],"calculations":[{"expression":"6.2 - 5.7","result":"0.5","unit":"USD billion"}]}]}.
Rules:
1. Generate exactly one question for each bundle and preserve its bundle_index and scenario_type.
2. SINGLE_DOCUMENT uses its one source and must name that report's fiscal year or fiscal period in the question. CROSS_YEAR and CROSS_DOCUMENT must require facts from every source in that bundle.
3. The expected answer must be directly and completely supported by the sources. Preserve exact numbers, currency, units, dates, FY/H1, and percentage-point distinctions.
3a. For a numeric multi-source question, state the answer-bearing value from every source even when the values are equal. Do not copy one source's value to another source.
4. A cross-year question and answer must identify every source fiscal_year and use only a value whose row/category period matches that fiscal_year, or a current-period value with no separate historical row label. Never relabel a historical row as the report fiscal year. Never compare FY with H1 as if they were the same period.
5. source_ids_used must contain every source_id that supports the answer and no others.
6. Do not mention sources, chunks, pages, evidence, or retrieval in the question.
{question_style_rule}
8. Calculated values are allowed only when requested. Record every calculation using a simple arithmetic expression whose operands occur in the sources. Do not use thousands separators in calculation expressions. Otherwise return an empty calculations list.
9. Prefer useful finance questions over document-location questions.""".replace(
            "{question_style_rule}", question_style_rule
        )
        payload = {
            "prompt_version": GENERATION_PROMPT_VERSION,
            "requested_language": language,
            "requested_difficulty": difficulty,
            "allow_calculations": allow_calculations,
            "question_style": question_style,
            "bundles": bundles,
        }
        try:
            parsed = GeneratedQuestionBatch.model_validate(
                await self._json_completion(system, json.dumps(payload, ensure_ascii=False))
            )
        except ValidationError as exc:
            raise RuntimeError(f"{self.name} returned invalid generated evaluation cases.") from exc
        return parsed.cases


class AlibabaEvaluationProvider(OpenAICompatibleEvaluationProvider):
    def __init__(self, settings: Settings | None = None, *, model: str | None = None) -> None:
        settings = settings or get_settings()
        super().__init__(
            name="alibaba",
            api_key=settings.dashscope_api_key,
            base_url=settings.alibaba_base_url,
            model=model or settings.alibaba_evaluation_model,
            timeout=settings.table_repair_timeout_seconds,
        )

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


def generation_provider_catalog(settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    return [
        {
            "provider": "alibaba",
            "display_name": "Alibaba Cloud / Qwen",
            "model": model,
            "available": bool(settings.dashscope_api_key),
        }
        for model in settings.alibaba_generation_model_list
    ] + [
        {
            "provider": "deepseek",
            "display_name": "DeepSeek",
            "model": model,
            "available": bool(settings.deepseek_api_key),
        }
        for model in settings.deepseek_generation_model_list
    ]


def create_generation_provider(
    provider_name: str,
    model: str | None = None,
    settings: Settings | None = None,
) -> OpenAICompatibleEvaluationProvider:
    settings = settings or get_settings()
    provider_name = provider_name.strip().lower()
    allowed = {
        "alibaba": settings.alibaba_generation_model_list,
        "deepseek": settings.deepseek_generation_model_list,
    }
    if provider_name not in allowed:
        raise ValueError(f"Unsupported evaluation generation provider: {provider_name}")
    selected_model = model or (allowed[provider_name][0] if allowed[provider_name] else "")
    if selected_model not in allowed[provider_name]:
        raise ValueError(
            f"Model {selected_model!r} is not allowed for provider {provider_name}."
        )
    if provider_name == "alibaba":
        return AlibabaEvaluationProvider(settings, model=selected_model)
    return OpenAICompatibleEvaluationProvider(
        name="deepseek",
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=selected_model,
        timeout=settings.table_repair_timeout_seconds,
    )
