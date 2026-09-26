import json
import re
from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel, Field

from app.config import Settings
from app.schemas import ParsedTable, SourceToken, TableIssue


class CellRepairProposal(BaseModel):
    row_index: int
    column_index: int
    value: str
    source_token_ids: list[str] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class RepairProposalResponse(BaseModel):
    repairs: list[CellRepairProposal] = Field(default_factory=list)


class TableRepairProvider(ABC):
    name: str
    model: str

    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def propose_repairs(
        self,
        *,
        table: ParsedTable,
        issues: list[TableIssue],
        tokens: list[SourceToken],
        image_data_url: str,
    ) -> RepairProposalResponse:
        raise NotImplementedError


class DisabledTableRepairProvider(TableRepairProvider):
    name = "disabled"
    model = ""

    @property
    def available(self) -> bool:
        return False

    def propose_repairs(self, **_: object) -> RepairProposalResponse:
        return RepairProposalResponse()


class OpenAICompatibleVisionProvider(TableRepairProvider):
    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def propose_repairs(
        self,
        *,
        table: ParsedTable,
        issues: list[TableIssue],
        tokens: list[SourceToken],
        image_data_url: str,
    ) -> RepairProposalResponse:
        prompt = {
            "task": "Propose evidence-backed repairs for missing table cells.",
            "constraints": [
                "Return JSON only with a top-level repairs array.",
                "Only propose REPLACE_CELL repairs for currently empty cells.",
                "Every character in value must be supported by source_token_ids.",
                "Do not change or invent numeric values.",
                "Use one-based row_index and zero-based column_index.",
                "Return an empty repairs array when evidence is ambiguous.",
            ],
            "table": table.model_dump(mode="json"),
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "source_tokens": [token.model_dump(mode="json") for token in tokens],
            "response_shape": {
                "repairs": [
                    {
                        "row_index": 1,
                        "column_index": 0,
                        "value": "Exact text from source tokens",
                        "source_token_ids": ["p1w1"],
                        "reason": "Short evidence-based explanation",
                        "confidence": 0.95,
                    }
                ]
            },
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": "You repair financial table structure using only supplied visual and token evidence.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                        ],
                    },
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
        payload = json.loads(match.group(1) if match else content)
        return RepairProposalResponse.model_validate(payload)


def create_table_repair_provider(settings: Settings) -> TableRepairProvider:
    if not settings.table_repair_enabled:
        return DisabledTableRepairProvider()
    provider = settings.table_repair_provider.strip().lower()
    if provider == "alibaba":
        return OpenAICompatibleVisionProvider(
            name="alibaba",
            api_key=settings.dashscope_api_key,
            base_url=settings.alibaba_base_url,
            model=settings.alibaba_vision_model,
            timeout_seconds=settings.table_repair_timeout_seconds,
        )
    if provider == "openai_compatible":
        return OpenAICompatibleVisionProvider(
            name="openai_compatible",
            api_key=settings.openai_compatible_api_key,
            base_url=settings.openai_compatible_base_url,
            model=settings.openai_compatible_vision_model,
            timeout_seconds=settings.table_repair_timeout_seconds,
        )
    raise ValueError(f"Unsupported TABLE_REPAIR_PROVIDER: {settings.table_repair_provider}")
