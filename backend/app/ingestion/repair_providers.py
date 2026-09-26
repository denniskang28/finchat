import json
import re
from abc import ABC, abstractmethod
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.schemas import ParsedTable, SourceToken, TableIssue


class CellRepairProposal(BaseModel):
    operation: Literal[
        "REPLACE_CELL",
        "REPLACE_HEADER",
        "MERGE_HEADER",
        "DELETE_EMPTY_COLUMN",
        "TRIM_CONTAMINATED_CELL",
        "REASSIGN_TOKEN",
    ] = "REPLACE_CELL"
    row_index: int | None = None
    column_index: int
    value: str | None = None
    source_token_ids: list[str] = Field(default_factory=list)
    source_row_index: int | None = None
    source_column_index: int | None = None
    consume_row_indices: list[int] = Field(default_factory=list)
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
        compact_table = {
            "table_id": table.table_id,
            "page_number": table.page_number,
            "bbox": table.bbox,
            "title": table.title,
            "columns": [
                {
                    "column_index": index,
                    "label": column.label,
                    "source_labels": column.source_labels,
                    "metadata": column.metadata,
                }
                for index, column in enumerate(table.columns)
            ],
            "rows": [
                {
                    "row_index": row.row_index,
                    "cells": row.raw_cells,
                    "metadata": row.metadata,
                }
                for row in table.rows
            ],
        }
        prompt = {
            "task": "Propose evidence-backed structural repairs for a financial table.",
            "constraints": [
                "Return JSON only with a top-level repairs array.",
                "Allowed operations: REPLACE_CELL, REPLACE_HEADER, MERGE_HEADER, DELETE_EMPTY_COLUMN, TRIM_CONTAMINATED_CELL, REASSIGN_TOKEN.",
                "REPLACE_CELL fills an empty cell; row_index is one-based and column_index is zero-based.",
                "REPLACE_HEADER replaces a placeholder header using source tokens.",
                "MERGE_HEADER combines multi-level header tokens and may list consumed header rows in consume_row_indices.",
                "When HEADER_ROW_AS_DATA is reported, use MERGE_HEADER for every non-empty cell in that row, combine its label with the outer header, and include that row in consume_row_indices. Do not use REPLACE_HEADER alone for that pattern.",
                "DELETE_EMPTY_COLUMN is allowed only when every data cell in that column is empty.",
                "TRIM_CONTAMINATED_CELL replaces an existing contaminated cell with a shorter value supported by a subset of its visual tokens.",
                "REASSIGN_TOKEN moves an entire source cell to an empty target cell and requires source_row_index and source_column_index.",
                "Every non-empty value must be supported by source_token_ids.",
                "Do not change or invent numeric values.",
                "Return an empty repairs array when evidence is ambiguous.",
            ],
            "table": compact_table,
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "source_tokens": [token.model_dump(mode="json") for token in tokens],
            "response_shape": {
                "repairs": [
                    {
                        "operation": "MERGE_HEADER",
                        "row_index": 1,
                        "column_index": 2,
                        "value": "1 year / With illiquidity premium",
                        "source_token_ids": ["p1w1"],
                        "source_row_index": None,
                        "source_column_index": None,
                        "consume_row_indices": [1],
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
                "enable_thinking": False,
                "max_tokens": 3000,
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
        valid_repairs = []
        for candidate in payload.get("repairs", []):
            try:
                valid_repairs.append(CellRepairProposal.model_validate(candidate))
            except ValidationError:
                continue
        return RepairProposalResponse(repairs=valid_repairs)


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
