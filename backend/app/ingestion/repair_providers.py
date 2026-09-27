import json
import re
from abc import ABC, abstractmethod
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

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


class EvidenceValue(BaseModel):
    value: str
    source_token_ids: list[str] = Field(default_factory=list)


class TableReconstructionProposal(BaseModel):
    columns: list[EvidenceValue] = Field(min_length=2)
    rows: list[list[EvidenceValue]] = Field(min_length=4)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class ChartRowProposal(BaseModel):
    label: EvidenceValue
    share: EvidenceValue


class ChartExtractionProposal(BaseModel):
    rows: list[ChartRowProposal] = Field(min_length=3)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class PageColumnProposal(BaseModel):
    label: EvidenceValue
    unit: EvidenceValue | None = None
    value_type: Literal["TEXT", "NUMBER", "PERCENT", "CURRENCY"] = "TEXT"
    is_row_label: bool = False

    @field_validator("value_type", mode="before")
    @classmethod
    def normalize_value_type(cls, value: object) -> object:
        if isinstance(value, str) and value.upper() == "PERCENTAGE":
            return "PERCENT"
        return value

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit(cls, value: object) -> object:
        if isinstance(value, str):
            return {"value": value, "source_token_ids": []}
        return value


class PageStructureProposal(BaseModel):
    source_kind: Literal["TABLE", "CHART"]
    title: EvidenceValue | None = None
    columns: list[PageColumnProposal] = Field(min_length=2)
    rows: list[list[EvidenceValue]] = Field(min_length=1)
    context: list[EvidenceValue] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("context", mode="before")
    @classmethod
    def normalize_context(cls, value: object) -> object:
        return [] if value is None else value


class ContentFactProposal(BaseModel):
    label: EvidenceValue
    value: EvidenceValue | None = None


class ContentSectionProposal(BaseModel):
    heading_path: list[EvidenceValue] = Field(min_length=1)
    facts: list[ContentFactProposal] = Field(min_length=1)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class PageStructureResponse(BaseModel):
    structures: list[PageStructureProposal] = Field(default_factory=list)
    content_sections: list[ContentSectionProposal] = Field(default_factory=list)


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

    def propose_reconstruction(
        self,
        *,
        table: ParsedTable,
        issues: list[TableIssue],
        tokens: list[SourceToken],
        image_data_url: str,
    ) -> TableReconstructionProposal | None:
        return None

    def propose_chart_extraction(
        self,
        *,
        table: ParsedTable,
        issues: list[TableIssue],
        tokens: list[SourceToken],
        image_data_url: str,
    ) -> ChartExtractionProposal | None:
        return None

    def propose_page_structures(
        self,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        tokens: list[SourceToken],
        image_data_url: str,
    ) -> PageStructureResponse | None:
        return None


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
        request_options: dict | None = None,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.request_options = (
            {"enable_thinking": False} if request_options is None else request_options
        )

    def _request_body(self, *, max_tokens: int, messages: list[dict]) -> dict:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        body.update(self.request_options)
        return body

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
            json=self._request_body(
                max_tokens=3000,
                messages=[
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
            ),
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

    def propose_page_structures(
        self,
        *,
        page_number: int,
        page_width: float,
        page_height: float,
        tokens: list[SourceToken],
        image_data_url: str,
    ) -> PageStructureResponse | None:
        prompt = {
            "task": "Identify every financial table and data chart, and organize all other decision-relevant page content into evidence-backed sections and facts.",
            "constraints": [
                "Return JSON only with top-level structures and content_sections arrays.",
                "Use an empty structures array when the page has no financial table or data chart, but still extract narrative sections and KPI facts.",
                "Separate adjacent tables, charts, and narrative panels into distinct structures.",
                "Do not turn prose-only callouts, footnotes, page numbers, or decorative content into rows.",
                "Every visible title, header, unit, context note, and non-empty cell must cite the exact source_token_ids that spell it.",
                "Preserve source text, signs, decimal points, parentheses, footnote markers, and numeric values exactly.",
                "Use an empty value and no token ids for a genuinely empty cell.",
                "The first column may use the inferred label 'Row label' or 'Category' without token ids when no visible header exists.",
                "Put units such as ($b), %, or US$m in the column unit field; never return a unit as a data row.",
                "For charts, return the visible series as rows and use source_kind CHART.",
                "Context must be short, directly relevant evidence from the same page; omit unrelated narrative.",
                "For content_sections, create one section per visual topic or panel and return its visible heading hierarchy in heading_path.",
                "Represent KPI cards as facts with an exact label and exact value. Represent qualitative bullets as facts with a label and null value.",
                "Keep each KPI atomic: include its descriptor and adjacent qualifier in one fact label; do not emit the qualifier as a separate null-valued fact.",
                "Do not repeat table or chart rows in content_sections unless the same content is also needed to make a surrounding narrative panel independently understandable.",
                "Every content heading, fact label, and fact value must cite the exact source_token_ids that spell it.",
                "Do not include sources, footnotes, page numbers, logos, or decorative labels as standalone content sections.",
                "Do not infer, calculate, correct, or invent values that are not present in the supplied tokens.",
            ],
            "page": {
                "page_number": page_number,
                "width": page_width,
                "height": page_height,
            },
            "source_tokens": [token.model_dump(mode="json") for token in tokens],
            "response_shape": {
                "structures": [
                    {
                        "source_kind": "TABLE",
                        "title": {
                            "value": "Exact visible title",
                            "source_token_ids": ["p1w1"],
                        },
                        "columns": [
                            {
                                "label": {"value": "Row label", "source_token_ids": []},
                                "unit": None,
                                "value_type": "TEXT",
                                "is_row_label": True,
                            },
                            {
                                "label": {"value": "Exact header", "source_token_ids": ["p1w2"]},
                                "unit": {"value": "($b)", "source_token_ids": ["p1w3"]},
                                "value_type": "CURRENCY",
                                "is_row_label": False,
                            },
                        ],
                        "rows": [
                            [
                                {"value": "Exact row label", "source_token_ids": ["p1w4"]},
                                {"value": "10.0", "source_token_ids": ["p1w5"]},
                            ]
                        ],
                        "context": [
                            {"value": "Exact related context", "source_token_ids": ["p1w6"]}
                        ],
                        "reason": "Short structural explanation",
                        "confidence": 0.99,
                    }
                ],
                "content_sections": [
                    {
                        "heading_path": [
                            {"value": "Exact panel heading", "source_token_ids": ["p1w10"]},
                            {"value": "Exact topic heading", "source_token_ids": ["p1w11"]}
                        ],
                        "facts": [
                            {
                                "label": {"value": "Exact visible fact label", "source_token_ids": ["p1w12"]},
                                "value": {"value": "Exact visible value", "source_token_ids": ["p1w13"]}
                            },
                            {
                                "label": {"value": "Exact qualitative bullet", "source_token_ids": ["p1w14"]},
                                "value": None
                            }
                        ],
                        "reason": "Short grouping explanation",
                        "confidence": 0.99
                    }
                ]
            },
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._request_body(
                max_tokens=12000,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You parse financial-report pages into auditable structures. "
                            "Use only supplied image and PDF token evidence."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                        ],
                    },
                ],
            ),
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
        structures = []
        for candidate in payload.get("structures", []):
            try:
                structures.append(PageStructureProposal.model_validate(candidate))
            except ValidationError:
                continue
        content_sections = []
        for candidate in payload.get("content_sections", []):
            try:
                content_sections.append(ContentSectionProposal.model_validate(candidate))
            except ValidationError:
                continue
        return PageStructureResponse(
            structures=structures,
            content_sections=content_sections,
        )

    def propose_reconstruction(
        self,
        *,
        table: ParsedTable,
        issues: list[TableIssue],
        tokens: list[SourceToken],
        image_data_url: str,
    ) -> TableReconstructionProposal | None:
        prompt = {
            "task": "Reconstruct only the financial table inside the supplied crop.",
            "constraints": [
                "Return JSON only.",
                f"Return exactly {len(table.columns)} columns and {len(table.rows)} rows.",
                "Preserve every supplied dense row in the same order.",
                "Every header and cell must include the exact source_token_ids that spell its value.",
                "The inferred structural header 'Row label' is the only value allowed to have no source_token_ids.",
                "Preserve source text, footnote markers, and numeric values exactly.",
                "Do not include chart labels or narrative outside the cropped table.",
                "Return null when the crop is ambiguous.",
            ],
            "segmentation_hint": {
                "bbox": table.bbox,
                "columns": [column.label for column in table.columns],
                "dense_rows": [row.raw_cells for row in table.rows],
            },
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "source_tokens": [token.model_dump(mode="json") for token in tokens],
            "response_shape": {
                "columns": [
                    {"value": "Exact header", "source_token_ids": ["p1w1"]}
                ],
                "rows": [
                    [
                        {"value": "Exact cell", "source_token_ids": ["p1w2"]}
                    ]
                ],
                "reason": "Short evidence-based explanation",
                "confidence": 0.99,
            },
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._request_body(
                max_tokens=5000,
                messages=[
                    {
                        "role": "system",
                        "content": "You reconstruct financial tables using only supplied visual and PDF token evidence.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                        ],
                    },
                ],
            ),
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
        if payload is None:
            return None
        return TableReconstructionProposal.model_validate(payload)

    def propose_chart_extraction(
        self,
        *,
        table: ParsedTable,
        issues: list[TableIssue],
        tokens: list[SourceToken],
        image_data_url: str,
    ) -> ChartExtractionProposal | None:
        prompt = {
            "task": "Associate labels with the percentage shares in this distribution chart.",
            "constraints": [
                "Return JSON only.",
                f"Return exactly {len(table.rows)} rows.",
                "Use every expected percentage exactly once.",
                "Every label and share must include the exact source_token_ids that spell it.",
                "Do not use prose percentages or labels outside the chart.",
                "Preserve footnote markers exactly.",
                "Return null when any label-share association is ambiguous.",
            ],
            "expected_shares": [
                {
                    "value": row.raw_cells[1],
                    "source_token_id": row.metadata.get("share_token_id"),
                }
                for row in table.rows
            ],
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "source_tokens": [token.model_dump(mode="json") for token in tokens],
            "response_shape": {
                "rows": [
                    {
                        "label": {
                            "value": "Exact category label",
                            "source_token_ids": ["p1w1"],
                        },
                        "share": {
                            "value": "10%",
                            "source_token_ids": ["p1w2"],
                        },
                    }
                ],
                "reason": "Short evidence-based explanation",
                "confidence": 0.99,
            },
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._request_body(
                max_tokens=3000,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract chart series using only supplied visual and PDF token evidence.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                        ],
                    },
                ],
            ),
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
        if payload is None:
            return None
        valid_rows = []
        for candidate in payload.get("rows", []):
            try:
                valid_rows.append(ChartRowProposal.model_validate(candidate))
            except ValidationError:
                continue
        if len(valid_rows) < 3:
            return None
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return ChartExtractionProposal(
            rows=valid_rows,
            reason=str(payload.get("reason", "")),
            confidence=min(1.0, max(0.0, confidence)),
        )


def create_table_repair_provider(
    settings: Settings, provider_name: str | None = None
) -> TableRepairProvider:
    if not settings.table_repair_enabled:
        return DisabledTableRepairProvider()
    provider = (provider_name or settings.table_repair_provider).strip().lower()
    if provider == "alibaba":
        return OpenAICompatibleVisionProvider(
            name="alibaba",
            api_key=settings.dashscope_api_key,
            base_url=settings.alibaba_base_url,
            model=settings.alibaba_vision_model,
            timeout_seconds=settings.table_repair_timeout_seconds,
        )
    if provider == "deepseek":
        return OpenAICompatibleVisionProvider(
            name="deepseek",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_vision_model,
            timeout_seconds=settings.table_repair_timeout_seconds,
            request_options={
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            },
        )
    if provider == "openai_compatible":
        return OpenAICompatibleVisionProvider(
            name="openai_compatible",
            api_key=settings.openai_compatible_api_key,
            base_url=settings.openai_compatible_base_url,
            model=settings.openai_compatible_vision_model,
            timeout_seconds=settings.table_repair_timeout_seconds,
        )
    raise ValueError(f"Unsupported table repair provider: {provider}")
