from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ParsedColumn(BaseModel):
    key: str
    source_labels: list[str]
    label: str
    semantic_label: str
    unit: str | None = None
    value_type: Literal["TEXT", "NUMBER", "PERCENT", "CURRENCY"] = "TEXT"
    period_label: str | None = None
    as_of_date: str | None = None
    is_row_label: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedRow(BaseModel):
    row_index: int
    row_label: str
    values: dict[str, str | int | float | None]
    display_values: dict[str, str | None]
    raw_cells: list[str | None]
    is_total: bool = False
    footnote_markers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedTable(BaseModel):
    table_id: str
    document_id: UUID
    page_number: int
    bbox: tuple[float, float, float, float]
    source_kind: Literal["TABLE", "CHART"] = "TABLE"
    title: str
    subtitle: str | None = None
    context: dict[str, str] = Field(default_factory=dict)
    columns: list[ParsedColumn]
    rows: list[ParsedRow]
    footnotes: list[str] = Field(default_factory=list)
    extraction_method: str
    parse_warnings: list[str] = Field(default_factory=list)


class SourceToken(BaseModel):
    token_id: str
    text: str
    bbox: tuple[float, float, float, float]


class TableIssue(BaseModel):
    code: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    message: str
    row_index: int | None = None
    column_index: int | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class TableRepair(BaseModel):
    operation: Literal["REPLACE_CELL", "MERGE_SPLIT_ROW"] = "REPLACE_CELL"
    source: Literal["DETERMINISTIC", "LLM"]
    provider: str | None = None
    model: str | None = None
    row_index: int
    column_index: int
    original_value: str | None = None
    repaired_value: str
    source_token_ids: list[str] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    applied: bool = True


class TableQuality(BaseModel):
    status: Literal["PASS", "REPAIRED", "NEEDS_REVIEW"] = "PASS"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    initial_issues: list[TableIssue] = Field(default_factory=list)
    remaining_issues: list[TableIssue] = Field(default_factory=list)


class ParsedTableArtifact(BaseModel):
    raw_table: ParsedTable
    canonical_table: ParsedTable
    quality: TableQuality = Field(default_factory=TableQuality)
    repairs: list[TableRepair] = Field(default_factory=list)


class RowRepresentation(BaseModel):
    comparison_key: str
    row_index: int
    row_label: str
    content: str


class TableDebug(BaseModel):
    parsed_table: ParsedTable
    raw_parsed_table: ParsedTable | None = None
    quality: TableQuality = Field(default_factory=TableQuality)
    repairs: list[TableRepair] = Field(default_factory=list)
    raw_rows: list[RowRepresentation]
    markdown: str
    semantic_rows: list[RowRepresentation]


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    title: str | None
    status: str
    page_count: int | None
    metadata: dict[str, Any]
    created_at: datetime


class UploadResponse(DocumentSummary):
    pass


class PageText(BaseModel):
    page_number: int
    text: str
