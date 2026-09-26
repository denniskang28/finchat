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


class RowRepresentation(BaseModel):
    comparison_key: str
    row_index: int
    row_label: str
    content: str


class TableDebug(BaseModel):
    parsed_table: ParsedTable
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

