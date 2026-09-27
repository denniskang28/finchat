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
    font_size: float | None = None


class ContentFact(BaseModel):
    fact_id: str
    label: str
    value: str | None = None
    content: str
    source_token_ids: list[str]
    bbox: tuple[float, float, float, float]
    footnotes: list[str] = Field(default_factory=list)


class ContentSection(BaseModel):
    section_id: str
    page_number: int
    heading_path: list[str]
    content: str
    raw_content: str
    source_token_ids: list[str]
    bbox: tuple[float, float, float, float]
    facts: list[ContentFact] = Field(default_factory=list)


class UncoveredToken(BaseModel):
    token_id: str
    text: str
    bbox: tuple[float, float, float, float]
    reason: Literal["NUMERIC", "PROMINENT_TEXT"]


class ContentCoverage(BaseModel):
    important_token_count: int
    covered_important_token_count: int
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    uncovered_tokens: list[UncoveredToken] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PageContent(BaseModel):
    page_number: int
    title: str
    summary: str
    sections: list[ContentSection] = Field(default_factory=list)
    coverage: ContentCoverage
    parse_warnings: list[str] = Field(default_factory=list)


class TableIssue(BaseModel):
    code: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    message: str
    row_index: int | None = None
    column_index: int | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class TableRepair(BaseModel):
    operation: Literal[
        "REPLACE_CELL",
        "MERGE_SPLIT_ROW",
        "REPLACE_HEADER",
        "MERGE_HEADER",
        "DELETE_EMPTY_COLUMN",
        "TRIM_CONTAMINATED_CELL",
        "REASSIGN_TOKEN",
        "RECONSTRUCT_TABLE",
    ] = "REPLACE_CELL"
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


class ChunkSummary(BaseModel):
    id: UUID
    page_number: int
    chunk_type: str
    representation: str
    comparison_key: str | None
    content_preview: str
    created_at: datetime


class ChunkDetail(BaseModel):
    id: UUID
    document_id: UUID
    page_number: int
    chunk_type: str
    representation: str
    comparison_key: str | None
    content: str
    raw_content: str | None
    semantic_content: str | None
    metadata: dict[str, Any]
    embedding_dimensions: int | None
    created_at: datetime
