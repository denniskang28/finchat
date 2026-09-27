from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    knowledge_base_id: UUID | None = None
    filename: str
    title: str | None
    status: str
    page_count: int | None
    company: str | None = None
    fiscal_year: int | None = None
    document_type: str | None = None
    language: str | None = None
    metadata: dict[str, Any]
    created_at: datetime


class UploadResponse(DocumentSummary):
    pass


class DocumentMetadataUpdate(BaseModel):
    company: str | None = Field(default=None, max_length=200)
    fiscal_year: int | None = Field(default=None, ge=1900, le=2200)
    document_type: str | None = Field(default=None, max_length=50)
    language: str | None = Field(default=None, max_length=20)


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class KnowledgeBaseSummary(BaseModel):
    id: UUID
    name: str
    description: str | None
    document_count: int = 0
    ready_document_count: int = 0
    created_at: datetime


class IngestionJobSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    status: str
    stage: str
    progress: int
    attempts: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class AsyncUploadResponse(BaseModel):
    document: DocumentSummary
    job: IngestionJobSummary


class BulkIndexResponse(BaseModel):
    knowledge_base_id: UUID
    queued_jobs: int
    skipped_active_jobs: int
    job_ids: list[UUID]


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


class RetrievalRequest(BaseModel):
    document_id: UUID | None = None
    document_ids: list[UUID] = Field(default_factory=list, max_length=500)
    knowledge_base_id: UUID | None = None
    company: str | None = Field(default=None, max_length=200)
    fiscal_year: int | None = Field(default=None, ge=1900, le=2200)
    document_type: str | None = Field(default=None, max_length=50)
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["BASELINE", "SEMANTIC", "PRODUCTION"] = "PRODUCTION"

    @model_validator(mode="after")
    def validate_scope(self):
        scopes = bool(self.document_id) + bool(self.document_ids) + bool(self.knowledge_base_id)
        if scopes != 1:
            raise ValueError(
                "Provide exactly one scope: document_id, document_ids, or knowledge_base_id."
            )
        return self


class RetrievalHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    rank: int
    final_rank: int | None = None
    chunk_type: str
    representation: str
    comparison_key: str | None
    file: str
    page: int
    table_title: str | None = None
    row_label: str | None = None
    raw_content: str | None = None
    semantic_content: str | None = None
    parent_content: str | None = None
    content: str
    vector_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrievalDebugResponse(BaseModel):
    original_query: str
    retrieval_mode: Literal["BASELINE", "SEMANTIC", "PRODUCTION"]
    document_ids: list[UUID]
    knowledge_base_id: UUID | None = None
    query_years: list[int] = Field(default_factory=list)
    embedding_model: str
    rerank_model: str
    vector_results: list[RetrievalHit]
    lexical_results: list[RetrievalHit]
    rrf_results: list[RetrievalHit]
    reranked_results: list[RetrievalHit]


class IndexRequest(BaseModel):
    force: bool = False


class IndexResponse(BaseModel):
    document_id: UUID
    model: str
    dimensions: int
    indexed_chunks: int
    skipped_chunks: int
    total_indexable_chunks: int


class GoldenQueryResult(BaseModel):
    query_id: str
    query: str
    mode: Literal["BASELINE", "SEMANTIC"]
    expected_target_ids: list[str]
    vector_ranks: dict[str, int | None]
    lexical_ranks: dict[str, int | None]
    rrf_ranks: dict[str, int | None]
    rerank_ranks: dict[str, int | None]
    target_ranks: dict[str, int | None]
    first_relevant_rank: int | None
    first_complete_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    failure_stage: str | None = None


class RetrievalMetrics(BaseModel):
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float


class RetrievalEvaluationResponse(BaseModel):
    document_id: UUID
    results: list[GoldenQueryResult]
    metrics: dict[Literal["BASELINE", "SEMANTIC"], RetrievalMetrics]


class QAEvidence(BaseModel):
    evidence_number: int
    chunk_id: UUID
    filename: str
    page: int
    chunk_type: str
    content: str


class QARetrieveResponse(BaseModel):
    retrieval_id: UUID
    retrieval: RetrievalDebugResponse
    final_evidence: list[QAEvidence]


class QACitation(BaseModel):
    evidence_number: int
    filename: str
    page: int
    chunk_id: UUID


class QAAnswerResponse(BaseModel):
    retrieval_id: UUID
    answer: str
    insufficient_evidence: bool
    citations: list[QACitation]
    model: str


class EvaluationEvidenceTarget(BaseModel):
    document_sha256: str | None = None
    filename: str | None = None
    page: int | None = Field(default=None, ge=1)
    comparison_key: str | None = None
    content_hash: str | None = None
    chunk_id: UUID | None = None

    @model_validator(mode="after")
    def validate_locator(self):
        if not any(
            (
                self.document_sha256,
                self.filename,
                self.page,
                self.comparison_key,
                self.content_hash,
                self.chunk_id,
            )
        ):
            raise ValueError("Evidence target must contain at least one locator.")
        return self


class EvaluationDatasetCreate(BaseModel):
    knowledge_base_id: UUID
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class EvaluationCaseCreate(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    expected_answer: str = Field(min_length=1, max_length=8000)
    language: str = Field(default="en", min_length=2, max_length=20)
    expected_insufficient: bool = False
    required_evidence: list[EvaluationEvidenceTarget] = Field(default_factory=list)
    scope: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, max_length=20)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    status: Literal["DRAFT", "APPROVED"] = "DRAFT"


class EvaluationCaseUpdate(BaseModel):
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    expected_answer: str | None = Field(default=None, min_length=1, max_length=8000)
    language: str | None = Field(default=None, min_length=2, max_length=20)
    expected_insufficient: bool | None = None
    required_evidence: list[EvaluationEvidenceTarget] | None = None
    scope: dict[str, Any] | None = None
    tags: list[str] | None = Field(default=None, max_length=20)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    status: Literal["DRAFT", "APPROVED"] | None = None


class EvaluationCaseSummary(BaseModel):
    id: UUID
    dataset_id: UUID
    question: str
    expected_answer: str
    language: str
    expected_insufficient: bool
    required_evidence: list[EvaluationEvidenceTarget]
    scope: dict[str, Any]
    tags: list[str]
    difficulty: str
    source: str
    status: str
    created_at: datetime


class EvaluationDatasetSummary(BaseModel):
    id: UUID
    knowledge_base_id: UUID
    name: str
    description: str | None
    status: str
    version: int
    case_count: int = 0
    approved_case_count: int = 0
    created_at: datetime
    published_at: datetime | None


class EvaluationDatasetDetail(EvaluationDatasetSummary):
    cases: list[EvaluationCaseSummary]


class EvaluationGenerateRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=30)
    language: Literal["en", "zh"] = "en"
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "mixed"
    include_insufficient: bool = False


class EvaluationImportResponse(BaseModel):
    imported: int
    rejected: int
    errors: list[str]


class EvaluationRunCreate(BaseModel):
    dataset_id: UUID
    retrieval_mode: Literal["BASELINE", "SEMANTIC", "PRODUCTION"] = "PRODUCTION"


class EvaluationRunSummary(BaseModel):
    id: UUID
    dataset_id: UUID
    knowledge_base_id: UUID
    status: str
    retrieval_mode: str
    progress: int
    config: dict[str, Any]
    metrics: dict[str, Any]
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class EvaluationCaseResultSummary(BaseModel):
    id: UUID
    run_id: UUID
    case_id: UUID
    status: str
    question: str
    expected_answer: str
    retrieval: dict[str, Any]
    final_evidence: list[dict[str, Any]]
    answer: str | None
    insufficient_evidence: bool | None
    citations: list[dict[str, Any]]
    deterministic_metrics: dict[str, Any]
    judge: dict[str, Any]
    failure_stage: str | None
    latency_seconds: float | None
    error: str | None


class EvaluationRunDetail(EvaluationRunSummary):
    results: list[EvaluationCaseResultSummary]
