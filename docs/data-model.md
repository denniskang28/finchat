# Semantic Table RAG POC - Data Model

## 1. Modeling principles

- Preserve source text and normalized meaning separately.
- Keep enough provenance to explain every generated chunk.
- Store each retrieval representation as a separate chunk because it needs its own lexical index and embedding.
- Give raw and semantic variants of the same logical row a shared `comparison_key`.
- Keep parsed tables in ingestion memory and chunk metadata; do not add relational table/cell storage for the POC.
- Treat dates, units, merged-header paths, `n/a`, totals, and footnotes as data, not formatting noise.

## 2. Domain models

The definitions below are contracts for the later Pydantic implementation. `Decimal` is used for normalized financial values; `display_values` preserves the exact source strings.

```python
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

Scalar = str | int | Decimal | None
BBox = tuple[float, float, float, float]


class ParsedColumn(BaseModel):
    key: str
    source_labels: list[str]          # full merged-header path, outer to inner
    label: str                       # display label
    semantic_label: str              # e.g. "Risk discount rate"
    unit: str | None                 # e.g. "%" or "USD billion"
    value_type: Literal["TEXT", "NUMBER", "PERCENT", "CURRENCY"]
    period_label: str | None         # exact source label
    as_of_date: date | None
    is_row_label: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedRow(BaseModel):
    row_index: int
    row_label: str
    values: dict[str, Scalar]         # keyed by ParsedColumn.key
    display_values: dict[str, str | None]
    raw_cells: list[str | None]       # source order; no inferred labels
    is_total: bool = False
    footnote_markers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedTable(BaseModel):
    table_id: str                    # stable within the document
    document_id: UUID
    page_number: int                 # 1-based PDF page
    bbox: BBox
    source_kind: Literal["TABLE", "CHART"]
    title: str
    subtitle: str | None
    context: dict[str, str]          # portfolio, as_of_date, currency, etc.
    columns: list[ParsedColumn]
    rows: list[ParsedRow]
    footnotes: list[str] = Field(default_factory=list)
    extraction_method: str           # e.g. "pdfplumber.lines"
    parse_warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    id: UUID
    document_id: UUID
    page_number: int
    chunk_type: Literal[
        "TEXT", "TABLE_SUMMARY", "TABLE_ROW", "TABLE_MARKDOWN", "CHART_SERIES"
    ]
    representation: Literal[
        "TEXT", "RAW_ROW", "MARKDOWN", "SEMANTIC_ROW", "SUMMARY"
    ]
    comparison_key: str | None       # same key for raw/semantic logical row
    content: str                     # the text indexed in this representation
    raw_content: str | None
    semantic_content: str | None
    metadata: dict[str, Any]
    embedding: list[float] | None = None
    created_at: datetime


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_number: int
    chunk_type: str
    representation: str
    comparison_key: str | None
    content: str
    raw_content: str | None
    semantic_content: str | None
    metadata: dict[str, Any]

    vector_rank: int | None
    vector_score: float | None       # cosine similarity, higher is better
    lexical_rank: int | None
    lexical_score: float | None      # ts_rank_cd, higher is better
    rrf_rank: int | None
    rrf_score: float | None
    rerank_rank: int | None
    rerank_score: float | None
    final_rank: int | None
```

### Required invariants

- `page_number` is 1-based and refers to the PDF page, not an inferred section number.
- Column keys are unique within a table.
- `display_values` and `raw_cells` retain source precision and `n/a`; normalization never invents zeroes.
- Every `RAW_ROW` and `SEMANTIC_ROW` pair has the same `comparison_key` and source metadata.
- `content` is exactly what is embedded, lexically indexed, reranked, and displayed for that arm.
- Scores from different retrieval stages are never collapsed into a generic `score`.

## 3. Minimum PostgreSQL + pgvector schema

Application code generates UUIDs. The embedding dimension is fixed by migration and must match the configured provider dimension.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    title TEXT,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PROCESSING', 'READY', 'FAILED')),
    page_count INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    chunk_type TEXT NOT NULL CHECK (chunk_type IN (
        'TEXT', 'TABLE_SUMMARY', 'TABLE_ROW', 'TABLE_MARKDOWN', 'CHART_SERIES'
    )),
    representation TEXT NOT NULL CHECK (representation IN (
        'TEXT', 'RAW_ROW', 'MARKDOWN', 'SEMANTIC_ROW', 'SUMMARY'
    )),
    comparison_key TEXT,
    content TEXT NOT NULL,
    raw_content TEXT,
    semantic_content TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple'::regconfig, content)
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX uq_chunks_representation_comparison
    ON chunks (document_id, representation, comparison_key)
    WHERE comparison_key IS NOT NULL;

CREATE INDEX idx_chunks_document_representation
    ON chunks (document_id, representation, chunk_type);

CREATE INDEX idx_chunks_search
    ON chunks USING GIN (search_vector);

CREATE INDEX idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);
```

This schema intentionally has no `tables`, `rows`, `cells`, query-log, user, or tenant tables. A retrieval trace may initially be returned to the caller and logged as structured application output rather than persisted.

## 4. Chunk metadata

Metadata is descriptive and filterable but does not replace `content`.

```json
{
  "element_type": "TABLE_ROW",
  "source_kind": "TABLE",
  "table_id": "p91_corporate_bonds_geography",
  "table_title": "Corporate Bonds by Geography",
  "table_context": {
    "portfolio": "Non-par and Surplus Assets",
    "as_of_date": "2025-12-31"
  },
  "row_index": 2,
  "row_label": "United States",
  "is_total": false,
  "source_bbox": [34.9, 138.7, 365.8, 252.9],
  "extraction_method": "pdfplumber.lines",
  "parse_warnings": []
}
```

## 5. Representation generation

### 5.1 Raw row

Purpose: strict baseline that measures how well retrieval works when a row loses its surrounding two-dimensional context.

Generation rule:

```text
join(row.raw_cells, " | ")
```

Do not add the table title, headers, dates, units, or scope. Preserve source strings.

Examples:

```text
United States | 6.2 | 22%
```

```text
Mainland China | 10.00 | 3.74 | 6.26 | 8.30 | 2.70 | 5.60
```

The chunk uses `chunk_type=TABLE_ROW`, `representation=RAW_ROW`.

### 5.2 Original Markdown

Purpose: whole-table baseline, parser inspection, evidence display, and answer-context fallback.

Generation rules:

- Include source title, subtitle/scope, applicable date/currency note, and footnotes.
- Preserve row order and exact display values.
- Because Markdown has no column spans, flatten every merged header into a complete, unambiguous path such as `As at 31 Dec 2025 / Risk Discount Rates`.
- Do not paraphrase labels or values.
- Emit one chunk per logical table, with `chunk_type=TABLE_MARKDOWN` and `representation=MARKDOWN`.

Example excerpt:

```markdown
### Corporate Bonds by Geography

Scope: Non-par and Surplus Assets
As of: 31 Dec 2025

| Geography | $b | % of total |
|---|---:|---:|
| Asia Pacific | 17.3 | 63% |
| United States | 6.2 | 22% |
| Other | 4.2 | 15% |
| Total | 27.7 | 100% |

Note: Due to rounding, numbers presented in the table or the chart may not add up precisely.
```

### 5.3 Context-enriched semantic row

Purpose: experimental representation in which each row remains understandable after removal from the table.

Generation is deterministic and uses no per-row LLM call:

1. Add table title.
2. Add table-level scope and date/currency context.
3. Add the row label with its semantic dimension name.
4. For each non-empty value, add the complete semantic column label, period, value, and unit.
5. Add only footnotes that apply to this row or its values.
6. Preserve source precision; never derive a value during rendering.

Page 91 example:

```text
Table: Corporate Bonds by Geography.
Portfolio: Non-par and Surplus Assets.
As of: 31 Dec 2025.
Geography: United States.
Corporate bond amount: USD 6.2 billion.
Share of total corporate bond portfolio: 22%.
```

Page 70 example:

```text
Table: Risk Discount Rate and Risk Premium.
Unit: percent.
Market: Mainland China.
As at 30 Nov 2010, risk discount rate: 10.00%; long-term 10-year government bond rate: 3.74%; risk premium: 6.26%.
As at 31 Dec 2025, risk discount rate: 8.30%; long-term 10-year government bond rate: 2.70%; risk premium: 5.60%.
```

Page 93 targeted chart-series example:

```text
Chart: AIA China Prudent Investment Portfolio.
Portfolio: AIA China invested assets, non-par and surplus assets.
As of: 31 Dec 2025.
Asset class: Corporate bonds.
Share of invested assets: 5%.
Note: Includes less than 1% in loans and deposits.
```

The first two examples use `chunk_type=TABLE_ROW`; the page 93 adapter uses `chunk_type=CHART_SERIES`. Both use `representation=SEMANTIC_ROW` so the same experimental arm can retrieve self-contained structured records while retaining `source_kind` in metadata.

### 5.4 Semantic labels

Use deterministic mappings for the inspected tables. For a complex unseen table, one optional LLM call may map all headers for the table using its title, context, headers, and first two or three rows. The result must validate against the extracted column keys and is stored in metadata. Never call an LLM per row or cell.
