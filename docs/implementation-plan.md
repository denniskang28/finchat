# Context-Enriched Semantic Table RAG POC - Implementation Plan

## 1. Objective and decision boundary

The POC answers one question: does a context-enriched, self-contained row retrieve and support answers more reliably than either a raw extracted row or a whole-table Markdown chunk for financial-report questions?

The three experimental representations are:

1. `RAW_ROW`: extracted row cells joined in source order, without added table, column, date, unit, or scope context.
2. `MARKDOWN`: one structure-preserving Markdown chunk for the complete table.
3. `SEMANTIC_ROW`: one deterministic, context-enriched record per row.

Each retrieval run must select exactly one table representation. Narrative text can be included in every run, but raw, Markdown, and semantic table chunks must never be pooled in the same experimental arm.

This plan does not add a financial fact store, ontology, agent framework, query-planning DAG, OpenSearch, Redis, Kafka, OCR platform, or microservices.

## 2. Sample PDF inspection

Source: `docs/samples/AIA Group 2025 Annual Results Analyst Presentation EN (FINAL) - 27 Mar.pdf`

The source is a 99-page, tagged PowerPoint export with a 960 x 540 point page size. The target pages contain embedded text and vector drawing primitives; none of the three inspected pages contains a raster image. Printed slide numbers and PDF page indices match for pages 70, 91, and 93.

| PDF page | Content | Structure observed | Extraction result |
|---:|---|---|---|
| 70 | Risk Discount Rate and Risk Premium | One ruled table; two date groups, three measures per date, 13 market rows plus a weighted-average row; table and row footnotes | `pdfplumber.find_tables()` produced one 16 x 7 table and preserved the two-level header as merged cells plus null continuations. Values and `n/a` cells were correct. |
| 91 | Corporate Bonds by Geography and by Sector | Two ruled tables plus a separate donut chart; common scope is Non-par and Surplus Assets; note establishes 31 Dec 2025 | The geography table was extracted exactly as 5 x 3. The sector table's right edge was joined to nearby chart geometry, so the parser must crop or discard trailing empty/chart columns. |
| 93 | AIA China investment allocation | Donut chart with positioned labels and percentages, plus narrative bullets and two footnotes; not a table | No ruled table is detected. The generic fallback finds a compact set of 3-10 pure percentages summing to 100%, then asks the configured vision provider to associate labels; every association is verified against PDF tokens and geometry. |

Verified target values:

- Page 91 geography: Asia Pacific `$17.3b / 63%`; United States `$6.2b / 22%`; Other `$4.2b / 15%`; Total `$27.7b / 100%`.
- Page 70 Mainland China at 31 Dec 2025: risk discount rate `8.30%`, long-term 10-year government bonds `2.70%`, risk premium `5.60%`.
- Page 93 AIA China allocation: government and government agency bonds `74%`, corporate bonds `5%`, equities `18%`, real estate `2%`, other `1%`.

## 3. Parser recommendation

Use `pdfplumber` as the primary table and positioned-text parser for this sample.

Reasons:

- It extracted page 70 and the page 91 geography table correctly with default line-based detection.
- It exposes words, characters, lines, rectangles, curves, bounding boxes, and crop operations through one API.
- It makes parser behavior inspectable, which is more important for this POC than broad document-format coverage.
- Page 93 can be handled with its word coordinates without OCR because all labels are embedded text.

Use PyMuPDF only for document-level reading and page rendering in the application. Poppler rendering remains suitable for test fixtures and visual verification. Do not put Docling, OCR, or a vision model on the primary path for this PDF. A vision fallback may be evaluated later only when a target page has no usable embedded text or fails explicit parser assertions.

The page 93 result uses the same generic percentage-distribution fallback available to every page. It contains no known category names or page-number routing. If the candidate percentages do not sum to 100%, the visual associations are ambiguous, or any proposed value lacks token evidence, the result remains `NEEDS_REVIEW`.

## 4. Planned POC flow

```text
PDF upload
  -> page text and vector geometry
  -> ruled-table candidates / generic percentage-chart candidates
  -> ParsedTable validation
  -> RAW_ROW + MARKDOWN + SEMANTIC_ROW renderers
  -> separate embeddings per representation
  -> PostgreSQL + pgvector
  -> lexical and vector candidates
  -> reciprocal-rank fusion
  -> Alibaba Cloud rerank
  -> top evidence
  -> optional answer generation
  -> answer plus complete retrieval trace
```

Answer generation is deliberately last. Retrieval must be judged independently before an LLM is allowed to make the result look plausible.

## 5. Implementation phases and review gates

### Phase 1 - Parser fixtures

- Add page-level fixtures for pages 70, 91, and 93.
- Implement table extraction with explicit page, bounding-box, row-count, column-count, and required-label assertions.
- Normalize multi-level headers into complete column paths using the common repair pipeline.
- Reconstruct a contiguous dense table region when sparse trailing columns contain chart contamination.
- Emit token-verified percentage series through an explicit `source_kind=CHART` path when a no-table page contains a compact 100% distribution.
- Persist parse warnings; never silently switch parser strategy.

Gate: expected labels, values, units, dates, and footnotes match the inspected pages.

### Phase 2 - Representation generation

- Render raw rows without inferred context.
- Render one structure-preserving Markdown chunk per table.
- Render semantic rows with table title, scope, period, row label, semantic column label, value, and unit.
- Assign a shared `comparison_key` to raw and semantic versions of the same logical row.

Gate: snapshot tests prove that only the semantic representation adds context and that source values are unchanged.

### Phase 3 - Storage and Alibaba providers

- Create only `documents` and `chunks` tables.
- Add the 1024-dimension pgvector column, generated `tsvector`, GIN index, and HNSW index.
- Implement provider interfaces for chat, document/query embedding, and reranking.
- Inject all model IDs, region endpoints, workspace ID, dimensions, and top-K values through configuration.

Gate: provider contract tests use fakes; one opt-in integration test verifies each configured Alibaba endpoint.

### Phase 4 - Retrieval and debug API

- Run lexical and vector searches with the same document and representation filters.
- Fuse 1-based ranks with RRF using `k=60`.
- Rerank the top 20 fused candidates and retain the top 6 evidence chunks.
- Return every stage, rank, score, filter, model ID, and latency in the debug payload.

Gate: a retrieval run is reproducible from its debug response, and a missing/failed channel is visible rather than treated as an empty success.

### Phase 5 - Golden evaluation

- Run the 10 fixed queries against `RAW_ROW`, `MARKDOWN`, and `SEMANTIC_ROW` arms.
- Record Hit@1, Hit@3, Hit@5, MRR, answer accuracy, and citation/page accuracy.
- Report per-query ranks as well as aggregate metrics; do not declare success from aggregate answer accuracy alone.

Gate: the POC can state whether semantic rows improve table-row Hit@3 over both baselines and show where gains or regressions occurred.

### Phase 6 - Minimal QA and UI

- Add answer generation using only the selected top evidence.
- Add upload, question input, representation selector, answer, table inspection, and four retrieval-stage panels.
- Keep `Retrieve Again` separate from `Regenerate Answer`.

Gate: the UI exposes raw and semantic content, exact source page, table/row identity, all scores, and the evidence actually sent to chat.

## 6. Acceptance criteria

- Page 70 and page 91 target tables parse with exact row/column semantics through the generic pipeline.
- Page 93 is clearly identified as a chart and either passes generic token verification or reports an explicit unsupported/failed state.
- Every logical table row has comparable raw and semantic variants.
- The three table representations can be tested independently with identical retrieval settings.
- The correct target is in the top 3 for each supported golden query, with failures retained in the report.
- The final answer cites only retrieved evidence and preserves period and units.
- Parser, lexical, vector, fusion, rerank, and answer failures are distinguishable.

## 7. Known limitations

- English financial reports only.
- Native-text PDFs only; no general OCR path in the first implementation.
- Conventional two-dimensional tables plus compact percentage-distribution charts that satisfy the generic candidate rules.
- No claim of general chart extraction, cross-document analytics, or production scale.
- PostgreSQL `simple` full-text search does not provide useful Chinese segmentation. Chinese golden queries therefore primarily test multilingual embeddings and reranking unless optional query rewrite is explicitly enabled and reported.
