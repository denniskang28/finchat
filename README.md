# Semantic Table RAG POC

This repository implements PDF ingestion, layout-aware content extraction, table extraction, evidence-backed repair, chunk inspection, a baseline-versus-semantic retrieval experiment, and a minimal evidence-constrained QA layer.

Every page uses the same LLM-primary, evidence-verified pipeline; there are no page-number or document-specific parser branches:

1. `pdfplumber` extracts native text, coordinates, source tokens, and a raw table baseline.
2. A configurable multimodal provider identifies tables, data charts, narrative sections, KPI facts, and qualitative callouts on the complete page.
3. Every LLM heading, fact, header, unit, context note, and cell must map back to PDF tokens before it is accepted. Numeric evidence always requires an exact match.
4. Numeric signs, punctuation, footnote markers, row geometry, and inferred chart units are validated deterministically.
5. A rejected or unavailable page-level result falls back to the existing `pdfplumber` extraction and evidence-backed table repair pipeline, with the reason shown in debug warnings.
6. Raw extraction and canonical LLM structures remain available side by side in the debug UI.
7. Each page emits a hierarchy of `PAGE_SUMMARY`, `SECTION`, and `FACT` chunks. Important-token coverage exposes headings and numbers that were not assigned to any accepted structure.
8. A durable PostgreSQL queue lets an independent worker parse and index uploads asynchronously. Knowledge bases and finance metadata scope ingestion and retrieval.
9. Retrieval indexes narrative `TEXT`/`SECTION`/`FACT`, `TABLE_SUMMARY`, `RAW_ROW`, and `SEMANTIC_ROW` chunks with Alibaba Cloud embeddings in PostgreSQL pgvector.
10. The retrieval debugger exposes lexical top 20, vector top 20, RRF top 20, and Alibaba reranker top 6 results.
11. DeepSeek answers from a persisted snapshot of those final six chunks only. It never receives the PDF or earlier retrieval candidates.

## Run

```bash
cp .env.example .env
# Add your Alibaba Cloud Model Studio key to DASHSCOPE_API_KEY.
docker compose up --build
```

- UI: http://localhost:5173
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

Select a knowledge base, upload a PDF, add finance metadata, and choose either Qwen or DeepSeek. The API stores the file and returns immediately; the worker then parses and embeds it. The UI polls the document state until it is `READY` or `FAILED`. Each document records the provider, model, and wall-clock parsing duration so equivalent uploads can be compared.

Use `Search all documents` in the sidebar for an explicit knowledge-base-wide query, or open a document's `Retrieval` tab to compare current-file and knowledge-base scopes. Baseline uses `TEXT + RAW_ROW`; Semantic uses `TEXT + TABLE_SUMMARY + SEMANTIC_ROW`; Production adds verified narrative `SECTION` and `FACT` chunks. Company and fiscal-year filters are available for knowledge-base retrieval. `Build index` queues work rather than holding an HTTP request open.

Semantic embeddings include company, fiscal year, document type, and report title. Queries that explicitly mention multiple available years automatically run a focused dense retrieval for each year before vector fusion, RRF, and reranking. The Retrieval Debug UI shows the detected query years and the number of documents actually searched.

The knowledge-base screen is the QA workspace. Its left side contains the conversation, answers, and citations; its right side exposes every retrieval stage and the exact final evidence sent to DeepSeek. `Retrieve Again` creates a new retrieval snapshot without changing the existing answer. `Regenerate Answer` calls DeepSeek again with the current snapshot without rerunning retrieval.

Use `Evaluation` in the sidebar to create versioned test sets manually, import JSON/CSV cases, or generate grounded single-document, cross-year, and cross-document draft cases with an allowed Qwen or DeepSeek model. Generated table questions carry an exact table-title disambiguator and period/value grounding checks. Cases must be approved before a test set can be published. Published versions are immutable; clone one to make the next version. Evaluation runs execute asynchronously through the worker and retain lexical, vector, structured table-expansion, RRF and rerank traces, final evidence, DeepSeek answers, citations, deterministic finance checks, and Qwen Judge output. See `docs/evaluation-design.md`.

The worker is horizontally scalable because jobs are claimed with PostgreSQL `FOR UPDATE SKIP LOCKED`:

```bash
docker compose up --scale worker=4
```

See `docs/knowledge-base-architecture.md` for the data flow, operational boundaries, and remaining production work.

## Retrieval providers

Retrieval uses Alibaba Cloud Model Studio. Document and query embeddings use distinct `text_type` values through the native embedding API.

```dotenv
ALIBABA_EMBEDDING_MODEL=qwen3.7-text-embedding
ALIBABA_EMBEDDING_DIMENSIONS=1024
ALIBABA_EMBEDDING_URL=https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
ALIBABA_RERANK_MODEL=qwen3-rerank
ALIBABA_RERANK_URL=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
ALIBABA_EVALUATION_MODEL=qwen3.8-flash
```

The fixed experiment is lexical top 20 + vector top 20 -> RRF (`k=60`) top 20 -> rerank -> final top 6. See `docs/retrieval-evaluation.md` for the golden-query results.

## QA provider

Answer generation uses DeepSeek's OpenAI-compatible chat endpoint. The API key is shared with the optional DeepSeek parser configuration, while the answer model is configured independently:

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-chat
```

`POST /api/qa/retrieve` runs retrieval and persists the final evidence snapshot. `POST /api/qa/{retrieval_id}/answer` generates or regenerates an answer using only that snapshot. See `docs/qa-evaluation.md` for the end-to-end golden run.

## Table repair provider

Alibaba Cloud is the default provider and uses its OpenAI-compatible multimodal endpoint:

```dotenv
TABLE_REPAIR_ENABLED=true
TABLE_PARSE_MODE=llm_primary
TABLE_PARSE_WORKERS=4
TABLE_REPAIR_PROVIDER=alibaba
DASHSCOPE_API_KEY=
ALIBABA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ALIBABA_VISION_MODEL=qwen3.8-flash
```

DeepSeek's OpenAI-compatible multimodal endpoint is also supported:

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_VISION_MODEL=deepseek-flash
```

An empty API key is valid for the default provider: deterministic repairs still run and LLM escalation is skipped. Providers selected explicitly in the upload UI must have an API key. The provider endpoint reports availability without exposing credentials.

## Test

```bash
docker compose run --rm backend python -m pytest -q
```
