# Semantic Table RAG POC

This repository implements PDF ingestion, layout-aware content extraction, table extraction, evidence-backed repair, chunk inspection, and a baseline-versus-semantic retrieval experiment. It does not implement answer generation.

Every page uses the same LLM-primary, evidence-verified pipeline; there are no page-number or document-specific parser branches:

1. `pdfplumber` extracts native text, coordinates, source tokens, and a raw table baseline.
2. A configurable multimodal provider identifies tables, data charts, narrative sections, KPI facts, and qualitative callouts on the complete page.
3. Every LLM heading, fact, header, unit, context note, and cell must map back to PDF tokens before it is accepted. Numeric evidence always requires an exact match.
4. Numeric signs, punctuation, footnote markers, row geometry, and inferred chart units are validated deterministically.
5. A rejected or unavailable page-level result falls back to the existing `pdfplumber` extraction and evidence-backed table repair pipeline, with the reason shown in debug warnings.
6. Raw extraction and canonical LLM structures remain available side by side in the debug UI.
7. Each page emits a hierarchy of `PAGE_SUMMARY`, `SECTION`, and `FACT` chunks. Important-token coverage exposes headings and numbers that were not assigned to any accepted structure.
8. Retrieval indexes `TEXT`, `TABLE_SUMMARY`, `RAW_ROW`, and `SEMANTIC_ROW` chunks with Alibaba Cloud embeddings in PostgreSQL pgvector.
9. The retrieval debugger exposes lexical top 20, vector top 20, RRF top 20, and Alibaba reranker top 6 results without generating an answer.

## Run

```bash
cp .env.example .env
# Add your Alibaba Cloud Model Studio key to DASHSCOPE_API_KEY.
docker compose up --build
```

- UI: http://localhost:5173
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

Upload a PDF in the UI and choose either Qwen or DeepSeek. Parsing is synchronous for this POC; the AIA 2025 report remains the golden validation sample. Each document records the provider, model, and wall-clock parsing duration so equivalent uploads can be compared.

Open the `Retrieval` tab for a parsed document, click `Build index`, select `Baseline` or `Semantic`, and run a query. Baseline uses `TEXT + RAW_ROW`; Semantic uses `TEXT + TABLE_SUMMARY + SEMANTIC_ROW`.

## Retrieval providers

Retrieval uses Alibaba Cloud Model Studio. Document and query embeddings use distinct `text_type` values through the native embedding API.

```dotenv
ALIBABA_EMBEDDING_MODEL=qwen3.7-text-embedding
ALIBABA_EMBEDDING_DIMENSIONS=1024
ALIBABA_EMBEDDING_URL=https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
ALIBABA_RERANK_MODEL=qwen3-rerank
ALIBABA_RERANK_URL=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
```

The fixed experiment is lexical top 20 + vector top 20 -> RRF (`k=60`) top 20 -> rerank -> final top 6. See `docs/retrieval-evaluation.md` for the golden-query results.

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
