# Semantic Table RAG POC

This repository implements PDF ingestion, table extraction, evidence-backed table repair, and table-representation inspection. It does not implement embeddings, retrieval, reranking, or answer generation.

Every page uses the same LLM-primary, evidence-verified pipeline; there are no page-number or document-specific parser branches:

1. `pdfplumber` extracts native text, coordinates, source tokens, and a raw table baseline.
2. A configurable multimodal provider identifies every table and data chart on the complete page.
3. Every LLM header, unit, context note, and cell must map back to exact PDF tokens before it is accepted.
4. Numeric signs, punctuation, footnote markers, row geometry, and inferred chart units are validated deterministically.
5. A rejected or unavailable page-level result falls back to the existing `pdfplumber` extraction and evidence-backed table repair pipeline, with the reason shown in debug warnings.
6. Raw extraction and canonical LLM structures remain available side by side in the debug UI.

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
