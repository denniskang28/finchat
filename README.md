# Semantic Table RAG POC

This repository implements PDF ingestion, table extraction, evidence-backed table repair, and table-representation inspection. It does not implement embeddings, retrieval, reranking, or answer generation.

The parser uses a hybrid pipeline:

1. `pdfplumber` extracts native text, coordinates, and initial tables.
2. Deterministic quality rules detect and repair issues supported by source tokens.
3. Remaining low-confidence tables may be sent to a configurable multimodal provider.
4. LLM repairs are accepted only when every replacement maps back to source PDF tokens.
5. Raw extraction, canonical table, quality findings, and repair logs remain visible in the debug UI.

## Run

```bash
cp .env.example .env
# Add your Alibaba Cloud Model Studio key to DASHSCOPE_API_KEY.
docker compose up --build
```

- UI: http://localhost:5173
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

Upload the AIA 2025 PDF in the UI. Parsing is synchronous for this POC.

## Table repair provider

Alibaba Cloud is the default provider and uses its OpenAI-compatible multimodal endpoint:

```dotenv
TABLE_REPAIR_ENABLED=true
TABLE_REPAIR_PROVIDER=alibaba
DASHSCOPE_API_KEY=
ALIBABA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ALIBABA_VISION_MODEL=qwen-vl-max-latest
```

An empty API key is valid: deterministic repairs still run and LLM escalation is skipped. To use another OpenAI-compatible multimodal service later, set `TABLE_REPAIR_PROVIDER=openai_compatible` and configure the corresponding `OPENAI_COMPATIBLE_*` variables. The health endpoint reports provider availability without exposing credentials.

## Test

```bash
docker compose run --rm backend python -m pytest -q
```
