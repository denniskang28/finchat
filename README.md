# Semantic Table RAG POC

This repository currently implements ingestion and table-representation inspection only. It does not implement embeddings, retrieval, reranking, or answer generation.

## Run

```bash
docker compose up --build
```

- UI: http://localhost:5173
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

Upload the AIA 2025 PDF in the UI. Parsing is synchronous for this POC.

## Test

```bash
docker compose run --rm backend python -m pytest -q
```
