# Knowledge Base Evaluation

## Scope

This module evaluates the existing retrieval and minimal QA pipeline. It does not optimize prompts, change retrieval parameters, or run an agent. Every evaluation answer follows the same `QAService` path as an interactive answer:

```text
question -> hybrid retrieval -> final reranked evidence snapshot
         -> DeepSeek answer -> citations
         -> deterministic checks -> Qwen Judge
```

The answer model receives only the final reranked evidence.

## Test set lifecycle

1. Create a draft test set for one knowledge base.
2. Add cases manually, import UTF-8 JSON/CSV, or ask Qwen to generate candidates from sampled indexed chunks.
3. Review each case and mark it `APPROVED`.
4. Publish the version. Published versions are immutable.
5. Clone a published set to create the next editable version.

AI-generated cases are always drafts. The system, rather than the model, writes the evidence locator. Generated numeric answers are accepted only when every numeric token occurs in the sampled source chunk. Cases are not filtered according to whether the current retriever can find them.

Stable evidence locators may contain document SHA256, filename, page, comparison key, content hash, and chunk ID. Chunk ID is useful for one ingestion, while the other fields preserve identity after re-ingestion.

JSON imports accept either a list or `{ "cases": [...] }`. CSV imports use the same field names; `required_evidence`, `scope`, and `tags` are JSON-encoded cells.

## Run lifecycle

Creating a run freezes:

- test set version;
- retrieval mode and all top-k/RRF settings;
- embedding, reranker, answer, and judge model names;
- ready document IDs, filenames, and source SHA256 values.

The worker claims pending runs with PostgreSQL row locking and writes each case result as it completes. One failed case does not discard earlier results. A run stores the complete vector, lexical, RRF, and reranked trace plus the exact final evidence sent to DeepSeek.

## Metrics

Deterministic metrics:

- Hit@1, Hit@3, Hit@5, evidence recall at 6, and MRR;
- exact number, unit, and period token preservation;
- expected versus actual insufficient-evidence status;
- citation precision, recall, and page accuracy;
- answer-language match.

Qwen Judge separately scores correctness, completeness, faithfulness, and language match. Deterministic financial checks remain visible and are not replaced by the Judge score. Provider failures count as failed cases in aggregate metrics. Retrieval metrics are not applicable to deliberately unanswerable cases with no required evidence.

Failure stages identify the first useful diagnosis: document scope, first-stage retrieval, RRF, reranking, numeric/unit/period errors, citation errors, false sufficient/insufficient answers, hallucination, judge failure, or another provider failure.

## API

```text
GET  /api/evaluation/knowledge-bases/{knowledge_base_id}/datasets
POST /api/evaluation/datasets
GET  /api/evaluation/datasets/{dataset_id}
POST /api/evaluation/datasets/{dataset_id}/cases
PATCH /api/evaluation/cases/{case_id}
POST /api/evaluation/datasets/{dataset_id}/import
POST /api/evaluation/datasets/{dataset_id}/generate
POST /api/evaluation/datasets/{dataset_id}/publish
POST /api/evaluation/datasets/{dataset_id}/clone

POST /api/evaluation/runs
GET  /api/evaluation/knowledge-bases/{knowledge_base_id}/runs
GET  /api/evaluation/runs/{run_id}
```

## POC boundaries

- Runs are processed sequentially within each worker process; scale workers to run multiple evaluations concurrently.
- Reports return the complete trace and are intended for POC-sized test sets. Pagination/export should be added before using thousands of cases in one run.
- AI generation is evidence-grounded but still requires human approval.
- The current negative-case generator creates an explicitly unsupported synthetic question when requested; it does not claim broad adversarial coverage.
