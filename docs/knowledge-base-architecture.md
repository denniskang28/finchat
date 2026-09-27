# Knowledge Base Architecture

This remains a retrieval POC. It supports multi-document ingestion and retrieval, but does not generate answers.

## Data flow

1. `POST /api/knowledge-bases/{id}/documents` validates and stores a PDF, records its SHA-256 and finance metadata, creates a durable `PENDING` job, and returns HTTP 202.
2. A worker atomically claims one job with `FOR UPDATE SKIP LOCKED`. Multiple worker replicas may run concurrently without processing the same job.
3. An ingestion job runs the common page parser, writes all inspectable chunks, embeds indexable chunks, and moves the document through `PROCESSING`, `PARSED`, `INDEXING`, and `READY`.
4. Failed jobs retry up to `WORKER_MAX_ATTEMPTS`. Parsing failure marks the document `FAILED`; a reindex failure leaves an already parsed document `READY` and records the failure on the job.
5. Retrieval resolves a document, explicit document list, or knowledge base to ready document IDs before running lexical and vector search in PostgreSQL.

## Retrieval scope

`POST /api/retrieval/search` requires exactly one scope:

- `document_id` for a single file;
- `document_ids` for an explicit comparison set, up to 500 documents;
- `knowledge_base_id` for cross-document search.

Knowledge-base searches may filter by `company`, `fiscal_year`, and `document_type`. The pipeline remains lexical top 20 plus vector top 20, RRF top 20, rerank, then final top 6. Multi-document final results allow at most three hits per document to avoid one report consuming all six positions. Production mode excludes anonymous semantic rows such as `Row 1`, because they have a value but no usable financial entity label; the Baseline and Semantic experiment modes remain unchanged.

## Storage and indexes

- `knowledge_bases` owns logical corpora.
- `documents` stores the knowledge-base foreign key, content hash, company, fiscal year, type, language, parse state, and parser metadata.
- `ingestion_jobs` is the durable queue and audit trail.
- `chunks` stores source content, raw and semantic alternatives, generated `tsvector`, and 1024-dimensional pgvector embeddings.
- PostgreSQL uses a GIN index for lexical retrieval, HNSW for cosine search, and B-tree indexes for knowledge-base and finance metadata filtering.

## POC boundaries

The design can ingest thousands of reports by adding worker replicas, but production deployment still needs object storage instead of a shared local volume, schema migrations instead of startup DDL, authentication and tenant isolation, dead-letter/requeue controls, observability, rate limiting, backup policy, and corpus-scale load/evaluation tests. Exact filtered ANN recall must also be measured on the target corpus; if it degrades, partitioning or iterative pgvector scans should be tuned before changing databases.
