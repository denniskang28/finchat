# Retrieval Evaluation

## Scope

This report evaluates retrieval only. No query rewrite or answer generation was used.

- Evaluation date: 2026-09-27
- Source: `AIA Group 2025 Annual Results Analyst Presentation EN (FINAL) - 27 Mar.pdf`
- Parsed document ID: `c2cafa84-af44-4fe0-af78-981ccce8d0be`
- Golden set: G01-G10 from `docs/golden-queries.md`
- Embedding: `qwen3.7-text-embedding`, 1,024 dimensions
- Reranker: `qwen3-rerank`
- Lexical: PostgreSQL `websearch_to_tsquery('simple', query)` and `ts_rank_cd`
- Dense: pgvector cosine similarity with an HNSW index
- Fusion: reciprocal rank fusion with `k=60`
- Pipeline: lexical 20 + vector 20 -> RRF 20 -> rerank -> final 6

## Experimental Arms

| Mode | Indexed retrieval candidates |
|---|---|
| BASELINE | 99 `TEXT` chunks + 427 `RAW_ROW` chunks |
| SEMANTIC | 99 `TEXT` chunks + 108 `TABLE_SUMMARY` chunks + 413 `SEMANTIC_ROW` chunks |

Every indexed chunk has a stored 1,024-dimensional vector. The modes use the same document, query, embedding provider, PostgreSQL lexical search, candidate limits, RRF constant, reranker, and final result count.

## Aggregate Results

Hit@K requires every required target for a query to appear within K. For G05, both Communication Services and Utilities must be present. MRR follows the standard first-relevant-result definition.

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| BASELINE | 0% | 0% | 0% | 0.00 |
| SEMANTIC | 50% | 100% | 100% | 0.80 |
| Absolute change | +50 pp | +100 pp | +100 pp | +0.80 |

## Per-Query Results

`Final rank` is the rank after reranking. A dash means that the required raw target did not survive first-stage retrieval and RRF. For G05, the displayed rank is the position at which all required targets have appeared.

| Query | Baseline final rank | Semantic vector rank(s) | Semantic final rank | Hit@3 |
|---|---:|---:|---:|---:|
| G01 | - | 10 | 2 | Yes |
| G02 | - | 2 | 1 | Yes |
| G03 | - | 1 | 1 | Yes |
| G04 | - | 1 | 1 | Yes |
| G05 | - | 1, 2 | 2 | Yes |
| G06 | - | 1 | 2 | Yes |
| G07 | - | 1 | 1 | Yes |
| G08 | - | 7 | 1 | Yes |
| G09 | - | 2 | 2 | Yes |
| G10 | - | 1 | 2 | Yes |

Semantic first-relevant reciprocal ranks were: G01 0.5, G02 1.0, G03 1.0, G04 1.0, G05 1.0, G06 0.5, G07 1.0, G08 1.0, G09 0.5, and G10 0.5.

## Interpretation

The result supports the POC hypothesis on this fixed sample. Raw rows such as `United States | 6.2 | 22%` omit the table title, portfolio scope, measure meanings, date, units, and footnotes. None of the ten required raw-row targets entered the fused top 20. The reranker therefore could not recover them.

Semantic rows restore the missing context directly in each independently retrievable chunk. All ten golden queries retrieved every required row in the final top three. Reranking materially helped the harder cases: G01 moved from vector rank 10 to final rank 2, and G08 moved from vector rank 7 to final rank 1.

The PostgreSQL lexical channel returned no required target rows for this set. Chinese queries against English rows are expected to have weak or empty lexical matches under the `simple` configuration. The English questions are long conjunctive web-search queries, while raw rows contain only labels and values. This behavior is visible in the debug payload rather than hidden by fallback logic.

## Limitations

- The test contains ten questions from one financial presentation; it demonstrates a strong directional result, not a general benchmark.
- The Baseline arm is intentionally strict and excludes table summaries. It measures context loss in independently indexed raw rows.
- Hit@K grades target retrieval, not answer correctness. No LLM answer was generated.
- Provider scores are model-specific and should not be compared as calibrated probabilities.
- The report reflects the current parsed artifact. Parser changes require rebuilding embeddings and rerunning the evaluation.
