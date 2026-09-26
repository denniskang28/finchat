# Semantic Table RAG POC - Retrieval Design

## 1. Retrieval contract

Input is a question, one or more document IDs, and one or more requested experimental modes. Each mode runs independently with exactly one table representation. Output is an inspectable sequence of lexical results, vector results, RRF results, reranked results, and the evidence selected for answer generation.

Default settings:

```text
VECTOR_TOP_K=20
LEXICAL_TOP_K=20
MERGED_TOP_K=20
RERANK_TOP_K=6
ANSWER_TOP_K=6
RRF_K=60
ENABLE_QUERY_REWRITE=false
```

## 2. Alibaba Cloud provider interfaces

Business services depend on these protocols, not on an SDK or HTTP response shape. Model IDs and region-specific base URLs are configuration values.

```python
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatResult(BaseModel):
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    provider_request_id: str | None


class EmbeddingResult(BaseModel):
    vectors: list[list[float]]
    model: str
    dimensions: int
    input_tokens: int | None
    provider_request_id: str | None


class RerankDocument(BaseModel):
    id: str
    text: str


class RerankItem(BaseModel):
    document_id: str
    source_index: int
    score: float
    rank: int


class RerankResult(BaseModel):
    items: list[RerankItem]
    model: str
    provider_request_id: str | None


class ChatProvider(Protocol):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult: ...


class EmbeddingProvider(Protocol):
    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        dimensions: int,
    ) -> EmbeddingResult: ...

    async def embed_query(
        self,
        query: str,
        *,
        dimensions: int,
    ) -> EmbeddingResult: ...


class RerankProvider(Protocol):
    async def rerank(
        self,
        query: str,
        documents: Sequence[RerankDocument],
        *,
        top_n: int,
    ) -> RerankResult: ...
```

Provider requirements:

- `AlibabaChatProvider` uses the OpenAI-compatible Chat Completions interface.
- `AlibabaEmbeddingProvider` keeps document and query methods separate so task instructions can differ. It verifies vector count and dimension before returning.
- `AlibabaRerankProvider` sends candidate text in input order and maps returned indices back to stable chunk IDs. It must reject missing, duplicate, or out-of-range indices.
- Providers use bounded timeouts, limited retries for transient failures, and surface provider request IDs. They do not silently substitute models.
- The initial configured models are `qwen3.8-flash`, `qwen3.7-text-embedding` at 1024 dimensions, and `qwen3-rerank`, subject to availability in the selected Alibaba Cloud region.

Official references:

- [OpenAI-compatible chat](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)
- [Text embedding](https://www.alibabacloud.com/help/en/model-studio/embedding)
- [Text rerank](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/text-rerank-api)

## 3. Experimental retrieval modes

| Mode | Included table chunks | Common chunks |
|---|---|---|
| `raw` | `representation=RAW_ROW` | `TEXT` |
| `markdown` | `representation=MARKDOWN` | `TEXT` |
| `semantic` | `representation IN (SEMANTIC_ROW, SUMMARY)` | `TEXT` |

All modes use the same document filter, query, embedding model, distance metric, candidate counts, RRF constant, reranker, and final top K. A comparison run must not mix table representations.

## 4. Hybrid retrieval algorithm

### 4.1 Query preparation

- Trim and Unicode-normalize the original question.
- Retain the original question for embedding and reranking.
- Query rewrite is off for the initial experiment. If later enabled, return both strings and use the rewritten query for both first-stage channels; reranking still uses the original question.
- Apply `document_id` and representation filters before ranking.

Chinese questions against an English report may produce no useful PostgreSQL lexical matches under the `simple` configuration. That is an expected, visible channel result, not a reason to hide the lexical stage or silently translate the query.

### 4.2 Lexical search

Use PostgreSQL full-text search over `chunks.search_vector`:

```sql
WITH q AS (
  SELECT websearch_to_tsquery('simple'::regconfig, :query) AS query
)
SELECT c.id,
       ts_rank_cd(c.search_vector, q.query) AS lexical_score
FROM chunks c, q
WHERE c.document_id = ANY(:document_ids)
  AND c.representation = ANY(:allowed_representations)
  AND c.search_vector @@ q.query
ORDER BY lexical_score DESC, c.id
LIMIT 20;
```

Assign 1-based `lexical_rank`. Do not normalize this score together with vector similarity; only rank positions enter RRF.

### 4.3 Vector search

Embed the query once and use cosine distance:

```sql
SELECT c.id,
       1 - (c.embedding <=> :query_embedding) AS vector_score
FROM chunks c
WHERE c.document_id = ANY(:document_ids)
  AND c.representation = ANY(:allowed_representations)
  AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> :query_embedding, c.id
LIMIT 20;
```

Assign 1-based `vector_rank`. `vector_score` is reported as cosine similarity, so higher is better.

### 4.4 Reciprocal Rank Fusion

Union candidates by chunk ID and compute:

```text
rrf_score(chunk) =
    (1 / (60 + vector_rank)  if present else 0)
  + (1 / (60 + lexical_rank) if present else 0)
```

Sort by descending `rrf_score`, then best available component rank, then chunk ID for deterministic ties. Keep the top 20 and assign 1-based `rrf_rank`.

RRF deliberately combines ranks rather than incomparable raw lexical and vector scores.

### 4.5 Rerank and evidence selection

- Send the original question and the top 20 fused chunk `content` values to the reranker.
- Preserve the chunk-ID-to-provider-index map.
- Sort by returned relevance score, then source index for deterministic ties.
- Keep the top 6 as `reranked_results` and `llm_evidence`.
- Do not use a score threshold until golden-query distributions are observed; fixed top K is easier to compare in the POC.
- The answer model receives only the question and these evidence chunks, never the full PDF.

If reranking is disabled, `llm_evidence` is the first six RRF results and every rerank field is null. If a configured retrieval channel fails, the request fails with its stage identified; it does not silently degrade.

## 5. Retrieval debug API

### Request

```http
POST /api/retrieval/debug
Content-Type: application/json
```

```json
{
  "document_ids": ["b23ffb88-716b-4654-953c-b1605c8f730b"],
  "query": "2025年底AIA美国公司债有多少，占公司债组合多少？",
  "modes": ["raw", "markdown", "semantic"],
  "include_content": true,
  "generate_answer": false
}
```

Running all three modes in one request ensures identical inputs and configuration. `generate_answer` defaults to false because retrieval evaluation is the primary purpose of this endpoint.

### Response

```json
{
  "request_id": "0c40bded-0195-4d04-a82f-871a72504e5e",
  "original_query": "2025年底AIA美国公司债有多少，占公司债组合多少？",
  "rewritten_query": null,
  "document_ids": ["b23ffb88-716b-4654-953c-b1605c8f730b"],
  "config": {
    "embedding_model": "qwen3.7-text-embedding",
    "embedding_dimensions": 1024,
    "rerank_model": "qwen3-rerank",
    "vector_top_k": 20,
    "lexical_top_k": 20,
    "merged_top_k": 20,
    "rerank_top_k": 6,
    "rrf_k": 60
  },
  "runs": [
    {
      "mode": "semantic",
      "filters": {
        "representations": ["SEMANTIC_ROW", "SUMMARY", "TEXT"]
      },
      "vector_results": [
        {
          "chunk_id": "...",
          "rank": 1,
          "score": 0.8421,
          "page_number": 91,
          "chunk_type": "TABLE_ROW",
          "representation": "SEMANTIC_ROW",
          "comparison_key": "p91_t1:united_states",
          "table_title": "Corporate Bonds by Geography",
          "row_label": "United States",
          "content": "Table: Corporate Bonds by Geography...",
          "raw_content": "United States | 6.2 | 22%",
          "semantic_content": "Table: Corporate Bonds by Geography..."
        }
      ],
      "lexical_results": [],
      "rrf_results": [
        {
          "chunk_id": "...",
          "rank": 1,
          "score": 0.016393,
          "vector_rank": 1,
          "lexical_rank": null
        }
      ],
      "reranked_results": [
        {
          "chunk_id": "...",
          "rank": 1,
          "score": 0.947,
          "rrf_rank": 1
        }
      ],
      "llm_evidence": [
        {
          "chunk_id": "...",
          "final_rank": 1,
          "page_number": 91,
          "content": "Table: Corporate Bonds by Geography..."
        }
      ],
      "answer": null,
      "citations": [],
      "latency_ms": {
        "embedding": 0,
        "vector": 0,
        "lexical": 0,
        "fusion": 0,
        "rerank": 0,
        "answer": null,
        "total": 0
      },
      "warnings": []
    }
  ]
}
```

Every full hit in every stage uses the `RetrievedChunk` fields defined in `data-model.md`; abbreviated objects above avoid repeating unchanged fields. Scores are null when a chunk did not occur in that stage. Stage arrays preserve the ranking returned at that stage rather than being overwritten by final order.

## 6. Evaluation rules

- A target row hit is matched by `comparison_key`, not by generated text equality.
- A Markdown hit is matched by `table_id` because one chunk represents the whole table.
- Multi-row questions list all required comparison keys and count as a hit only when all required rows appear within K.
- Report Hit@1, Hit@3, Hit@5, MRR, answer accuracy, and citation/page accuracy for each mode.
- Preserve the raw stage outputs for every failed query so parser, representation, retrieval, rerank, and answer errors can be classified separately.
- Primary success criterion: `SEMANTIC_ROW` table-row Hit@3 materially exceeds `RAW_ROW` and whole-table `MARKDOWN` on the fixed golden set without reducing citation/page accuracy.
