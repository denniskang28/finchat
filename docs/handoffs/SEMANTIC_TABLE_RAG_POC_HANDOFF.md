# Semantic Table RAG POC — Codex Handoff

> 目标：快速验证“PDF 表格 → 语义化 Markdown/Chunk → Hybrid RAG → QA”这条路线是否足够准确，先不建设完整 Table Store / Financial Fact Store。
>
> 本文档是一个 **POC / Validation Handoff**，重点是最基本核心逻辑、可观察性和可验证性，而不是企业级完整平台。

---

## 1. POC 目标

实现一个最小可运行系统：

1. 用户上传 PDF 财报。
2. 系统解析 PDF 中的普通文本和表格。
3. 对表格生成两种表示：
   - 原始 Markdown 表格；
   - Context-Enriched Semantic Rows。
4. 对文本 Chunk 和 Semantic Table Row 做 embedding。
5. 存入 PostgreSQL + pgvector。
6. 用户通过简单 Chat UI 提问。
7. 系统使用 Hybrid Retrieval：
   - PostgreSQL lexical search；
   - pgvector dense vector search；
   - Alibaba Cloud reranker。
8. 将 Top-K evidence 交给 Alibaba Cloud Qwen 模型生成答案。
9. UI 不仅显示答案，还必须显示：
   - 检索到的 chunk；
   - chunk 类型；
   - source PDF；
   - page；
   - table title；
   - semantic row；
   - lexical/vector/rerank score；
   - 最终送给 LLM 的 evidence。
10. 通过 AIA 2025 Annual Results PDF 验证这种方案是否能准确回答表格问题。

---

## 2. 本 POC 明确不做什么

为了快速验证核心思路，本阶段 **不要** 做：

- Financial Fact Store
- 完整财务 ontology
- 复杂 Query Planner DAG
- Agent / ReAct
- 多知识库权限
- SSO / RBAC
- 多租户
- Elasticsearch / OpenSearch
- Kafka / MQ
- 微服务拆分
- 高可用
- 完整 OCR 平台
- 所有复杂图表 100% 结构化
- 跨 10 年 / 20 个市场的大规模 OLAP 分析

本阶段只验证：

> **Context-Enriched Table RAG 是否能够显著提高财报表格问题的 retrieval 和 answer accuracy。**

---

## 3. 核心技术假设

传统做法：

```text
PDF
 ↓
Markdown
 ↓
Chunk
 ↓
Embedding
 ↓
RAG
```

问题是原始表格：

```text
| Geography | $b | % |
|---|---:|---:|
| United States | 6.2 | 22 |
```

如果被拆成普通文本：

```text
United States 6.2 22
```

`6.2` 和 `22` 的业务含义非常弱。

本 POC 要验证的是：

```text
PDF Table
   ↓
Structured Table
   ↓
Context-Enriched Semantic Rows
   ↓
Embedding + Lexical Search
```

例如生成：

```text
Table: Corporate Bonds by Geography.
Portfolio: Non-par and Surplus Assets.
As of: 31 Dec 2025.

Geography: United States.
Corporate bond amount: USD 6.2 billion.
Share of total corporate bond portfolio: 22%.
```

每一个 Semantic Row 都是一个 **self-contained semantic record**。

---

## 4. 推荐技术栈

### Backend

- Python 3.12+
- FastAPI
- SQLAlchemy 2.x
- Alembic
- Pydantic v2

### Database

- PostgreSQL 16+
- pgvector

一个数据库即可完成：

- document metadata
- chunks
- metadata JSONB
- dense vectors
- lexical search

不引入 OpenSearch。

### Frontend

推荐：

- React / Next.js
- Tailwind CSS

UI 重点不是美观，而是方便验证：

```text
Answer
Evidence
Retrieved Chunks
Scores
Raw/Semantic Table
```

### PDF parsing

POC 第一版：

- PyMuPDF：读取 PDF、分页、文本、页面渲染；
- 表格 first-pass 可选 Docling 或 pdfplumber；
- 复杂表格解析失败时，可选 Alibaba Cloud Qwen Vision / OCR 作为 fallback。

**不要为了 POC 先开发完整 PDF parser。**

---

## 5. Alibaba Cloud Model Studio

模型名全部通过配置注入，不写死。

推荐初始配置：

```env
ALI_CHAT_MODEL=qwen3.8-flash
ALI_EMBED_MODEL=qwen3.7-text-embedding
ALI_RERANK_MODEL=qwen3-rerank
```

用途：

```text
qwen3.8-flash
→ Answer generation / optional query rewrite

qwen3.7-text-embedding
→ Dense embedding

qwen3-rerank
→ Retrieval reranking
```

可选：

```env
ALI_VISION_MODEL=qwen3-vl-plus
```

用于后续复杂 table / chart fallback。

> 模型可用性依 region 而变化，运行时以 Alibaba Cloud Model Studio 控制台为准。

建议 provider abstraction：

```python
class ChatModel(Protocol):
    async def complete(...): ...

class EmbeddingModel(Protocol):
    async def embed(...): ...

class Reranker(Protocol):
    async def rerank(...): ...
```

不要把 DashScope 调用散布到业务代码中。

---

## 6. Alibaba API 集成原则

Chat model 优先使用 OpenAI-compatible client：

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=settings.alibaba_api_key,
    base_url=settings.alibaba_base_url,
)
```

Embedding / Rerank 可直接使用 DashScope API，以使用对应 embedding/reranking 能力。

配置：

```env
ALIBABA_API_KEY=
ALIBABA_BASE_URL=
ALIBABA_WORKSPACE_ID=
ALIBABA_REGION=
```

不要在代码中固定 region URL。

---

## 7. POC 总体架构

```text
                   ┌──────────────────────┐
                   │       React UI       │
                   │ Upload / Chat / Debug│
                   └──────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    FastAPI      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       PDF Ingestion                    QA Pipeline
              │                             │
              ▼                             ▼
       PDF Parser                    Optional Rewrite
              │                             │
        ┌─────┴──────┐               Hybrid Retrieve
        ▼            ▼                      │
      Text          Table                   ▼
        │            │                    Rerank
        │            ▼                      │
        │     Original Markdown             ▼
        │            │                Evidence Pack
        │            ▼                      │
        │      Semantic Rows                ▼
        └──────┬─────┘                 Qwen Answer
               ▼
        Chunk + Embedding
               │
               ▼
       PostgreSQL + pgvector
```

---

## 8. 数据模型

POC 不建 `table_cell` / `table_row` relational tables。

只需要：

```text
documents
chunks
```

### 8.1 documents

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    title TEXT,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL,
    page_count INT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 8.2 chunks

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id UUID PRIMARY KEY,

    document_id UUID NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    page_number INT NOT NULL,
    chunk_type TEXT NOT NULL,

    content TEXT NOT NULL,
    raw_content TEXT,
    semantic_content TEXT,

    metadata JSONB NOT NULL DEFAULT '{}',

    embedding vector(1024),
    search_vector tsvector,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`chunk_type` 初始只支持：

```text
TEXT
TABLE_SUMMARY
TABLE_ROW
TABLE_MARKDOWN
```

后续可增加：

```text
CHART
CHART_SERIES
FOOTNOTE
```

---

## 9. Chunk Metadata

每个 table row 必须保留足够 metadata：

```json
{
  "element_type": "TABLE_ROW",
  "table_id": "p91_t1",
  "table_title": "Corporate Bonds by Geography",
  "table_context": {
    "portfolio": "Non-par and Surplus Assets",
    "as_of_date": "2025-12-31"
  },
  "row_label": "United States",
  "source": {
    "page": 91
  }
}
```

第一版不要过度 schema 化，`metadata JSONB` 足够。

---

## 10. Table 解析后的内部结构

parser 输出统一 JSON：

```json
{
  "table_id": "p91_t1",
  "page": 91,
  "title": "Corporate Bonds by Geography",
  "context": [
    "Non-par and Surplus Assets",
    "As of 31 Dec 2025"
  ],
  "headers": [
    {
      "key": "geography",
      "label": "Geography"
    },
    {
      "key": "amount",
      "label": "Amount",
      "semantic_label": "Corporate bond amount",
      "unit": "USD billion"
    },
    {
      "key": "share",
      "label": "% of total",
      "semantic_label": "Share of total corporate bond portfolio",
      "unit": "%"
    }
  ],
  "rows": [
    {
      "geography": "Asia Pacific",
      "amount": 17.3,
      "share": 63
    },
    {
      "geography": "United States",
      "amount": 6.2,
      "share": 22
    }
  ]
}
```

这个 JSON 可以直接放 `TABLE_MARKDOWN` chunk metadata，POC 不需要单独建关系表。

---

## 11. Original Markdown

每张表保留完整 Markdown：

```markdown
### Corporate Bonds by Geography

Portfolio: Non-par and Surplus Assets  
As of: 31 Dec 2025

| Geography | Amount (USD bn) | % of total |
|---|---:|---:|
| Asia Pacific | 17.3 | 63% |
| United States | 6.2 | 22% |
| Other | 4.2 | 15% |
| Total | 27.7 | 100% |
```

生成：

```text
chunk_type = TABLE_MARKDOWN
```

主要用于：

- Debug；
- Evidence display；
- Answer context fallback。

---

## 12. Context-Enriched Semantic Row

这是本 POC **最核心代码**。

不要：

```text
United States | 6.2 | 22
```

而要：

```text
Table: Corporate Bonds by Geography.
Portfolio: Non-par and Surplus Assets.
As of: 31 Dec 2025.

Geography: United States.
Corporate bond amount: USD 6.2 billion.
Share of total corporate bond portfolio: 22%.
```

然后：

```text
chunk_type = TABLE_ROW
```

每一行独立 embedding。

---

## 13. Semantic Row Generator

第一版尽量规则化，不要每一行调用 LLM。

接口：

```python
class SemanticTableRenderer:

    def render_table_summary(
        self,
        table: ParsedTable,
    ) -> str:
        ...

    def render_row(
        self,
        table: ParsedTable,
        row: dict,
    ) -> str:
        ...
```

简单实现：

```python
def render_row(table, row):

    parts = [f"Table: {table.title}."]

    for ctx in table.context:
        parts.append(ctx.rstrip(".") + ".")

    for header in table.headers:

        value = row.get(header.key)

        if value is None:
            continue

        label = header.semantic_label or header.label

        if header.unit:
            parts.append(
                f"{label}: {value} {header.unit}."
            )
        else:
            parts.append(
                f"{label}: {value}."
            )

    return "\n".join(parts)
```

重点不是实现复杂 NLP，而是让数值具备 row/column/table/date/unit 上下文。

---

## 14. semantic_label 如何产生

### Mode A — 规则优先

例如：

```text
Table title = Corporate Bonds by Geography
Column = $b
```

组合成：

```text
Corporate bond amount
```

### Mode B — 每张复杂表调用一次 Qwen

如果表头不容易理解：

```text
$bn
2025
%
```

则把以下内容一次性发给 Qwen：

```text
table title
headers
context
前2-3行
```

要求返回：

```json
{
  "columns": [
    {
      "key": "amount",
      "semantic_label": "Corporate bond amount",
      "unit": "USD billion"
    }
  ]
}
```

注意：

> 不要每 cell / 每 row 调一次 LLM。

最多每张复杂表一次。

---

## 15. 普通文本 Chunk

普通 narrative 文本正常 chunk。

推荐：

```text
500–900 tokens
overlap 80–120 tokens
```

metadata：

```json
{
  "element_type": "TEXT",
  "page": 25
}
```

不要把表格和普通 narrative 强行拼在同一个 chunk。

---

## 16. Embedding

参与 retrieval 的：

```text
TEXT
TABLE_SUMMARY
TABLE_ROW
```

全部生成 embedding。

POC 推荐：

```text
qwen3.7-text-embedding
dimension = 1024
```

数据库：

```sql
embedding vector(1024)
```

接口：

```python
class AlibabaEmbeddingService:

    async def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        ...

    async def embed_query(
        self,
        query: str
    ) -> list[float]:
        ...
```

文档与 query embedding 分接口，为后续 query instruction 留空间。

---

## 17. Lexical Search

不要纯 vector。

PostgreSQL：

```sql
UPDATE chunks
SET search_vector =
    to_tsvector(
        'simple',
        coalesce(semantic_content, content)
    );
```

GIN index：

```sql
CREATE INDEX idx_chunks_search
ON chunks
USING GIN(search_vector);
```

第一版使用 `simple`，避免 `VONB / OPAT / UFSG / 1H26 / AIA` 等专有词被过度词干化。

---

## 18. Vector Index

```sql
CREATE INDEX idx_chunks_embedding
ON chunks
USING hnsw (
    embedding vector_cosine_ops
);
```

---

## 19. Hybrid Retrieval

用户：

```text
2025年AIA美国公司债是多少，占比多少？
```

### Step 1 — optional query rewrite

可让 Qwen 生成短 retrieval query：

```text
AIA 2025 United States corporate bond amount portfolio share
```

POC 默认先关闭：

```env
ENABLE_QUERY_REWRITE=false
```

### Step 2 — Vector Search

```text
top 20
```

### Step 3 — Lexical Search

```text
top 20
```

### Step 4 — Merge

使用 Reciprocal Rank Fusion：

```python
score += 1 / (60 + rank)
```

合并后取：

```text
Top 20
```

### Step 5 — Rerank

发送：

```text
query + top 20 chunks
```

到：

```text
qwen3-rerank
```

最终取：

```text
Top 6
```

---

## 20. Retrieval 返回对象

不要只返回字符串。

```python
class RetrievedChunk(BaseModel):

    chunk_id: str

    document_id: str
    filename: str

    page_number: int
    chunk_type: str

    content: str
    raw_content: str | None

    metadata: dict

    vector_score: float | None
    lexical_score: float | None
    rrf_score: float | None
    rerank_score: float | None

    final_rank: int
```

这是 Debug UI 最重要的数据结构之一。

---

## 21. Answer Prompt

最终回答只使用：

```text
question
+
top evidence
```

不要把整个 PDF 给模型。

System Prompt：

```text
You are answering questions from financial-report evidence.

Use only the supplied evidence.

Rules:
1. Do not invent values.
2. Preserve units and reporting periods.
3. Distinguish FY, H1, H2 and point-in-time dates.
4. If the evidence is insufficient, say so.
5. Prefer exact values from table evidence.
6. If multiple pieces of evidence conflict, expose the conflict.
7. Do not silently annualise half-year data.
8. Keep the answer concise.
9. Refer to source pages where appropriate.
```

---

## 22. Answer API

```http
POST /api/chat
```

Request：

```json
{
  "document_ids": ["..."],
  "question": "2025年AIA美国公司债是多少，占比多少？",
  "debug": true
}
```

Response：

```json
{
  "answer": "截至2025年12月31日...",

  "citations": [
    {
      "document_id": "...",
      "page": 91,
      "chunk_id": "..."
    }
  ],

  "debug": {
    "original_query": "...",
    "rewritten_query": null,
    "vector_results": [],
    "lexical_results": [],
    "merged_results": [],
    "reranked_results": [],
    "llm_evidence": []
  }
}
```

---

## 23. Debug UI — 本 POC 的重点

UI 不要只做聊天框。

建议：

```text
┌───────────────────────────────────────────────────────┐
│ PDF Upload                                            │
├───────────────────────────────┬───────────────────────┤
│ Chat                          │ Retrieval Debug       │
│                               │                       │
│ Q: ...                        │ Original Query        │
│                               │ Rewritten Query       │
│ Answer                        │                       │
│                               │ [Top 1] TABLE_ROW     │
│ Sources: p91                  │ page: 91              │
│                               │ vector: ...           │
│                               │ lexical: ...          │
│                               │ RRF: ...              │
│                               │ rerank: ...           │
│                               │ content...            │
└───────────────────────────────┴───────────────────────┘
```

---

## 24. Retrieval Debug Panel

每个 hit 卡片显示：

```text
Final Rank
Chunk Type
Filename
Page
Table Title
Row Label
Vector Score
Lexical Score
RRF Score
Rerank Score
```

两个 tab：

```text
Semantic Content
Raw Content
```

例如：

### Semantic Content

```text
Table: Corporate Bonds by Geography.
Portfolio: Non-par and Surplus Assets.
As of: 31 Dec 2025.

Geography: United States.
Corporate bond amount: USD 6.2 billion.
Share of total corporate bond portfolio: 22%.
```

### Raw Content

```text
United States | 6.2 | 22%
```

这是验证 semantic enrichment 是否有效的关键 UI。

---

## 25. Retrieval Compare 模式

必须提供 A/B toggle：

```text
[ ] Semantic Table Enrichment
```

### OFF

索引：

```text
United States | 6.2 | 22
```

### ON

索引：

```text
Corporate Bonds by Geography...
United States...
Corporate bond amount...
22%...
```

最好允许页面并排显示 baseline 与 experiment 的排名。

本 POC 最重要的问题就是：

> Context-Enriched Table RAG 到底有没有明显提升？

---

## 26. Upload / Ingestion API

```http
POST /api/documents
multipart/form-data
file=<pdf>
```

返回：

```json
{
  "document_id": "...",
  "status": "PROCESSING"
}
```

POC 可以同步处理，不加 MQ。

```http
GET /api/documents/{id}
```

返回：

```json
{
  "status": "READY",
  "page_count": 99,
  "text_chunks": 240,
  "tables": 32,
  "table_row_chunks": 410
}
```

---

## 27. Table Debug API

```http
GET /api/documents/{document_id}/tables
```

返回：

```json
{
  "table_id": "p91_t1",
  "page": 91,
  "title": "...",
  "markdown": "...",
  "semantic_rows": []
}
```

前端增加 `Tables` tab，用来先验证 parser 是否正确。

---

## 28. Repository Structure

```text
semantic-table-rag-poc/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   └── debug.py
│   │   ├── config/
│   │   │   └── settings.py
│   │   ├── db/
│   │   │   ├── models.py
│   │   │   ├── session.py
│   │   │   └── migrations/
│   │   ├── ingestion/
│   │   │   ├── pdf_parser.py
│   │   │   ├── table_parser.py
│   │   │   ├── table_renderer.py
│   │   │   ├── text_chunker.py
│   │   │   └── ingestion_service.py
│   │   ├── providers/
│   │   │   └── alibaba/
│   │   │       ├── chat.py
│   │   │       ├── embedding.py
│   │   │       └── rerank.py
│   │   ├── retrieval/
│   │   │   ├── vector_search.py
│   │   │   ├── lexical_search.py
│   │   │   ├── rrf.py
│   │   │   ├── reranker.py
│   │   │   └── hybrid_retriever.py
│   │   ├── qa/
│   │   │   ├── prompts.py
│   │   │   └── qa_service.py
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app/
│   ├── components/
│   │   ├── ChatPanel.tsx
│   │   ├── RetrievalDebug.tsx
│   │   ├── ChunkCard.tsx
│   │   ├── UploadPanel.tsx
│   │   └── TableDebug.tsx
│   └── package.json
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 29. Docker Compose

只需要：

```text
postgres
backend
frontend
```

PostgreSQL：

```yaml
services:

  postgres:
    image: pgvector/pgvector:pg16

  backend:
    build: ./backend

  frontend:
    build: ./frontend
```

不要引入：

```text
Redis
Kafka
OpenSearch
MinIO
```

---

## 30. Feature Flags

```env
ENABLE_QUERY_REWRITE=false

ENABLE_LEXICAL_SEARCH=true
ENABLE_VECTOR_SEARCH=true
ENABLE_RERANK=true

ENABLE_SEMANTIC_TABLE_ROWS=true
ENABLE_LLM_TABLE_HEADER_MAPPING=true

DEBUG_RETRIEVAL=true
```

这样很容易做 A/B。

---

## 31. AIA 2025 PDF 验证范围

第一阶段只加载：

```text
AIA Group 2025 Annual Results Analyst Presentation
```

不要一开始加载几十份文档。

重点页面：

```text
Page 91 — Corporate Bonds by Geography
Risk Discount Rate / Risk Premium table
Page 93 — AIA China investment allocation（第二阶段）
```

---

## 32. Golden Case 1 — Table Retrieval

问题：

```text
2025年底AIA美国公司债有多少，占公司债组合多少？
```

正确 evidence：

```text
United States
$6.2bn
22%
```

成功标准：

```text
Top 3 中命中正确 TABLE_ROW
```

---

## 33. Golden Case 2 — Semantic Paraphrase

问题：

```text
AIA在美国配置了多少企业债？
```

原文使用：

```text
Corporate Bonds
United States
```

验证 embedding + rerank 是否仍能命中。

---

## 34. Golden Case 3 — Share Query

问题：

```text
美国债券在AIA公司债组合中占几成？
```

验证 semantic row 对中文 paraphrase 的 retrieval。

---

## 35. Golden Case 4 — Multi-level Table

问题：

```text
2025年底Mainland China的Risk Discount Rate是多少？
```

预期 semantic record：

```text
Market: Mainland China.
As of: 31 Dec 2025.
Risk Discount Rate: 8.30%.
```

这是验证多级 column header flattening 的核心案例。

---

## 36. Golden Case 5 — Chart / Page 93（第二阶段）

目标数据：

```text
Government & Government Agency Bonds 74%
Corporate Bonds 5%
Equities 18%
Real Estate 2%
Other 1%
```

问题：

```text
2025年底AIA China的股票投资占比是多少？
```

预期：

```text
18%
```

第一阶段若时间紧，可先不实现 chart。

---

## 37. A/B Validation

必须比较：

### Baseline

```text
Raw Table Markdown / Raw Row
```

### Experiment

```text
Context-Enriched Semantic Row
```

记录：

```text
Query
Expected Chunk
Baseline Rank
Semantic Rank
Baseline Rerank Score
Semantic Rerank Score
Answer Correct?
```

CSV 即可，不需要先做 evaluation 平台。

---

## 38. 最小 Evaluation 指标

只需要：

```text
Hit@1
Hit@3
Hit@5
MRR
Answer Accuracy
Citation/Page Accuracy
```

重点：

```text
TABLE_ROW Hit@3
```

---

## 39. Query Planner — POC 极简版

本次不要做完整 planner。

只做：

```text
User Query
   ↓
optional rewrite
   ↓
Hybrid Retrieval
   ↓
Rerank
   ↓
Answer
```

最多增加一个简单 question type：

```text
FACT
NARRATIVE
```

第一版甚至可以不做。

---

## 40. Query-time LLM 调用次数

默认：

```text
User
 ↓
Retrieval
 ↓
Rerank
 ↓
LLM Answer
```

只需要：

```text
1 次生成式 LLM
```

如果：

```env
ENABLE_QUERY_REWRITE=true
```

则：

```text
2 次生成式 LLM
```

Rerank / embedding 不算生成式 LLM。

---

## 41. Ingestion 阶段 LLM 调用

普通表：

```text
0 次
```

复杂表：

```text
最多每张表 1 次
```

用于 semantic header mapping。

禁止：

```text
每 row 调一次 LLM
每 cell 调一次 LLM
```

---

## 42. 可观察性

每个 query 记录：

```json
{
  "query": "...",
  "rewrite": null,
  "vector_top_k": 20,
  "lexical_top_k": 20,
  "merged_top_k": 20,
  "rerank_top_k": 6,
  "retrieved_chunk_ids": [],
  "final_evidence_ids": [],
  "latency": {
    "embedding_ms": 0,
    "vector_ms": 0,
    "lexical_ms": 0,
    "rerank_ms": 0,
    "llm_ms": 0,
    "total_ms": 0
  }
}
```

POC 最重要的是：

> 出错时能明确知道是 parser 错、retrieval 错、rerank 错，还是 answer LLM 错。

---

## 43. UI 必须区分 retrieval 四个阶段

显示：

```text
① Vector Results
② Lexical Results
③ RRF Merged Results
④ Reranked Results
```

不要只显示最终 Top 5。

否则无法验证 hybrid search 各组件价值。

---

## 44. Answer 与 Retrieval 解耦

提供两个按钮：

```text
Regenerate Answer
Retrieve Again
```

`Regenerate Answer` 只用当前 evidence 重新生成答案。

`Retrieve Again` 才重新检索。

这样能区分：

```text
evidence 对但答案错
```

与：

```text
evidence 本身没找对
```

---

## 45. Answer LLM 不得偷看整个 PDF

这是重要验证条件。

Answer LLM 只能收到：

```text
Top N chunks
```

不能把 full PDF 作为额外上下文。

否则无法验证 RAG 本身是否成功。

---

## 46. Minimum Acceptance Criteria

### Upload

- 可上传 AIA 2025 PDF；
- ingestion 后能查看 text/table chunk 数量。

### Table

- 正确解析 Page 91 Corporate Bonds 表；
- 正确解析至少一个 multi-level-header table；
- 能显示 original markdown；
- 能显示 semantic rows。

### Retrieval

- vector search 可运行；
- lexical search 可运行；
- RRF merge 可运行；
- qwen3-rerank 可运行。

### QA

- 可以提问；
- 返回 Qwen 生成答案；
- 返回 source page；
- UI 显示最终 evidence。

### Debug

必须显示：

- semantic chunk
- raw chunk
- vector score
- lexical score
- merged score
- rerank score
- rank
- page
- chunk type

### Validation

Golden queries 至少 10 条。

必须比较：

```text
raw table representation
vs
semantic table representation
```

的 Hit@3。

---

## 47. 推荐开发顺序

### Phase 1 — Skeleton

```text
FastAPI
Postgres + pgvector
React UI
Docker Compose
```

确认服务可启动。

### Phase 2 — PDF ingestion

```text
upload PDF
document metadata
page text extraction
basic text chunks
```

先不要做 QA。

### Phase 3 — Table parsing

只针对 sample PDF。

至少成功：

```text
Page 91
```

输出：

```text
ParsedTable
Original Markdown
Semantic Rows
```

先写单元测试。

### Phase 4 — Embedding

实现 Alibaba embedding provider。

将：

```text
TEXT
TABLE_SUMMARY
TABLE_ROW
```

写入 pgvector。

### Phase 5 — Retrieval

按顺序实现：

```text
vector
lexical
RRF
rerank
```

提供 debug API。

先不要生成答案。

### Phase 6 — Retrieval Debug UI

手工输入：

```text
2025年底AIA美国公司债有多少？
```

确认正确 chunk 在 Top 3。

### Phase 7 — QA

最后再增加：

```text
Qwen Answer
```

避免“答案看起来对，但 retrieval 实际错”的假象。

### Phase 8 — A/B

实现 raw vs semantic 对比，输出简单 evaluation report。

---

## 48. Coding Principles

- 不过度抽象；
- 不引入 Agent Framework；
- 不引入 LangGraph；
- 不引入 OpenSearch；
- 优先自己写少量 retrieval orchestration；
- Provider 与业务逻辑分离；
- 所有模型名配置化；
- 所有 prompt 放独立文件；
- 所有关键步骤可 debug；
- 不允许静默 fallback；
- parser 失败必须可见；
- retrieval score 必须保留；
- answer evidence 必须可见。

---

## 49. 建议不使用 LangChain

这个 POC 核心逻辑只有：

```text
parse
chunk
embed
retrieve
rerank
answer
```

直接实现更清楚。

推荐：

```text
FastAPI
+
SQLAlchemy
+
pgvector
+
Alibaba provider clients
```

---

## 50. 第一版允许的限制

允许：

- 只支持英文财报；
- PDF 不是纯扫描件；
- Table parser 只覆盖主要二维表；
- Chart 先不支持；
- 不保证所有 merged table 都正确；
- 不保证所有表格都能解析；
- 不支持跨大量文档的大规模分析；
- 不做完整 semantic planner。

因为本阶段目标不是生产系统。

---

## 51. 本 POC 最终要回答的五个问题

### Q1

Context-Enriched Table Row 是否比 raw table row 更容易被检索到？

### Q2

Hybrid Search 是否明显优于 pure vector search？

### Q3

Qwen reranker 是否改善 Top-3 precision？

### Q4

对于表格数字问题，最终 answer 是否可以稳定引用正确 page/chunk？

### Q5

如果不建设 Financial Fact Store，仅靠 Semantic Table RAG，能覆盖多少日常财报 QA？

只要这五个问题能得到清晰结论，这个 POC 就成功了。

---

## 52. Codex 第一条 Prompt

把本文档和 AIA 2025 PDF 放到 Codex workspace，然后使用：

```text
Read SEMANTIC_TABLE_RAG_POC_HANDOFF.md completely.

This is a proof-of-concept project. Do not expand the scope.

First inspect the attached AIA Group 2025 Annual Results PDF, especially:
- page 91 Corporate Bonds by Geography
- the Risk Discount Rate table
- page 93 AIA China investment allocation

Before coding:
1. write a short implementation plan;
2. confirm the minimum database schema;
3. define ParsedTable, Chunk and RetrievedChunk models;
4. define the Alibaba Cloud model provider interfaces;
5. define how raw table rows and context-enriched semantic rows will be generated;
6. define the retrieval debug payload;
7. define 10 golden queries.

Then implement in phases.

Do not introduce:
- Financial Fact Store
- agents
- LangGraph
- OpenSearch
- Redis
- Kafka
- microservices

The main experiment is:
Raw table representation
vs
Context-Enriched Semantic Table RAG.

The UI must expose retrieval internals:
- vector hits
- lexical hits
- merged hits
- reranked hits
- scores
- page
- raw content
- semantic content

Do not consider the POC successful merely because the final LLM answer looks correct.
The retrieved evidence must also be demonstrably correct.

Stop after the implementation plan and architecture skeleton for review before implementing the full application.
```

---

## 53. 推荐环境变量

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/semantic_rag

ALIBABA_API_KEY=
ALIBABA_BASE_URL=
ALIBABA_REGION=
ALIBABA_WORKSPACE_ID=

ALI_CHAT_MODEL=qwen3.8-flash
ALI_EMBED_MODEL=qwen3.7-text-embedding
ALI_RERANK_MODEL=qwen3-rerank

EMBEDDING_DIMENSION=1024

VECTOR_TOP_K=20
LEXICAL_TOP_K=20
MERGED_TOP_K=20
RERANK_TOP_K=6
ANSWER_TOP_K=6

ENABLE_QUERY_REWRITE=false
ENABLE_VECTOR_SEARCH=true
ENABLE_LEXICAL_SEARCH=true
ENABLE_RERANK=true
ENABLE_SEMANTIC_TABLE_ROWS=true
ENABLE_LLM_TABLE_HEADER_MAPPING=true

DEBUG_RETRIEVAL=true
```

---

## 54. 官方模型资料

实现时以 Alibaba Cloud Model Studio 当前 region 实际可用模型为准：

- Chat / OpenAI-compatible API:  
  https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope

- Embedding:  
  https://www.alibabacloud.com/help/en/model-studio/embedding

- Rerank:  
  https://www.alibabacloud.com/help/en/model-studio/rerank

- Model list:  
  https://www.alibabacloud.com/help/en/model-studio/models

---

## 55. 最终架构总结

本 POC 不验证：

```text
PDF → database fact model
```

而验证：

```text
PDF
 ↓
Table Extraction
 ↓
Original Table Markdown
 ↓
Context-Enriched Semantic Rows
 ↓
Hybrid Search
  ├── lexical
  └── vector
 ↓
RRF
 ↓
Qwen Rerank
 ↓
Top Evidence
 ↓
Qwen Answer
 ↓
Answer + Retrieval Debug
```

核心设计原则：

> **Do not make the table fit the database. Make every table row understandable outside the table.**

也就是：

> **不急着把所有 PDF 表格标准化成数据库事实，而是先把每一个表格数据点变成脱离原二维位置后仍然能够被语义理解和检索的 self-contained record。**

这就是本 POC 唯一需要重点验证的核心逻辑。
