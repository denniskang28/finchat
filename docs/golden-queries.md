# Semantic Table RAG POC - Golden Queries

## 1. Purpose

These queries test the inspected PDF only. They cover lost table context, semantic paraphrases, multilingual retrieval, multi-level headers, missing values, multi-row evidence, and the targeted page 93 chart adapter.

For `RAW_ROW` and `SEMANTIC_ROW`, the expected target is a `comparison_key`. For `MARKDOWN`, the expected target is the containing `table_id`. Unless noted otherwise, retrieval succeeds when all required targets appear in the top 3 after reranking.

## 2. Golden set

| ID | Query | Expected answer | Page | Required target(s) | What it tests |
|---|---|---|---:|---|---|
| G01 | 2025年底AIA美国公司债有多少，占公司债组合多少？ | USD 6.2 billion and 22%. | 91 | `p91_corporate_bonds_geography:united_states` | Chinese paraphrase; amount and share whose meanings are absent from the raw row. |
| G02 | As of 31 Dec 2025, how much of the non-par and surplus corporate bond portfolio was in Asia Pacific? | USD 17.3 billion, or 63% of the total. | 91 | `p91_corporate_bonds_geography:asia_pacific` | Exact scope/date plus amount/share retrieval. |
| G03 | 除亚太和美国外，其他地区的公司债规模和占比是多少？ | USD 4.2 billion and 15%. | 91 | `p91_corporate_bonds_geography:other` | Semantic interpretation of the row label `Other`. |
| G04 | How much of the corporate bond portfolio was allocated to banks in the financial sector? | USD 5.2 billion, or 19% of the total. | 91 | `p91_corporate_bonds_sector:financials_banks` | Semantic paraphrase of the source label `Financials - Banks`. |
| G05 | 通信服务和公用事业公司债各有多少？分别占比多少？ | Communication Services: USD 1.9 billion and 7%; Utilities: USD 1.9 billion and 7%. | 91 | `p91_corporate_bonds_sector:communication_services`; `p91_corporate_bonds_sector:utilities` | Multi-row retrieval and equal values without collapsing distinct labels. |
| G06 | What were Mainland China's risk discount rate and risk premium as at 31 Dec 2025? | Risk discount rate 8.30%; risk premium 5.60%. | 70 | `p70_risk_rates:mainland_china` | Multi-level date/measure header flattening. |
| G07 | 香港在2010年11月30日和2025年12月31日的风险溢价分别是多少？ | 4.47% and 4.45%, respectively. | 70 | `p70_risk_rates:hong_kong` | Chinese query, two periods, same measure. |
| G08 | Which market had no disclosed 2010 values but a 14.70% risk discount rate in 2025, and what was its 2025 risk premium? | Sri Lanka; its 2025 risk premium was 4.70%. | 70 | `p70_risk_rates:sri_lanka` | Preservation of `n/a`, period disambiguation, and row footnote marker. |
| G09 | 2025年底AIA China投资资产中股票占比是多少？ | 18%. | 93 | `p93_china_allocation:equities` | Targeted chart-series extraction and Chinese semantic retrieval. |
| G10 | What share of AIA China's invested assets was in corporate bonds, and what does that category include? | 5%; it includes less than 1% in loans and deposits. | 93 | `p93_china_allocation:corporate_bonds` | Chart label/value association plus category-specific footnote retrieval. |

## 3. Containing table IDs

| Target prefix | Markdown target |
|---|---|
| `p91_corporate_bonds_geography:*` | `p91_corporate_bonds_geography` |
| `p91_corporate_bonds_sector:*` | `p91_corporate_bonds_sector` |
| `p70_risk_rates:*` | `p70_risk_rates` |
| `p93_china_allocation:*` | `p93_china_allocation` |

Page 93's Markdown baseline is a normalized two-column series generated from the targeted chart adapter. It must be labeled `source_kind=CHART`; it is not evidence that the source contained a table.

## 4. Evaluation record

Record one row per query and representation:

```text
query_id
mode
expected_target_ids
first_relevant_rank
all_required_targets_in_top_1
all_required_targets_in_top_3
all_required_targets_in_top_5
vector_ranks
lexical_ranks
rrf_ranks
rerank_ranks
answer_correct
citation_page_correct
failure_stage
notes
```

`G05` requires both rows within K. Page 93 failures must be classified as chart-adapter failures rather than table-parser failures.

## 5. Answer grading

- Exact numeric values, units, periods, and row labels are required.
- Equivalent currency wording such as `$6.2bn` and `USD 6.2 billion` is accepted.
- Percentage-point and percent changes are not interchangeable.
- A correct answer with the wrong page or unsupported evidence is a failure.
- An answer must not fill `n/a` with zero or infer a missing value.
- Rounding notes and footnotes are required only when the question asks for them or they change the meaning of the value.
