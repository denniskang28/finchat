# Minimal QA evaluation

Run date: 27 September 2026

Configuration:

- Retrieval mode: `PRODUCTION`
- Scope: AIA Group 2025 Annual Results Analyst Presentation
- Retrieval: Alibaba Cloud embeddings and reranker
- Answer model: `deepseek-chat`
- Answer context: persisted final reranked evidence only, maximum six chunks

## Golden results

All ten golden questions produced the expected answer, retained the requested language and financial units/periods, and cited the expected PDF page.

| Query | Required target rank(s) | Answer | Citation page | Result |
|---|---:|---|---:|---|
| G01 | 1 | USD 6.2 billion; 22% | 91 | Pass |
| G02 | 1 | USD 17.3 billion; 63% | 91 | Pass |
| G03 | 1 | USD 4.2 billion; 15% | 91 | Pass |
| G04 | 1 | USD 5.2 billion; 19% | 91 | Pass |
| G05 | 1, 2 | USD 1.9 billion and 7% for each row | 91 | Pass |
| G06 | 1 | 8.30%; 5.60% | 70 | Pass |
| G07 | 1 | 4.47%; 4.45% | 70 | Pass |
| G08 | 1 | Sri Lanka; 4.70% | 70 | Pass |
| G09 | 1 | 18% | 93 | Pass |
| G10 | 3 | 5%; includes less than 1% in loans and deposits | 93 | Pass |

Answer accuracy: 10/10. Citation-page accuracy: 10/10. All required targets occurred in the final reranked evidence: 10/10.

## Insufficient-evidence check

The unsupported question `What was AIA Mars revenue in FY2025?` returned `insufficient_evidence=true`, explicitly stated that the supplied evidence did not contain the requested value, and returned no citations.

These results validate this fixed sample and configuration only; they are not a general accuracy claim for arbitrary financial documents.
