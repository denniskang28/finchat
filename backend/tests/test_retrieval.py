from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.retrieval.context import enrich_semantic_content
from app.retrieval.service import (
    _candidate_filter,
    extract_query_years,
    focused_query,
    grade_targets,
    is_group_vonb_chunk,
    is_group_vonb_time_series,
    lexical_query_terms,
    reciprocal_rank_fusion,
)
from app.schemas import RetrievalRequest


def test_rrf_rewards_chunks_found_by_both_retrievers():
    shared = uuid4()
    lexical_only = uuid4()
    vector_only = uuid4()

    ordered, scores = reciprocal_rank_fusion(
        [[lexical_only, shared], [vector_only, shared]],
        k=60,
        limit=3,
    )

    assert ordered[0] == shared
    assert scores[shared] == 2 / 62
    assert scores[lexical_only] == 1 / 61
    assert scores[vector_only] == 1 / 61


def test_golden_multi_target_rank_requires_every_target():
    ranks, first_relevant_rank, complete_rank = grade_targets(
        {"p91_t2:communication_services": 1, "p91_t2:utilities": 4},
        ["p91_t2:communication_services", "p91_t2:utilities"],
    )

    assert ranks == {
        "p91_t2:communication_services": 1,
        "p91_t2:utilities": 4,
    }
    assert first_relevant_rank == 1
    assert complete_rank == 4


def test_golden_multi_target_rank_fails_when_one_target_is_missing():
    ranks, first_relevant_rank, complete_rank = grade_targets(
        {"p91_t2:communication_services": 1},
        ["p91_t2:communication_services", "p91_t2:utilities"],
    )

    assert ranks["p91_t2:utilities"] is None
    assert first_relevant_rank == 1
    assert complete_rank is None


@pytest.mark.parametrize(
    "scope",
    [
        {"document_id": uuid4()},
        {"document_ids": [uuid4(), uuid4()]},
        {"knowledge_base_id": uuid4()},
    ],
)
def test_retrieval_request_accepts_exactly_one_scope(scope):
    request = RetrievalRequest(query="What is the allocation?", **scope)

    assert request.mode == "PRODUCTION"


@pytest.mark.parametrize(
    "scope",
    [
        {},
        {"document_id": uuid4(), "knowledge_base_id": uuid4()},
        {"document_id": uuid4(), "document_ids": [uuid4()]},
    ],
)
def test_retrieval_request_rejects_missing_or_ambiguous_scope(scope):
    with pytest.raises(ValidationError, match="Provide exactly one scope"):
        RetrievalRequest(query="What is the allocation?", **scope)


def test_production_candidates_exclude_anonymous_semantic_rows():
    expression = str(_candidate_filter("PRODUCTION"))

    assert "chunks.metadata" in expression
    assert "!~*" in expression


def test_extract_query_years_only_returns_years_present_in_scope():
    assert extract_query_years(
        "Compare 2023, FY2024, 2025 and 1H26", {2024, 2025, 2026}
    ) == [2024, 2025, 2026]


def test_extract_query_years_resolves_chinese_relative_period_to_complete_annual_years():
    assert extract_query_years(
        "给我过去5年AIA VONB的变化",
        {2021, 2022, 2023, 2024, 2025, 2026},
        annual_years={2021, 2022, 2023, 2024, 2025},
    ) == [2021, 2022, 2023, 2024, 2025]


def test_group_vonb_time_series_requires_metric_and_multiple_years():
    assert is_group_vonb_time_series(
        "给我过去5年AIA VONB的变化", [2021, 2022, 2023, 2024, 2025]
    )
    assert not is_group_vonb_time_series("AIA VONB", [2025])


def test_group_vonb_chunk_excludes_market_level_rows():
    group_chunk = SimpleNamespace(
        content="Chart: Total Group VONB ($m). Category: VONB. 2024: 4,783.",
        metadata_json={"table_title": "Total Group VONB ($m)"},
    )
    market_chunk = SimpleNamespace(
        content="Table: Singapore ($m). Row label: VONB. 2024: 500.",
        metadata_json={"table_title": "Singapore ($m)"},
    )

    assert is_group_vonb_chunk(group_chunk)
    assert not is_group_vonb_chunk(market_chunk)


def test_lexical_query_terms_keep_financial_subject_and_remove_question_noise():
    assert lexical_query_terms(
        "What was the Unit-linked VONB margin for AIA Group in fiscal year 2025?"
    ) == ["unit", "linked", "vonb", "margin"]
    assert focused_query("What was the Unit-linked VONB margin?", 2025) == (
        "unit linked vonb margin. Fiscal year 2025."
    )


def test_document_context_is_replaced_instead_of_duplicated():
    document = SimpleNamespace(
        company="AIA Group",
        fiscal_year=2024,
        document_type="annual_results",
        title="Annual Results",
    )
    first = enrich_semantic_content("Row label: United States.", document)
    document.fiscal_year = 2025
    second = enrich_semantic_content(first, document)

    assert second.count("[Document context]") == 1
    assert "Fiscal year: 2025." in second
    assert "Fiscal year: 2024." not in second
    assert second.endswith("Row label: United States.")
