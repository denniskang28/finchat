from app.evaluation.metrics import aggregate_metrics, evidence_target_matches, score_case
import uuid
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.evaluation.provider import GeneratedCalculation, create_generation_provider
from app.evaluation.service import (
    EvaluationService,
    every_source_supports_expected_value,
    mentions_fiscal_year,
    numeric_tokens,
    row_period_matches_document,
    validated_calculation_tokens,
)
from app.models import Chunk, Document


def test_evidence_target_matches_stable_locator_and_citation_subset():
    target = {
        "filename": "report.pdf",
        "page": 91,
        "comparison_key": "p91_t1:united_states",
        "chunk_id": "chunk-1",
    }

    assert evidence_target_matches(
        target,
        {
            "file": "report.pdf",
            "page": 91,
            "comparison_key": "p91_t1:united_states",
            "chunk_id": "chunk-1",
        },
    )
    assert evidence_target_matches(
        target,
        {"filename": "report.pdf", "page": 91, "chunk_id": "chunk-1"},
    )
    assert not evidence_target_matches(
        target,
        {"filename": "report.pdf", "page": 90, "chunk_id": "chunk-1"},
    )


def test_section_target_accepts_fact_child_in_same_document_and_page():
    target = {
        "filename": "report.pdf",
        "page": 2,
        "comparison_key": "p2_section1_disclaimer",
        "chunk_id": "section-chunk",
    }

    assert evidence_target_matches(
        target,
        {
            "filename": "report.pdf",
            "page": 2,
            "chunk_id": "fact-chunk",
            "parent_key": "p2_section1_disclaimer",
        },
    )
    assert not evidence_target_matches(
        target,
        {
            "filename": "other.pdf",
            "page": 2,
            "parent_key": "p2_section1_disclaimer",
        },
    )


def test_score_case_checks_financial_values_periods_and_citations():
    target = {
        "filename": "report.pdf",
        "page": 91,
        "comparison_key": "p91_t1:united_states",
        "chunk_id": "chunk-1",
    }
    metrics = score_case(
        expected_answer="FY2025: USD 6.2 billion and 22%.",
        expected_insufficient=False,
        required_evidence=[target],
        retrieval_hits=[
            {
                "file": "report.pdf",
                "page": 91,
                "comparison_key": "p91_t1:united_states",
                "chunk_id": "chunk-1",
            }
        ],
        actual_answer="In FY2025 it was USD 6.2 billion, or 22% [E1].",
        actual_insufficient=False,
        citations=[{"filename": "report.pdf", "page": 91, "chunk_id": "chunk-1"}],
        language="en",
    )

    assert metrics["hit_at_1"] is True
    assert metrics["number_accuracy"] is True
    assert metrics["unit_accuracy"] is True
    assert metrics["period_accuracy"] is True
    assert metrics["citation_recall"] == 1.0


def test_multi_source_hit_and_complete_hit_are_reported_separately():
    targets = [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}]
    metrics = score_case(
        expected_answer="2024: 10%. 2025: 12%.",
        expected_insufficient=False,
        required_evidence=targets,
        retrieval_hits=[{"chunk_id": "chunk-1"}, {"chunk_id": "other"}],
        actual_answer="2024: 10%. 2025: 12%.",
        actual_insufficient=False,
        citations=[{"chunk_id": "chunk-1"}],
        language="en",
    )

    assert metrics["hit_at_1"] is True
    assert metrics["complete_hit_at_1"] is False
    assert metrics["complete_hit_at_5"] is False


def test_numeric_tokens_preserve_percent_and_normalize_commas():
    assert numeric_tokens("USD 1,234.5 million and 22%") == {"1234.5", "22%"}


def test_aggregate_metrics_keeps_retrieval_and_answer_layers_separate():
    metrics = aggregate_metrics(
        [
            {
                "status": "COMPLETE",
                "deterministic_metrics": {
                    "hit_at_1": True,
                    "hit_at_3": True,
                    "hit_at_5": True,
                    "mrr": 1.0,
                    "number_accuracy": True,
                    "unit_accuracy": True,
                    "period_accuracy": True,
                    "citation_precision": 1.0,
                    "citation_recall": 1.0,
                    "insufficient_accuracy": True,
                },
                "judge": {"correctness": 0.9, "completeness": 0.8, "faithfulness": 1.0},
                "latency_seconds": 2.0,
            },
            {
                "status": "FAILED",
                "deterministic_metrics": {},
                "judge": {},
                "latency_seconds": 4.0,
            },
        ]
    )

    assert metrics["case_count"] == 2
    assert metrics["completed_count"] == 1
    assert metrics["hit_at_1"] == 0.5
    assert metrics["judge_correctness"] == 0.45
    assert metrics["average_latency_seconds"] == 3.0


def test_insufficient_case_does_not_reduce_retrieval_metrics():
    metrics = score_case(
        expected_answer="The evidence is insufficient.",
        expected_insufficient=True,
        required_evidence=[],
        retrieval_hits=[],
        actual_answer="The evidence is insufficient.",
        actual_insufficient=True,
        citations=[],
        language="en",
    )

    assert metrics["hit_at_1"] is None
    assert metrics["mrr"] is None
    assert metrics["insufficient_accuracy"] is True


def _document(year: int) -> Document:
    return Document(
        id=uuid.uuid4(),
        filename=f"aia-{year}.pdf",
        file_path=f"/tmp/aia-{year}.pdf",
        status="READY",
        source_sha256=str(year) * 16,
        company="AIA Group",
        fiscal_year=year,
        document_type="annual results",
    )


def _corporate_bond_row(document: Document, value: str, share: str) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=document.id,
        page_number=91 if document.fiscal_year == 2025 else 65,
        chunk_type="TABLE_ROW",
        representation="SEMANTIC_ROW",
        comparison_key=f"p91_t1:united_states",
        content=(
            f"Company: AIA Group. Fiscal year: {document.fiscal_year}. "
            "Table: Corporate Bonds by Geography. Row label: United States. "
            f"$b: USD {value} billion. % of total: {share}%."
        ),
        metadata_json={
            "table_title": "Corporate Bonds by Geography",
            "row_label": "United States",
        },
    )


def test_cross_year_bundle_uses_matching_rows_from_distinct_years():
    document_2024 = _document(2024)
    document_2025 = _document(2025)
    rows = [
        (_corporate_bond_row(document_2024, "5.7", "20"), document_2024),
        (_corporate_bond_row(document_2025, "6.2", "22"), document_2025),
    ]

    bundles = EvaluationService()._build_evidence_bundles(
        rows,
        {"single_document": 0, "cross_year": 1, "cross_document": 0},
        [2024, 2025],
    )

    assert len(bundles) == 1
    assert bundles[0]["scenario_type"] == "CROSS_YEAR"
    assert {source["fiscal_year"] for source in bundles[0]["payload"]["sources"]} == {
        2024,
        2025,
    }
    assert len({source["document_id"] for source in bundles[0]["payload"]["sources"]}) == 2


def test_cross_year_bundle_skips_ambiguous_duplicate_identity_within_a_year():
    document_2024 = _document(2024)
    document_2025 = _document(2025)
    rows = [
        (_corporate_bond_row(document_2024, "5.7", "20"), document_2024),
        (_corporate_bond_row(document_2024, "1.1", "4"), document_2024),
        (_corporate_bond_row(document_2025, "6.2", "22"), document_2025),
    ]

    bundles = EvaluationService()._build_evidence_bundles(
        rows,
        {"single_document": 0, "cross_year": 1, "cross_document": 0},
        [2024, 2025],
    )

    assert bundles == []


def test_cross_year_bundle_skips_historical_row_mislabeled_as_report_year():
    document_2024 = _document(2024)
    document_2025 = _document(2025)
    row_2024 = _corporate_bond_row(document_2024, "32", "32")
    row_2025 = _corporate_bond_row(document_2025, "32", "32")
    row_2024.metadata_json["row_label"] = "2024"
    row_2025.metadata_json["row_label"] = "2024"

    assert row_period_matches_document(row_2024, document_2024)
    assert not row_period_matches_document(row_2025, document_2025)
    assert EvaluationService()._build_evidence_bundles(
        [(row_2024, document_2024), (row_2025, document_2025)],
        {"single_document": 0, "cross_year": 1, "cross_document": 0},
        [2024, 2025],
    ) == []


def test_fiscal_year_mentions_accept_full_year_and_financial_period_aliases():
    assert mentions_fiscal_year("FY2024 annual results", 2024)
    assert mentions_fiscal_year("1H26 interim results", 2026)
    assert not mentions_fiscal_year("FY2024 annual results", 2025)


def test_each_source_must_contribute_its_answer_value():
    document_2024 = _document(2024)
    document_2025 = _document(2025)
    rows = [
        (_corporate_bond_row(document_2024, "39", "39"), document_2024),
        (_corporate_bond_row(document_2025, "36", "36"), document_2025),
    ]

    assert every_source_supports_expected_value(
        {"39%", "36%", "2024", "2025"}, rows, [2024, 2025]
    )
    assert not every_source_supports_expected_value(
        {"39%", "2024", "2025"}, rows, [2024, 2025]
    )


def test_correct_insufficient_answer_is_not_labeled_hallucination():
    case = SimpleNamespace(required_evidence_json=[], expected_insufficient=True)
    stage = EvaluationService()._failure_stage(
        case,
        {},
        {
            "insufficient_accuracy": True,
            "number_accuracy": True,
            "unit_accuracy": True,
            "period_accuracy": True,
            "citation_recall": 1.0,
        },
        {"correctness": 1.0, "completeness": 1.0, "faithfulness": 0.75},
    )

    assert stage is None


def test_calculation_validation_requires_source_operands_and_exact_result():
    valid = GeneratedCalculation(expression="6.2 - 5.7", result="0.5", unit="USD billion")
    assert validated_calculation_tokens([valid], {"6.2", "5.7", "22%", "20%"}) == {"0.5"}
    assert validated_calculation_tokens(
        [GeneratedCalculation(expression="5,516 - 1,430", result="4,086")],
        {"5516", "1430"},
    ) == {"4086"}

    with pytest.raises(ValueError, match="absent"):
        validated_calculation_tokens(
            [GeneratedCalculation(expression="6.2 - 4.0", result="2.2")],
            {"6.2", "5.7"},
        )
    with pytest.raises(ValueError, match="does not match"):
        validated_calculation_tokens(
            [GeneratedCalculation(expression="6.2 - 5.7", result="0.6")],
            {"6.2", "5.7"},
        )


def test_generation_provider_is_selected_from_server_allowlist():
    settings = Settings(
        dashscope_api_key="qwen-key",
        deepseek_api_key="deepseek-key",
        alibaba_generation_models="qwen3.8-flash,qwen-plus",
        deepseek_generation_models="deepseek-chat",
    )

    qwen = create_generation_provider("alibaba", "qwen-plus", settings)
    deepseek = create_generation_provider("deepseek", "deepseek-chat", settings)

    assert (qwen.name, qwen.model) == ("alibaba", "qwen-plus")
    assert (deepseek.name, deepseek.model) == ("deepseek", "deepseek-chat")
    with pytest.raises(ValueError, match="not allowed"):
        create_generation_provider("deepseek", "untrusted-model", settings)
