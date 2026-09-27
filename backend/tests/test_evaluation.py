from app.evaluation.metrics import aggregate_metrics, evidence_target_matches, score_case
from app.evaluation.service import numeric_tokens


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
