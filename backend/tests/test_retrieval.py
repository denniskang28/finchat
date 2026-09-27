from uuid import uuid4

from app.retrieval.service import grade_targets, reciprocal_rank_fusion


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
