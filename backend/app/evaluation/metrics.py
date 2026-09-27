from __future__ import annotations

import re
from collections.abc import Iterable


NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")
UNIT_PATTERN = re.compile(
    r"\b(?:USD|HKD|RMB|CNY|US\$|HK\$|billion|million|thousand|bps?|FY\s?\d{2,4}|H[12]\s?\d{2,4})\b|%|亿美元|百万|十亿",
    re.IGNORECASE,
)
PERIOD_PATTERN = re.compile(
    r"\b(?:FY\s?\d{2,4}|H[12]\s?\d{2,4}|Q[1-4]\s?\d{2,4}|20\d{2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2})\b",
    re.IGNORECASE,
)


def _tokens(pattern: re.Pattern[str], text: str) -> set[str]:
    return {match.group(0).replace(",", "").replace(" ", "").lower() for match in pattern.finditer(text)}


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def evidence_target_matches(target: dict, hit: dict) -> bool:
    checks = []
    if target.get("chunk_id") and hit.get("chunk_id"):
        checks.append(str(hit.get("chunk_id")) == str(target["chunk_id"]))
    if target.get("comparison_key") and hit.get("comparison_key"):
        checks.append(hit.get("comparison_key") == target["comparison_key"])
    if target.get("filename") and (hit.get("file") or hit.get("filename")):
        checks.append((hit.get("file") or hit.get("filename")) == target["filename"])
    if target.get("page") and hit.get("page"):
        checks.append(hit.get("page") == target["page"])
    if target.get("content_hash") and hit.get("content_hash"):
        checks.append(hit.get("content_hash") == target["content_hash"])
    if target.get("document_sha256") and hit.get("document_sha256"):
        checks.append(hit.get("document_sha256") == target["document_sha256"])
    return bool(checks) and all(checks)


def _all_targets_within(targets: list[dict], hits: list[dict], k: int) -> bool:
    return bool(targets) and all(
        any(evidence_target_matches(target, hit) for hit in hits[:k])
        for target in targets
    )


def score_case(
    *,
    expected_answer: str,
    expected_insufficient: bool,
    required_evidence: list[dict],
    retrieval_hits: list[dict],
    actual_answer: str,
    actual_insufficient: bool,
    citations: list[dict],
    language: str,
) -> dict:
    target_ranks = []
    for target in required_evidence:
        rank = next(
            (
                index
                for index, hit in enumerate(retrieval_hits, start=1)
                if evidence_target_matches(target, hit)
            ),
            None,
        )
        target_ranks.append(rank)
    found = sum(rank is not None for rank in target_ranks)
    evidence_recall = found / len(required_evidence) if required_evidence else None

    expected_numbers = _tokens(NUMBER_PATTERN, expected_answer)
    actual_numbers = _tokens(NUMBER_PATTERN, actual_answer)
    expected_units = _tokens(UNIT_PATTERN, expected_answer)
    actual_units = _tokens(UNIT_PATTERN, actual_answer)
    expected_periods = _tokens(PERIOD_PATTERN, expected_answer)
    actual_periods = _tokens(PERIOD_PATTERN, actual_answer)

    citation_matches = [
        any(evidence_target_matches(target, {"file": citation.get("filename"), **citation}) for target in required_evidence)
        for citation in citations
    ]
    citation_precision = (
        sum(citation_matches) / len(citation_matches) if citations else (1.0 if not required_evidence else 0.0)
    )
    cited_targets = sum(
        any(
            evidence_target_matches(target, {"file": citation.get("filename"), **citation})
            for citation in citations
        )
        for target in required_evidence
    )
    citation_recall = cited_targets / len(required_evidence) if required_evidence else 1.0
    language_match = _contains_cjk(actual_answer) if language.startswith("zh") else not _contains_cjk(actual_answer)
    has_targets = bool(required_evidence)

    return {
        "target_ranks": target_ranks,
        "hit_at_1": _all_targets_within(required_evidence, retrieval_hits, 1) if has_targets else None,
        "hit_at_3": _all_targets_within(required_evidence, retrieval_hits, 3) if has_targets else None,
        "hit_at_5": _all_targets_within(required_evidence, retrieval_hits, 5) if has_targets else None,
        "evidence_recall_at_6": evidence_recall,
        "mrr": (1.0 / min(rank for rank in target_ranks if rank is not None) if found else 0.0) if has_targets else None,
        "number_accuracy": expected_numbers.issubset(actual_numbers),
        "unit_accuracy": expected_units.issubset(actual_units),
        "period_accuracy": expected_periods.issubset(actual_periods),
        "insufficient_accuracy": expected_insufficient == actual_insufficient,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_page_accuracy": citation_recall == 1.0,
        "language_match": language_match,
    }


def aggregate_metrics(results: Iterable[dict]) -> dict:
    rows = list(results)
    if not rows:
        return {}

    def average(path: tuple[str, ...], *, failed_zero: bool = True) -> float:
        values = []
        for row in rows:
            if row.get("status") != "COMPLETE" and failed_zero:
                values.append(0.0)
                continue
            value = row
            missing = False
            for part in path:
                if not isinstance(value, dict) or part not in value:
                    missing = True
                    break
                value = value[part]
            if missing:
                values.append(0.0)
                continue
            if value is not None:
                values.append(float(value))
        return round(sum(values) / len(values), 4) if values else 0.0

    completed = [row for row in rows if row.get("status") == "COMPLETE"]
    return {
        "case_count": len(rows),
        "completed_count": len(completed),
        "failed_count": len(rows) - len(completed),
        "hit_at_1": average(("deterministic_metrics", "hit_at_1")),
        "hit_at_3": average(("deterministic_metrics", "hit_at_3")),
        "hit_at_5": average(("deterministic_metrics", "hit_at_5")),
        "mrr": average(("deterministic_metrics", "mrr")),
        "number_accuracy": average(("deterministic_metrics", "number_accuracy")),
        "unit_accuracy": average(("deterministic_metrics", "unit_accuracy")),
        "period_accuracy": average(("deterministic_metrics", "period_accuracy")),
        "citation_precision": average(("deterministic_metrics", "citation_precision")),
        "citation_recall": average(("deterministic_metrics", "citation_recall")),
        "insufficient_accuracy": average(("deterministic_metrics", "insufficient_accuracy")),
        "judge_correctness": average(("judge", "correctness")),
        "judge_completeness": average(("judge", "completeness")),
        "judge_faithfulness": average(("judge", "faithfulness")),
        "average_latency_seconds": average(("latency_seconds",), failed_zero=False),
    }
