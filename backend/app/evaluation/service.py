from __future__ import annotations

import hashlib
import re
import time
import unicodedata
import uuid
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.evaluation.metrics import aggregate_metrics, evidence_target_matches, score_case
from app.evaluation.provider import AlibabaEvaluationProvider
from app.models import (
    Chunk,
    Document,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationRun,
)
from app.qa.service import QAService


NUMERIC_TOKEN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).lower().split())


def numeric_tokens(value: str) -> set[str]:
    return {
        token.replace(",", "").lower()
        for token in NUMERIC_TOKEN.findall(value)
    }


def _case_response(case: EvaluationCase) -> dict:
    return {
        "id": case.id,
        "dataset_id": case.dataset_id,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "language": case.language,
        "expected_insufficient": case.expected_insufficient,
        "required_evidence": case.required_evidence_json,
        "scope": case.scope_json,
        "tags": case.tags_json,
        "difficulty": case.difficulty,
        "source": case.source,
        "status": case.status,
        "created_at": case.created_at,
    }


class EvaluationService:
    def __init__(self) -> None:
        self.provider = AlibabaEvaluationProvider()

    async def generate_cases(
        self,
        dataset: EvaluationDataset,
        *,
        count: int,
        language: str,
        difficulty: str,
        include_insufficient: bool,
        session: AsyncSession,
    ) -> list[EvaluationCase]:
        source_count = count - 1 if include_insufficient and count > 1 else count
        rows = list(
            (
                await session.execute(
                    select(Chunk, Document)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(
                        Document.knowledge_base_id == dataset.knowledge_base_id,
                        Document.status == "READY",
                        or_(
                            Chunk.representation.in_(["SEMANTIC_ROW", "SUMMARY"]),
                            Chunk.chunk_type.in_(["FACT", "SECTION"]),
                        ),
                    )
                    .order_by(Chunk.chunk_type, Document.created_at, Chunk.page_number, Chunk.id)
                )
            ).all()
        )
        if not rows:
            raise ValueError("The knowledge base has no indexed evidence suitable for generation.")

        groups: dict[str, list[tuple[Chunk, Document]]] = {}
        for chunk, document in rows:
            groups.setdefault(chunk.chunk_type, []).append((chunk, document))
        selected: list[tuple[Chunk, Document]] = []
        group_names = sorted(groups)
        offsets = {name: 0 for name in group_names}
        while len(selected) < min(source_count, len(rows)):
            added = False
            for name in group_names:
                index = offsets[name]
                if index < len(groups[name]):
                    selected.append(groups[name][index])
                    offsets[name] += 1
                    added = True
                    if len(selected) >= source_count:
                        break
            if not added:
                break

        sources = [
            {
                "source_index": index,
                "filename": document.filename,
                "page": chunk.page_number,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
            }
            for index, (chunk, document) in enumerate(selected)
        ]
        generated = []
        for start in range(0, len(sources), 10):
            generated.extend(
                await self.provider.generate_questions(
                    sources[start : start + 10],
                    language=language,
                    difficulty=difficulty,
                )
            )

        existing = {
            normalize_question(question)
            for question in (
                await session.execute(
                    select(EvaluationCase.question).where(EvaluationCase.dataset_id == dataset.id)
                )
            ).scalars()
        }
        cases: list[EvaluationCase] = []
        used_sources: set[int] = set()
        for item in generated:
            if item.source_index >= len(selected) or item.source_index in used_sources:
                continue
            chunk, document = selected[item.source_index]
            expected_numbers = numeric_tokens(item.expected_answer)
            if expected_numbers and not expected_numbers.issubset(numeric_tokens(chunk.content)):
                continue
            normalized = normalize_question(item.question)
            if normalized in existing:
                continue
            target = {
                "document_sha256": document.source_sha256,
                "filename": document.filename,
                "page": chunk.page_number,
                "comparison_key": chunk.comparison_key,
                "content_hash": content_hash(chunk.content),
                "chunk_id": str(chunk.id),
            }
            case = EvaluationCase(
                id=uuid.uuid4(),
                dataset_id=dataset.id,
                question=item.question.strip(),
                expected_answer=item.expected_answer.strip(),
                language=language,
                expected_insufficient=False,
                required_evidence_json=[target],
                scope_json={},
                tags_json=list(dict.fromkeys(item.tags))[:20],
                difficulty=item.difficulty if item.difficulty in {"easy", "medium", "hard"} else "medium",
                source="AI",
                status="DRAFT",
            )
            cases.append(case)
            existing.add(normalized)
            used_sources.add(item.source_index)

        if include_insufficient:
            negative = EvaluationCase(
                id=uuid.uuid4(),
                dataset_id=dataset.id,
                question=(
                    "该公司在2099财年来自火星业务的经审计收入是多少？"
                    if language == "zh"
                    else "What was the company's audited revenue from operations on Mars in FY2099?"
                ),
                expected_answer=(
                    "证据不足，知识库没有该信息。"
                    if language == "zh"
                    else "The evidence is insufficient; the knowledge base does not contain this information."
                ),
                language=language,
                expected_insufficient=True,
                required_evidence_json=[],
                scope_json={},
                tags_json=["insufficient-evidence", "negative"],
                difficulty="easy",
                source="AI",
                status="DRAFT",
            )
            if normalize_question(negative.question) not in existing:
                cases.append(negative)

        session.add_all(cases)
        await session.commit()
        for case in cases:
            await session.refresh(case)
        return cases

    async def _expected_evidence(
        self,
        targets: list[dict],
        knowledge_base_id: UUID,
        session: AsyncSession,
    ) -> list[dict]:
        rows = list(
            (
                await session.execute(
                    select(Chunk, Document)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(Document.knowledge_base_id == knowledge_base_id)
                )
            ).all()
        )
        evidence = []
        for target in targets:
            for chunk, document in rows:
                candidate = {
                    "chunk_id": str(chunk.id),
                    "comparison_key": chunk.comparison_key,
                    "file": document.filename,
                    "page": chunk.page_number,
                    "content_hash": content_hash(chunk.content),
                    "document_sha256": document.source_sha256,
                }
                if evidence_target_matches(target, candidate):
                    evidence.append({**candidate, "content": chunk.content})
                    break
        return evidence

    async def process_run(self, run_id: UUID, session: AsyncSession) -> None:
        run = await session.get(EvaluationRun, run_id)
        if run is None:
            return
        dataset = await session.get(EvaluationDataset, run.dataset_id)
        if dataset is None:
            raise ValueError("Evaluation dataset no longer exists.")
        cases = list(
            (
                await session.execute(
                    select(EvaluationCase)
                    .where(
                        EvaluationCase.dataset_id == dataset.id,
                        EvaluationCase.status == "APPROVED",
                    )
                    .order_by(EvaluationCase.created_at, EvaluationCase.id)
                )
            ).scalars()
        )
        if not cases:
            raise ValueError("Published dataset has no approved cases.")
        documents = list(
            (
                await session.execute(
                    select(Document).where(
                        Document.knowledge_base_id == run.knowledge_base_id,
                        Document.status == "READY",
                    )
                )
            ).scalars()
        )
        if not documents:
            raise ValueError("Knowledge base has no ready documents.")

        result_rows: list[dict] = []
        for index, case in enumerate(cases, start=1):
            started = time.monotonic()
            result = EvaluationCaseResult(
                id=uuid.uuid4(), run_id=run.id, case_id=case.id, status="RUNNING"
            )
            session.add(result)
            await session.flush()
            try:
                scoped_documents = [
                    document
                    for document in documents
                    if (
                        not case.scope_json.get("company")
                        or (document.company or "").lower() == str(case.scope_json["company"]).lower()
                    )
                    and (
                        not case.scope_json.get("fiscal_year")
                        or document.fiscal_year == int(case.scope_json["fiscal_year"])
                    )
                    and (
                        not case.scope_json.get("document_type")
                        or document.document_type == case.scope_json["document_type"]
                    )
                ]
                if not scoped_documents:
                    raise ValueError("No ready documents matched the evaluation case scope.")
                snapshot, retrieval, evidence = await QAService().retrieve(
                    documents=scoped_documents,
                    question=case.question,
                    mode=run.retrieval_mode,
                    session=session,
                    knowledge_base_id=run.knowledge_base_id,
                )
                answer = await QAService().answer(snapshot)
                retrieval_payload = retrieval.model_dump(mode="json")
                document_by_id = {str(document.id): document for document in scoped_documents}
                for stage in ("vector_results", "lexical_results", "rrf_results", "reranked_results"):
                    for hit in retrieval_payload[stage]:
                        document = document_by_id.get(str(hit["document_id"]))
                        hit["content_hash"] = content_hash(hit["content"])
                        hit["document_sha256"] = document.source_sha256 if document else None
                citations = [item.model_dump(mode="json") for item in answer["citations"]]
                deterministic = score_case(
                    expected_answer=case.expected_answer,
                    expected_insufficient=case.expected_insufficient,
                    required_evidence=case.required_evidence_json,
                    retrieval_hits=retrieval_payload["reranked_results"],
                    actual_answer=answer["answer"],
                    actual_insufficient=answer["insufficient_evidence"],
                    citations=citations,
                    language=case.language,
                )
                expected_evidence = await self._expected_evidence(
                    case.required_evidence_json, run.knowledge_base_id, session
                )
                try:
                    judge = (
                        await self.provider.judge(
                            question=case.question,
                            expected_answer=case.expected_answer,
                            expected_evidence=expected_evidence,
                            actual_answer=answer["answer"],
                            citations=citations,
                        )
                    ).model_dump()
                except Exception as exc:
                    judge = {"error": f"{type(exc).__name__}: {exc}"}

                failure_stage = self._failure_stage(case, retrieval_payload, deterministic, judge)
                result.status = "COMPLETE"
                result.retrieval_json = retrieval_payload
                result.final_evidence_json = [item.model_dump(mode="json") for item in evidence]
                result.answer = answer["answer"]
                result.insufficient_evidence = answer["insufficient_evidence"]
                result.citations_json = citations
                result.deterministic_metrics_json = deterministic
                result.judge_json = judge
                result.failure_stage = failure_stage
                result.latency_seconds = round(time.monotonic() - started, 3)
            except Exception as exc:
                result.status = "FAILED"
                result.error = f"{type(exc).__name__}: {exc}"
                result.failure_stage = (
                    "DOCUMENT_SCOPE_FAILURE"
                    if "scope" in str(exc).lower()
                    else "PROVIDER_FAILURE"
                )
                result.latency_seconds = round(time.monotonic() - started, 3)

            run.progress = round(index / len(cases) * 100)
            await session.commit()
            result_rows.append(
                {
                    "status": result.status,
                    "deterministic_metrics": result.deterministic_metrics_json,
                    "judge": result.judge_json,
                    "latency_seconds": result.latency_seconds,
                }
            )

        run = await session.get(EvaluationRun, run_id)
        run.metrics_json = aggregate_metrics(result_rows)
        run.status = "COMPLETE"
        run.progress = 100
        run.completed_at = datetime.now(UTC)
        await session.commit()

    def _failure_stage(
        self, case: EvaluationCase, retrieval: dict, metrics: dict, judge: dict
    ) -> str | None:
        targets = case.required_evidence_json
        if targets and not metrics["hit_at_5"]:
            def complete(stage: str) -> bool:
                return all(
                    any(evidence_target_matches(target, hit) for hit in retrieval[stage])
                    for target in targets
                )

            if complete("rrf_results"):
                return "RERANK_FAILURE"
            if complete("vector_results") or complete("lexical_results"):
                return "RRF_FAILURE"
            return "FIRST_STAGE_FAILURE"
        if not metrics["insufficient_accuracy"]:
            return "FALSE_SUFFICIENT" if case.expected_insufficient else "FALSE_INSUFFICIENT"
        if not metrics["number_accuracy"]:
            return "NUMERIC_ERROR"
        if not metrics["unit_accuracy"]:
            return "UNIT_ERROR"
        if not metrics["period_accuracy"]:
            return "PERIOD_ERROR"
        if metrics["citation_recall"] < 1:
            return "CITATION_ERROR"
        if judge.get("error"):
            return "JUDGE_PROVIDER_FAILURE"
        if judge.get("faithfulness", 1) < 0.8:
            return "ANSWER_HALLUCINATION"
        if judge.get("correctness", 1) < 0.8 or judge.get("completeness", 1) < 0.8:
            return "ANSWER_INCORRECT"
        return None


__all__ = ["EvaluationService", "_case_response", "content_hash", "numeric_tokens"]
