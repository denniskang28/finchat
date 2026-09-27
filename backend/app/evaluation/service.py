from __future__ import annotations

import ast
import hashlib
import operator
import re
import time
import unicodedata
import uuid
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.evaluation.metrics import aggregate_metrics, evidence_target_matches, score_case
from app.evaluation.provider import (
    GENERATION_PROMPT_VERSION,
    AlibabaEvaluationProvider,
    GeneratedCalculation,
    create_generation_provider,
)
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


def semantic_identity(chunk: Chunk) -> str | None:
    metadata = chunk.metadata_json or {}
    title = metadata.get("table_title") or metadata.get("section_title") or ""
    row = metadata.get("row_label") or metadata.get("fact_label") or ""
    if not row and chunk.comparison_key and ":" in chunk.comparison_key:
        row = chunk.comparison_key.rsplit(":", 1)[-1]
    normalized = re.sub(r"[^a-z0-9]+", " ", f"{chunk.chunk_type} {title} {row}".lower()).strip()
    return normalized or None


def period_year(value: str | None) -> int | None:
    if not value:
        return None
    full = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", value)
    if full:
        return int(full.group(1))
    short = re.search(r"(?i)(?:FY|H[12]|[12]H|Q[1-4])\s*'?([0-9]{2})(?!\d)", value)
    return 2000 + int(short.group(1)) if short else None


def mentions_fiscal_year(value: str, year: int) -> bool:
    if str(year) in value:
        return True
    suffix = str(year)[-2:]
    return bool(
        re.search(
            rf"(?i)(?:FY|H[12]|[12]H|Q[1-4])\s*'?{re.escape(suffix)}(?!\d)",
            value,
        )
    )


def row_period_matches_document(chunk: Chunk, document: Document) -> bool:
    row_year = period_year((chunk.metadata_json or {}).get("row_label"))
    return row_year is None or row_year == document.fiscal_year


def _evaluate_expression(expression: str) -> float:
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.operand))
        raise ValueError("Calculation contains unsupported syntax.")

    return evaluate(ast.parse(expression, mode="eval"))


def validated_calculation_tokens(
    calculations: list[GeneratedCalculation], source_numbers: set[str]
) -> set[str]:
    results = set()
    for calculation in calculations:
        operands = numeric_tokens(calculation.expression)
        if not operands or not operands.issubset(source_numbers):
            raise ValueError("Calculation uses a number that is absent from its evidence.")
        result_tokens = numeric_tokens(calculation.result)
        if len(result_tokens) != 1:
            raise ValueError("Calculation result must contain exactly one number.")
        result_token = next(iter(result_tokens))
        expected = float(result_token.rstrip("%"))
        actual = _evaluate_expression(calculation.expression.replace("%", "").replace(",", ""))
        if abs(actual - expected) > max(1e-6, abs(expected) * 1e-4):
            raise ValueError("Calculation result does not match its expression.")
        results.add(result_token)
    return results


def every_source_supports_expected_value(
    expected_numbers: set[str],
    rows: list[tuple[Chunk, Document]],
    years: list[int],
) -> bool:
    period_numbers = {str(year) for year in years} | {str(year)[-2:] for year in years}
    expected_financial_values = expected_numbers - period_numbers
    return not expected_financial_values or all(
        bool((numeric_tokens(chunk.content) - period_numbers) & expected_financial_values)
        for chunk, document in rows
    )


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
        "generation_metadata": case.generation_metadata_json,
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
        scenario_mix: dict | None,
        generation_provider: str,
        generation_model: str | None,
        document_ids: list[UUID],
        years: list[int],
        company: str | None,
        allow_calculations: bool,
        session: AsyncSession,
    ) -> list[EvaluationCase]:
        requested = scenario_mix or {
            "single_document": count - 1 if include_insufficient and count > 1 else count,
            "cross_year": 0,
            "cross_document": 0,
        }
        provider = create_generation_provider(generation_provider, generation_model)
        statement = (
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
        )
        if document_ids:
            statement = statement.where(Document.id.in_(document_ids))
        if years:
            statement = statement.where(Document.fiscal_year.in_(years))
        if company:
            statement = statement.where(Document.company.ilike(company.strip()))
        rows = list(
            (
                await session.execute(
                    statement.order_by(
                        Chunk.chunk_type, Document.created_at, Chunk.page_number, Chunk.id
                    )
                )
            ).all()
        )
        if not rows:
            raise ValueError("The knowledge base has no indexed evidence suitable for generation.")

        bundles = self._build_evidence_bundles(rows, requested, years)
        if not bundles:
            raise ValueError("No evidence bundles matched the requested generation scenarios.")
        generated = []
        for start in range(0, len(bundles), 6):
            generated.extend(
                await provider.generate_questions(
                    [bundle["payload"] for bundle in bundles[start : start + 6]],
                    language=language,
                    difficulty=difficulty,
                    allow_calculations=allow_calculations,
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
        used_bundles: set[int] = set()
        for item in generated:
            if item.bundle_index >= len(bundles) or item.bundle_index in used_bundles:
                continue
            bundle = bundles[item.bundle_index]
            scenario_type = bundle["scenario_type"]
            if item.scenario_type.upper() != scenario_type:
                continue
            expected_source_ids = {source["source_id"] for source in bundle["payload"]["sources"]}
            if set(item.source_ids_used) != expected_source_ids:
                continue
            table_titles = {
                str(source.get("table_title") or "").strip()
                for source in bundle["payload"]["sources"]
                if str(source.get("table_title") or "").strip()
            }
            if not all(title.lower() in item.question.lower() for title in table_titles):
                continue
            documents = {str(document.id): document for chunk, document in bundle["rows"]}
            if scenario_type != "SINGLE_DOCUMENT" and len(documents) < 2:
                continue
            bundle_years = sorted(
                {document.fiscal_year for chunk, document in bundle["rows"] if document.fiscal_year}
            )
            if scenario_type == "SINGLE_DOCUMENT" and bundle_years:
                if not all(mentions_fiscal_year(item.question, year) for year in bundle_years):
                    continue
            if scenario_type == "CROSS_YEAR":
                if len(bundle_years) < 2:
                    continue
                question_and_answer = f"{item.question} {item.expected_answer}"
                if not all(
                    mentions_fiscal_year(question_and_answer, year)
                    for year in bundle_years
                ):
                    continue
            if scenario_type == "CROSS_DOCUMENT" and bundle_years:
                question_and_answer = f"{item.question} {item.expected_answer}"
                if not all(
                    mentions_fiscal_year(question_and_answer, year)
                    for year in bundle_years
                ):
                    continue
            source_numbers = set().union(
                *(numeric_tokens(chunk.content) for chunk, document in bundle["rows"])
            )
            try:
                calculated_numbers = validated_calculation_tokens(item.calculations, source_numbers)
            except (ValueError, SyntaxError, ZeroDivisionError):
                continue
            expected_numbers = numeric_tokens(item.expected_answer)
            if expected_numbers and not expected_numbers.issubset(
                source_numbers | calculated_numbers
            ):
                continue
            if not every_source_supports_expected_value(
                expected_numbers, bundle["rows"], bundle_years
            ):
                continue
            normalized = normalize_question(item.question)
            if normalized in existing:
                continue
            targets = [self._evidence_target(chunk, document) for chunk, document in bundle["rows"]]
            case = EvaluationCase(
                id=uuid.uuid4(),
                dataset_id=dataset.id,
                question=item.question.strip(),
                expected_answer=item.expected_answer.strip(),
                language=language,
                expected_insufficient=False,
                required_evidence_json=targets,
                scope_json={},
                tags_json=list(
                    dict.fromkeys([scenario_type.lower().replace("_", "-"), *item.tags])
                )[:20],
                generation_metadata_json={
                    "scenario_type": scenario_type,
                    "provider": provider.name,
                    "model": provider.model,
                    "prompt_version": GENERATION_PROMPT_VERSION,
                    "source_document_ids": list(documents),
                    "source_years": bundle_years,
                    "calculations": [value.model_dump() for value in item.calculations],
                    "validation": {
                        "unique_documents": len(documents),
                        "numeric_provenance": True,
                        "all_sources_used": True,
                    },
                },
                difficulty=item.difficulty if item.difficulty in {"easy", "medium", "hard"} else "medium",
                source="AI",
                status="DRAFT",
            )
            cases.append(case)
            existing.add(normalized)
            used_bundles.add(item.bundle_index)

        if not cases:
            raise ValueError(
                "The model returned no cases that passed grounding validation. "
                "Try another model or narrower document, company, and year filters."
            )

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
                generation_metadata_json={
                    "scenario_type": "INSUFFICIENT_EVIDENCE",
                    "provider": provider.name,
                    "model": provider.model,
                    "prompt_version": GENERATION_PROMPT_VERSION,
                },
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

    def _evidence_target(self, chunk: Chunk, document: Document) -> dict:
        return {
            "document_sha256": document.source_sha256,
            "filename": document.filename,
            "page": chunk.page_number,
            "comparison_key": chunk.comparison_key,
            "content_hash": content_hash(chunk.content),
            "chunk_id": str(chunk.id),
        }

    def _source_payload(self, source_id: str, chunk: Chunk, document: Document) -> dict:
        return {
            "source_id": source_id,
            "document_id": str(document.id),
            "filename": document.filename,
            "company": document.company,
            "fiscal_year": document.fiscal_year,
            "document_type": document.document_type,
            "page": chunk.page_number,
            "chunk_type": chunk.chunk_type,
            "table_title": chunk.metadata_json.get("table_title"),
            "row_label": chunk.metadata_json.get("row_label"),
            "content": chunk.content,
        }

    def _build_evidence_bundles(
        self,
        rows: list[tuple[Chunk, Document]],
        requested: dict,
        requested_years: list[int],
    ) -> list[dict]:
        bundles: list[dict] = []
        used_signatures: set[tuple[str, ...]] = set()

        type_groups: dict[str, list[tuple[Chunk, Document]]] = {}
        for row in rows:
            type_groups.setdefault(row[0].chunk_type, []).append(row)
        single_rows: list[tuple[Chunk, Document]] = []
        offsets = {name: 0 for name in sorted(type_groups)}
        while len(single_rows) < requested.get("single_document", 0):
            added = False
            for name in sorted(type_groups):
                index = offsets[name]
                if index < len(type_groups[name]):
                    single_rows.append(type_groups[name][index])
                    offsets[name] += 1
                    added = True
                    if len(single_rows) >= requested.get("single_document", 0):
                        break
            if not added:
                break
        for row in single_rows:
            bundles.append(self._bundle("SINGLE_DOCUMENT", [row], len(bundles)))

        comparable = [
            row
            for row in rows
            if row[1].fiscal_year
            and semantic_identity(row[0])
            and row_period_matches_document(row[0], row[1])
            and row[0].chunk_type not in {"TABLE_SUMMARY", "SECTION"}
        ]
        year_groups: dict[tuple[str, str, str], list[tuple[Chunk, Document]]] = {}
        for chunk, document in comparable:
            key = (
                (document.company or "").strip().lower(),
                (document.document_type or "").strip().lower(),
                semantic_identity(chunk) or "",
            )
            year_groups.setdefault(key, []).append((chunk, document))
        for values in year_groups.values():
            if requested.get("cross_year", 0) <= 0:
                break
            candidates_by_year: dict[int, list[tuple[Chunk, Document]]] = {}
            for value in values:
                candidates_by_year.setdefault(value[1].fiscal_year, []).append(value)
            by_year = {
                year: candidates[0]
                for year, candidates in candidates_by_year.items()
                if len(candidates) == 1
            }
            available = sorted(set(requested_years) & set(by_year) if requested_years else by_year)
            if len(available) < 2:
                continue
            chosen_years = available if requested_years else available[-2:]
            selected = [by_year[year] for year in chosen_years]
            signature = tuple(sorted(str(document.id) for chunk, document in selected)) + (
                semantic_identity(selected[0][0]) or "",
            )
            if signature in used_signatures:
                continue
            bundles.append(self._bundle("CROSS_YEAR", selected, len(bundles)))
            used_signatures.add(signature)
            if sum(bundle["scenario_type"] == "CROSS_YEAR" for bundle in bundles) >= requested.get("cross_year", 0):
                break

        identity_groups: dict[str, list[tuple[Chunk, Document]]] = {}
        for row in comparable:
            identity_groups.setdefault(semantic_identity(row[0]) or "", []).append(row)
        for values in identity_groups.values():
            if requested.get("cross_document", 0) <= 0:
                break
            candidates_by_document: dict[UUID, list[tuple[Chunk, Document]]] = {}
            for value in values:
                candidates_by_document.setdefault(value[1].id, []).append(value)
            distinct = {
                document_id: candidates[0]
                for document_id, candidates in candidates_by_document.items()
                if len(candidates) == 1
            }
            if len(distinct) < 2:
                continue
            selected = list(distinct.values())[:2]
            signature = tuple(sorted(str(document.id) for chunk, document in selected)) + (
                semantic_identity(selected[0][0]) or "",
            )
            if signature in used_signatures:
                continue
            bundles.append(self._bundle("CROSS_DOCUMENT", selected, len(bundles)))
            used_signatures.add(signature)
            if sum(bundle["scenario_type"] == "CROSS_DOCUMENT" for bundle in bundles) >= requested.get("cross_document", 0):
                break
        return bundles

    def _bundle(
        self,
        scenario_type: str,
        rows: list[tuple[Chunk, Document]],
        bundle_index: int,
    ) -> dict:
        return {
            "scenario_type": scenario_type,
            "rows": rows,
            "payload": {
                "bundle_index": bundle_index,
                "scenario_type": scenario_type,
                "sources": [
                    self._source_payload(f"S{index}", chunk, document)
                    for index, (chunk, document) in enumerate(rows, start=1)
                ],
            },
        }

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
                for stage in (
                    "vector_results",
                    "lexical_results",
                    "structured_results",
                    "rrf_results",
                    "reranked_results",
                ):
                    for hit in retrieval_payload[stage]:
                        document = document_by_id.get(str(hit["document_id"]))
                        hit["content_hash"] = content_hash(hit["content"])
                        hit["document_sha256"] = document.source_sha256 if document else None
                citations = [item.model_dump(mode="json") for item in answer["citations"]]
                hit_by_id = {
                    str(hit["chunk_id"]): hit
                    for hit in retrieval_payload["reranked_results"]
                }
                scoring_citations = [
                    {
                        **citation,
                        "parent_key": hit_by_id.get(
                            str(citation["chunk_id"]), {}
                        ).get("parent_key"),
                        "comparison_key": hit_by_id.get(
                            str(citation["chunk_id"]), {}
                        ).get("comparison_key"),
                        "document_sha256": hit_by_id.get(
                            str(citation["chunk_id"]), {}
                        ).get("document_sha256"),
                    }
                    for citation in citations
                ]
                deterministic = score_case(
                    expected_answer=case.expected_answer,
                    expected_insufficient=case.expected_insufficient,
                    required_evidence=case.required_evidence_json,
                    retrieval_hits=retrieval_payload["reranked_results"],
                    actual_answer=answer["answer"],
                    actual_insufficient=answer["insufficient_evidence"],
                    citations=scoring_citations,
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
        if targets and not metrics["complete_hit_at_5"]:
            def complete(stage: str) -> bool:
                return all(
                    any(evidence_target_matches(target, hit) for hit in retrieval[stage])
                    for target in targets
                )

            if complete("rrf_results"):
                return "RERANK_FAILURE"
            if (
                complete("vector_results")
                or complete("lexical_results")
                or complete("structured_results")
            ):
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
        if (
            case.expected_insufficient
            and metrics["insufficient_accuracy"]
            and judge.get("correctness", 0) >= 0.8
            and judge.get("completeness", 0) >= 0.8
        ):
            return None
        if judge.get("faithfulness", 1) < 0.8:
            return "ANSWER_HALLUCINATION"
        if judge.get("correctness", 1) < 0.8 or judge.get("completeness", 1) < 0.8:
            return "ANSWER_INCORRECT"
        return None


__all__ = ["EvaluationService", "_case_response", "content_hash", "numeric_tokens"]
