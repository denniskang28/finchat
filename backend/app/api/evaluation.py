import csv
import io
import json
import uuid
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.evaluation.service import EvaluationService, _case_response
from app.evaluation.provider import generation_provider_catalog
from app.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationRun,
    Document,
    KnowledgeBase,
)
from app.schemas import (
    EvaluationCaseCreate,
    EvaluationCaseResultSummary,
    EvaluationCaseSummary,
    EvaluationCaseUpdate,
    EvaluationDatasetCreate,
    EvaluationDatasetDetail,
    EvaluationDatasetSummary,
    EvaluationGenerateRequest,
    EvaluationImportResponse,
    EvaluationRunCreate,
    EvaluationRunDetail,
    EvaluationRunSummary,
)


router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/generation-providers")
async def list_generation_providers() -> list[dict]:
    return generation_provider_catalog()


def _dataset_response(dataset: EvaluationDataset, case_count: int, approved_count: int) -> dict:
    return {
        "id": dataset.id,
        "knowledge_base_id": dataset.knowledge_base_id,
        "name": dataset.name,
        "description": dataset.description,
        "status": dataset.status,
        "version": dataset.version,
        "case_count": case_count,
        "approved_case_count": approved_count,
        "created_at": dataset.created_at,
        "published_at": dataset.published_at,
    }


def _run_response(run: EvaluationRun) -> dict:
    return {
        "id": run.id,
        "dataset_id": run.dataset_id,
        "knowledge_base_id": run.knowledge_base_id,
        "status": run.status,
        "retrieval_mode": run.retrieval_mode,
        "progress": run.progress,
        "config": run.config_json,
        "metrics": run.metrics_json,
        "error": run.error,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


async def _dataset(dataset_id: UUID, session: AsyncSession) -> EvaluationDataset:
    dataset = await session.get(EvaluationDataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")
    return dataset


def _ensure_draft(dataset: EvaluationDataset) -> None:
    if dataset.status != "DRAFT":
        raise HTTPException(
            status_code=409,
            detail="Published evaluation datasets are immutable; clone a new version to edit.",
        )


@router.get("/knowledge-bases/{knowledge_base_id}/datasets", response_model=list[EvaluationDatasetSummary])
async def list_datasets(
    knowledge_base_id: UUID, session: AsyncSession = Depends(get_session)
) -> list[dict]:
    rows = (
        await session.execute(
            select(
                EvaluationDataset,
                func.count(EvaluationCase.id).label("case_count"),
                func.count(EvaluationCase.id)
                .filter(EvaluationCase.status == "APPROVED")
                .label("approved_count"),
            )
            .outerjoin(EvaluationCase, EvaluationCase.dataset_id == EvaluationDataset.id)
            .where(EvaluationDataset.knowledge_base_id == knowledge_base_id)
            .group_by(EvaluationDataset.id)
            .order_by(EvaluationDataset.created_at.desc())
        )
    ).all()
    return [_dataset_response(dataset, count, approved) for dataset, count, approved in rows]


@router.post("/datasets", response_model=EvaluationDatasetSummary, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    request: EvaluationDatasetCreate, session: AsyncSession = Depends(get_session)
) -> dict:
    if await session.get(KnowledgeBase, request.knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    dataset = EvaluationDataset(
        id=uuid.uuid4(),
        knowledge_base_id=request.knowledge_base_id,
        name=request.name.strip(),
        description=request.description.strip() if request.description else None,
        status="DRAFT",
        version=1,
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return _dataset_response(dataset, 0, 0)


@router.get("/datasets/{dataset_id}", response_model=EvaluationDatasetDetail)
async def get_dataset(
    dataset_id: UUID, session: AsyncSession = Depends(get_session)
) -> dict:
    dataset = await _dataset(dataset_id, session)
    cases = list(
        (
            await session.execute(
                select(EvaluationCase)
                .where(EvaluationCase.dataset_id == dataset.id)
                .order_by(EvaluationCase.created_at, EvaluationCase.id)
            )
        ).scalars()
    )
    approved = sum(case.status == "APPROVED" for case in cases)
    return {**_dataset_response(dataset, len(cases), approved), "cases": [_case_response(case) for case in cases]}


@router.post("/datasets/{dataset_id}/cases", response_model=EvaluationCaseSummary, status_code=status.HTTP_201_CREATED)
async def create_case(
    dataset_id: UUID,
    request: EvaluationCaseCreate,
    session: AsyncSession = Depends(get_session),
) -> dict:
    dataset = await _dataset(dataset_id, session)
    _ensure_draft(dataset)
    case = EvaluationCase(
        id=uuid.uuid4(),
        dataset_id=dataset.id,
        question=request.question.strip(),
        expected_answer=request.expected_answer.strip(),
        language=request.language.strip().lower(),
        expected_insufficient=request.expected_insufficient,
        required_evidence_json=[item.model_dump(mode="json") for item in request.required_evidence],
        scope_json=request.scope,
        tags_json=list(dict.fromkeys(tag.strip() for tag in request.tags if tag.strip())),
        difficulty=request.difficulty,
        source="MANUAL",
        status=request.status,
    )
    session.add(case)
    await session.commit()
    await session.refresh(case)
    return _case_response(case)


@router.patch("/cases/{case_id}", response_model=EvaluationCaseSummary)
async def update_case(
    case_id: UUID,
    request: EvaluationCaseUpdate,
    session: AsyncSession = Depends(get_session),
) -> dict:
    case = await session.get(EvaluationCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Evaluation case not found")
    dataset = await _dataset(case.dataset_id, session)
    _ensure_draft(dataset)
    changes = request.model_dump(exclude_unset=True)
    mapping = {
        "required_evidence": "required_evidence_json",
        "scope": "scope_json",
        "tags": "tags_json",
    }
    for field, value in changes.items():
        if field == "required_evidence":
            value = [item.model_dump(mode="json") for item in value]
        setattr(case, mapping.get(field, field), value)
    await session.commit()
    await session.refresh(case)
    return _case_response(case)


@router.post("/datasets/{dataset_id}/generate", response_model=list[EvaluationCaseSummary])
async def generate_cases(
    dataset_id: UUID,
    request: EvaluationGenerateRequest,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    dataset = await _dataset(dataset_id, session)
    _ensure_draft(dataset)
    try:
        cases = await EvaluationService().generate_cases(
            dataset,
            count=request.count,
            language=request.language,
            difficulty=request.difficulty,
            include_insufficient=request.include_insufficient,
            scenario_mix=(
                request.scenario_mix.model_dump() if request.scenario_mix else None
            ),
            generation_provider=request.generation_provider,
            generation_model=request.generation_model,
            document_ids=request.document_ids,
            years=request.years,
            company=request.company,
            allow_calculations=request.allow_calculations,
            session=session,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return [_case_response(case) for case in cases]


@router.post("/datasets/{dataset_id}/import", response_model=EvaluationImportResponse)
async def import_cases(
    dataset_id: UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> dict:
    dataset = await _dataset(dataset_id, session)
    _ensure_draft(dataset)
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Import file must be UTF-8.") from exc
    try:
        if file.filename and file.filename.lower().endswith(".csv"):
            records = list(csv.DictReader(io.StringIO(text)))
            for record in records:
                for field in ("required_evidence", "scope", "tags"):
                    if record.get(field):
                        record[field] = json.loads(record[field])
                if "expected_insufficient" in record:
                    record["expected_insufficient"] = str(record["expected_insufficient"]).lower() in {"1", "true", "yes"}
        else:
            payload = json.loads(text)
            records = payload.get("cases", []) if isinstance(payload, dict) else payload
    except (json.JSONDecodeError, csv.Error, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid import file: {exc}") from exc
    if not isinstance(records, list):
        raise HTTPException(status_code=400, detail="Import must contain a list of cases.")

    imported: list[EvaluationCase] = []
    errors = []
    for index, record in enumerate(records, start=1):
        try:
            parsed = EvaluationCaseCreate.model_validate(record)
        except ValidationError as exc:
            errors.append(f"Row {index}: {exc.errors()[0]['msg']}")
            continue
        imported.append(
            EvaluationCase(
                id=uuid.uuid4(),
                dataset_id=dataset.id,
                question=parsed.question.strip(),
                expected_answer=parsed.expected_answer.strip(),
                language=parsed.language,
                expected_insufficient=parsed.expected_insufficient,
                required_evidence_json=[item.model_dump(mode="json") for item in parsed.required_evidence],
                scope_json=parsed.scope,
                tags_json=parsed.tags,
                difficulty=parsed.difficulty,
                source="IMPORT",
                status=parsed.status,
            )
        )
    session.add_all(imported)
    await session.commit()
    return {"imported": len(imported), "rejected": len(errors), "errors": errors[:50]}


@router.post("/datasets/{dataset_id}/publish", response_model=EvaluationDatasetSummary)
async def publish_dataset(
    dataset_id: UUID, session: AsyncSession = Depends(get_session)
) -> dict:
    dataset = await _dataset(dataset_id, session)
    _ensure_draft(dataset)
    case_count, approved = (
        await session.execute(
            select(
                func.count(EvaluationCase.id),
                func.count(EvaluationCase.id).filter(EvaluationCase.status == "APPROVED"),
            ).where(EvaluationCase.dataset_id == dataset.id)
        )
    ).one()
    if approved == 0:
        raise HTTPException(status_code=409, detail="Approve at least one case before publishing.")
    dataset.status = "PUBLISHED"
    dataset.published_at = datetime.now(UTC)
    await session.commit()
    return _dataset_response(dataset, case_count, approved)


@router.post("/datasets/{dataset_id}/clone", response_model=EvaluationDatasetSummary, status_code=status.HTTP_201_CREATED)
async def clone_dataset(
    dataset_id: UUID, session: AsyncSession = Depends(get_session)
) -> dict:
    source = await _dataset(dataset_id, session)
    latest_version = (
        await session.execute(
            select(func.max(EvaluationDataset.version)).where(
                EvaluationDataset.knowledge_base_id == source.knowledge_base_id,
                EvaluationDataset.name == source.name,
            )
        )
    ).scalar_one() or source.version
    clone = EvaluationDataset(
        id=uuid.uuid4(),
        knowledge_base_id=source.knowledge_base_id,
        name=source.name,
        description=source.description,
        status="DRAFT",
        version=latest_version + 1,
    )
    session.add(clone)
    await session.flush()
    source_cases = list(
        (
            await session.execute(select(EvaluationCase).where(EvaluationCase.dataset_id == source.id))
        ).scalars()
    )
    clones = [
        EvaluationCase(
            id=uuid.uuid4(),
            dataset_id=clone.id,
            question=case.question,
            expected_answer=case.expected_answer,
            language=case.language,
            expected_insufficient=case.expected_insufficient,
            required_evidence_json=case.required_evidence_json,
            scope_json=case.scope_json,
            tags_json=case.tags_json,
            generation_metadata_json=case.generation_metadata_json,
            difficulty=case.difficulty,
            source=case.source,
            status=case.status,
        )
        for case in source_cases
    ]
    session.add_all(clones)
    await session.commit()
    await session.refresh(clone)
    approved = sum(case.status == "APPROVED" for case in clones)
    return _dataset_response(clone, len(clones), approved)


@router.post("/runs", response_model=EvaluationRunSummary, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    request: EvaluationRunCreate, session: AsyncSession = Depends(get_session)
) -> dict:
    dataset = await _dataset(request.dataset_id, session)
    if dataset.status != "PUBLISHED":
        raise HTTPException(status_code=409, detail="Publish the dataset before running it.")
    settings = get_settings()
    documents = list(
        (
            await session.execute(
                select(Document).where(
                    Document.knowledge_base_id == dataset.knowledge_base_id,
                    Document.status == "READY",
                )
            )
        ).scalars()
    )
    run = EvaluationRun(
        id=uuid.uuid4(),
        dataset_id=dataset.id,
        knowledge_base_id=dataset.knowledge_base_id,
        status="PENDING",
        retrieval_mode=request.retrieval_mode,
        progress=0,
        config_json={
            "dataset_version": dataset.version,
            "embedding_model": settings.alibaba_embedding_model,
            "rerank_model": settings.alibaba_rerank_model,
            "answer_model": settings.deepseek_chat_model,
            "judge_model": settings.alibaba_evaluation_model,
            "lexical_top_k": settings.retrieval_lexical_top_k,
            "vector_top_k": settings.retrieval_vector_top_k,
            "rrf_top_k": settings.retrieval_rrf_top_k,
            "final_top_k": settings.retrieval_final_top_k,
            "rrf_k": settings.retrieval_rrf_k,
            "documents": [
                {
                    "id": str(document.id),
                    "filename": document.filename,
                    "source_sha256": document.source_sha256,
                }
                for document in documents
            ],
        },
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    return _run_response(run)


@router.get("/knowledge-bases/{knowledge_base_id}/runs", response_model=list[EvaluationRunSummary])
async def list_runs(
    knowledge_base_id: UUID, session: AsyncSession = Depends(get_session)
) -> list[dict]:
    runs = (
        await session.execute(
            select(EvaluationRun)
            .where(EvaluationRun.knowledge_base_id == knowledge_base_id)
            .order_by(EvaluationRun.created_at.desc())
        )
    ).scalars()
    return [_run_response(run) for run in runs]


@router.get("/runs/{run_id}", response_model=EvaluationRunDetail)
async def get_run(run_id: UUID, session: AsyncSession = Depends(get_session)) -> dict:
    run = await session.get(EvaluationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    rows = (
        await session.execute(
            select(EvaluationCaseResult, EvaluationCase)
            .join(EvaluationCase, EvaluationCase.id == EvaluationCaseResult.case_id)
            .where(EvaluationCaseResult.run_id == run.id)
            .order_by(EvaluationCaseResult.created_at, EvaluationCaseResult.id)
        )
    ).all()
    results = [
        {
            "id": result.id,
            "run_id": result.run_id,
            "case_id": result.case_id,
            "status": result.status,
            "question": case.question,
            "expected_answer": case.expected_answer,
            "retrieval": result.retrieval_json,
            "final_evidence": result.final_evidence_json,
            "answer": result.answer,
            "insufficient_evidence": result.insufficient_evidence,
            "citations": result.citations_json,
            "deterministic_metrics": result.deterministic_metrics_json,
            "judge": result.judge_json,
            "failure_stage": result.failure_stage,
            "latency_seconds": result.latency_seconds,
            "error": result.error,
        }
        for result, case in rows
    ]
    return {**_run_response(run), "results": results}
