import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.documents import _document_response
from app.db import get_session
from app.ingestion.service import IngestionService
from app.models import Document, IngestionJob, KnowledgeBase
from app.schemas import (
    AsyncUploadResponse,
    BulkIndexResponse,
    DocumentSummary,
    IngestionJobSummary,
    KnowledgeBaseCreate,
    KnowledgeBaseSummary,
)


router = APIRouter(prefix="/api/knowledge-bases", tags=["knowledge-bases"])
jobs_router = APIRouter(prefix="/api/ingestion-jobs", tags=["ingestion-jobs"])


def _job_response(job: IngestionJob) -> dict:
    return {
        "id": job.id,
        "document_id": job.document_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "attempts": job.attempts,
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


@router.get("", response_model=list[KnowledgeBaseSummary])
async def list_knowledge_bases(
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    rows = (
        await session.execute(
            select(
                KnowledgeBase,
                func.count(Document.id).label("document_count"),
                func.count(Document.id).filter(Document.status == "READY").label(
                    "ready_document_count"
                ),
            )
            .outerjoin(Document, Document.knowledge_base_id == KnowledgeBase.id)
            .group_by(KnowledgeBase.id)
            .order_by(KnowledgeBase.created_at, KnowledgeBase.name)
        )
    ).all()
    return [
        {
            "id": knowledge_base.id,
            "name": knowledge_base.name,
            "description": knowledge_base.description,
            "document_count": document_count,
            "ready_document_count": ready_document_count,
            "created_at": knowledge_base.created_at,
        }
        for knowledge_base, document_count, ready_document_count in rows
    ]


@router.post("", response_model=KnowledgeBaseSummary, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    request: KnowledgeBaseCreate,
    session: AsyncSession = Depends(get_session),
) -> dict:
    knowledge_base = KnowledgeBase(
        id=uuid.uuid4(),
        name=request.name.strip(),
        description=request.description.strip() if request.description else None,
    )
    session.add(knowledge_base)
    await session.commit()
    await session.refresh(knowledge_base)
    return {
        "id": knowledge_base.id,
        "name": knowledge_base.name,
        "description": knowledge_base.description,
        "document_count": 0,
        "ready_document_count": 0,
        "created_at": knowledge_base.created_at,
    }


@router.get("/{knowledge_base_id}/documents", response_model=list[DocumentSummary])
async def list_knowledge_base_documents(
    knowledge_base_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    if await session.get(KnowledgeBase, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    documents = (
        await session.execute(
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
            .order_by(Document.created_at.desc())
        )
    ).scalars()
    return [_document_response(document) for document in documents]


@router.post(
    "/{knowledge_base_id}/documents",
    response_model=AsyncUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document_async(
    knowledge_base_id: UUID,
    file: UploadFile = File(...),
    provider: str | None = Form(None),
    company: str | None = Form(None),
    fiscal_year: int | None = Form(None),
    document_type: str | None = Form(None),
    language: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
) -> dict:
    if await session.get(KnowledgeBase, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    try:
        service = IngestionService(provider_name=provider)
        document = await service.create_document(
            file,
            session,
            knowledge_base_id=knowledge_base_id,
            company=company,
            fiscal_year=fiscal_year,
            document_type=document_type,
            language=language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = IngestionJob(
        id=uuid.uuid4(),
        document_id=document.id,
        status="PENDING",
        stage="QUEUED",
        progress=0,
        payload_json={"job_type": "INGEST", "provider": provider},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return {"document": _document_response(document), "job": _job_response(job)}


@router.post(
    "/{knowledge_base_id}/index",
    response_model=BulkIndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_knowledge_base(
    knowledge_base_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    if await session.get(KnowledgeBase, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    documents = list(
        (
            await session.execute(
                select(Document).where(
                    Document.knowledge_base_id == knowledge_base_id,
                    Document.status.in_(["READY", "PARSED"]),
                )
            )
        ).scalars()
    )
    active_document_ids = set(
        (
            await session.execute(
                select(IngestionJob.document_id).where(
                    IngestionJob.document_id.in_([document.id for document in documents]),
                    IngestionJob.status.in_(["PENDING", "RUNNING"]),
                )
            )
        ).scalars()
    ) if documents else set()
    jobs = [
        IngestionJob(
            id=uuid.uuid4(),
            document_id=document.id,
            status="PENDING",
            stage="INDEX_QUEUED",
            progress=0,
            payload_json={"job_type": "INDEX"},
        )
        for document in documents
        if document.id not in active_document_ids
    ]
    session.add_all(jobs)
    await session.commit()
    return {
        "knowledge_base_id": knowledge_base_id,
        "queued_jobs": len(jobs),
        "skipped_active_jobs": len(active_document_ids),
        "job_ids": [job.id for job in jobs],
    }


@jobs_router.get("/{job_id}", response_model=IngestionJobSummary)
async def get_ingestion_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    job = await session.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return _job_response(job)
