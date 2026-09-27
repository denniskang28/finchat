import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Document, IngestionJob
from app.retrieval.service import RetrievalService
from app.schemas import (
    IndexRequest,
    IndexResponse,
    IngestionJobSummary,
    RetrievalDebugResponse,
    RetrievalEvaluationResponse,
    RetrievalRequest,
)


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


async def _document(document_id: UUID, session: AsyncSession) -> Document:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


async def resolve_documents(
    request: RetrievalRequest, session: AsyncSession
) -> list[Document]:
    statement = select(Document).where(Document.status == "READY")
    if request.document_id:
        statement = statement.where(Document.id == request.document_id)
    elif request.document_ids:
        statement = statement.where(Document.id.in_(request.document_ids))
    else:
        statement = statement.where(Document.knowledge_base_id == request.knowledge_base_id)
    if request.company:
        statement = statement.where(Document.company.ilike(request.company.strip()))
    if request.fiscal_year:
        statement = statement.where(Document.fiscal_year == request.fiscal_year)
    if request.document_type:
        statement = statement.where(Document.document_type == request.document_type.strip())
    documents = list((await session.execute(statement.order_by(Document.created_at))).scalars())
    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No ready documents matched the requested retrieval scope.",
        )
    return documents


@router.post("/documents/{document_id}/index", response_model=IndexResponse)
async def index_document(
    document_id: UUID,
    request: IndexRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    await _document(document_id, session)
    try:
        return await RetrievalService().index_document(
            document_id, session, force=request.force
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/documents/{document_id}/index-async",
    response_model=IngestionJobSummary,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_document_async(
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> IngestionJob:
    document = await _document(document_id, session)
    if document.status not in {"READY", "PARSED"}:
        raise HTTPException(status_code=409, detail="Document is not ready to index")
    active_job = (
        await session.execute(
            select(IngestionJob).where(
                IngestionJob.document_id == document_id,
                IngestionJob.status.in_(["PENDING", "RUNNING"]),
            )
        )
    ).scalar_one_or_none()
    if active_job is not None:
        return active_job
    job = IngestionJob(
        id=uuid.uuid4(),
        document_id=document_id,
        status="PENDING",
        stage="INDEX_QUEUED",
        progress=0,
        payload_json={"job_type": "INDEX"},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


@router.post("/search", response_model=RetrievalDebugResponse)
async def search(
    request: RetrievalRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    documents = await resolve_documents(request, session)
    try:
        return await RetrievalService().search(
            documents,
            request.query,
            request.mode,
            session,
            knowledge_base_id=request.knowledge_base_id,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/documents/{document_id}/evaluate",
    response_model=RetrievalEvaluationResponse,
)
async def evaluate(
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    document = await _document(document_id, session)
    try:
        return await RetrievalService().evaluate(document, session)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
