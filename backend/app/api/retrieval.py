from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Document
from app.retrieval.service import RetrievalService
from app.schemas import (
    IndexRequest,
    IndexResponse,
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


@router.post("/search", response_model=RetrievalDebugResponse)
async def search(
    request: RetrievalRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    document = await _document(request.document_id, session)
    try:
        return await RetrievalService().search(
            document, request.query, request.mode, session
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
