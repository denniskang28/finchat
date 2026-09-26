from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.ingestion.service import IngestionService
from app.models import Chunk, Document
from app.schemas import DocumentSummary, PageText, TableDebug, UploadResponse


router = APIRouter(prefix="/api/documents", tags=["documents"])


def _document_response(document: Document) -> dict:
    return {
        "id": document.id,
        "filename": document.filename,
        "title": document.title,
        "status": document.status,
        "page_count": document.page_count,
        "metadata": document.metadata_json,
        "created_at": document.created_at,
    }


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...), session: AsyncSession = Depends(get_session)
) -> dict:
    try:
        document = await IngestionService().ingest(file, session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {exc}") from exc
    return _document_response(document)


@router.get("", response_model=list[DocumentSummary])
async def list_documents(session: AsyncSession = Depends(get_session)) -> list[dict]:
    result = await session.execute(select(Document).order_by(Document.created_at.desc()))
    return [_document_response(document) for document in result.scalars()]


async def _get_document(document_id: UUID, session: AsyncSession) -> Document:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}", response_model=DocumentSummary)
async def get_document(document_id: UUID, session: AsyncSession = Depends(get_session)) -> dict:
    return _document_response(await _get_document(document_id, session))


@router.get("/{document_id}/tables", response_model=list[TableDebug])
async def list_tables(
    document_id: UUID, session: AsyncSession = Depends(get_session)
) -> list[dict]:
    await _get_document(document_id, session)
    result = await session.execute(
        select(Chunk)
        .where(
            Chunk.document_id == document_id,
            Chunk.chunk_type == "TABLE_MARKDOWN",
        )
        .order_by(Chunk.page_number, Chunk.comparison_key)
    )
    return [chunk.metadata_json["table_debug"] for chunk in result.scalars()]


@router.get("/{document_id}/pages", response_model=list[PageText])
async def list_pages(
    document_id: UUID, session: AsyncSession = Depends(get_session)
) -> list[dict]:
    await _get_document(document_id, session)
    result = await session.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id, Chunk.chunk_type == "TEXT")
        .order_by(Chunk.page_number)
    )
    return [
        {"page_number": chunk.page_number, "text": chunk.content}
        for chunk in result.scalars()
    ]

