from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.retrieval import resolve_documents
from app.db import get_session
from app.models import QARetrieval
from app.qa.service import QAService
from app.schemas import (
    QAAnswerResponse,
    QARetrieveResponse,
    RetrievalRequest,
)


router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("/retrieve", response_model=QARetrieveResponse)
async def retrieve_for_qa(
    request: RetrievalRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    documents = await resolve_documents(request, session)
    try:
        snapshot, retrieval, evidence = await QAService().retrieve(
            documents=documents,
            question=request.query,
            mode=request.mode,
            session=session,
            knowledge_base_id=request.knowledge_base_id,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    await session.commit()
    return {
        "retrieval_id": snapshot.id,
        "retrieval": retrieval,
        "final_evidence": evidence,
    }


@router.post("/{retrieval_id}/answer", response_model=QAAnswerResponse)
async def answer_from_snapshot(
    retrieval_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    snapshot = await session.get(QARetrieval, retrieval_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="QA retrieval snapshot not found")
    try:
        return await QAService().answer(snapshot)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
