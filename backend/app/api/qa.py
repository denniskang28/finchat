import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.retrieval import resolve_documents
from app.db import get_session
from app.models import QARetrieval
from app.qa.provider import DeepSeekChatProvider
from app.retrieval.service import RetrievalService
from app.schemas import (
    QAAnswerResponse,
    QACitation,
    QAEvidence,
    QARetrieveResponse,
    RetrievalDebugResponse,
    RetrievalRequest,
)


router = APIRouter(prefix="/api/qa", tags=["qa"])


def build_qa_evidence(retrieval: RetrievalDebugResponse) -> list[QAEvidence]:
    return [
        QAEvidence(
            evidence_number=index,
            chunk_id=hit.chunk_id,
            filename=hit.file,
            page=hit.page,
            chunk_type=hit.chunk_type,
            content=hit.content,
        )
        for index, hit in enumerate(retrieval.reranked_results, start=1)
    ]


@router.post("/retrieve", response_model=QARetrieveResponse)
async def retrieve_for_qa(
    request: RetrievalRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    documents = await resolve_documents(request, session)
    try:
        retrieval_payload = await RetrievalService().search(
            documents,
            request.query,
            request.mode,
            session,
            knowledge_base_id=request.knowledge_base_id,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    retrieval = RetrievalDebugResponse.model_validate(retrieval_payload)
    evidence = build_qa_evidence(retrieval)
    snapshot = QARetrieval(
        id=uuid.uuid4(),
        question=request.query,
        retrieval_json=retrieval.model_dump(mode="json"),
        evidence_json=[item.model_dump(mode="json") for item in evidence],
    )
    session.add(snapshot)
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
    evidence = [QAEvidence.model_validate(item) for item in snapshot.evidence_json]
    provider = DeepSeekChatProvider()
    try:
        proposal = await provider.answer(snapshot.question, evidence)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    evidence_by_number = {item.evidence_number: item for item in evidence}
    citations = [
        QACitation(
            evidence_number=number,
            filename=evidence_by_number[number].filename,
            page=evidence_by_number[number].page,
            chunk_id=evidence_by_number[number].chunk_id,
        )
        for number in proposal.citation_numbers
    ]
    return {
        "retrieval_id": retrieval_id,
        "answer": proposal.answer,
        "insufficient_evidence": proposal.insufficient_evidence,
        "citations": citations,
        "model": provider.model,
    }
