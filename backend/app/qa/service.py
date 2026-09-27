import uuid
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, QARetrieval
from app.qa.provider import DeepSeekChatProvider
from app.retrieval.service import RetrievalService
from app.schemas import QACitation, QAEvidence, RetrievalDebugResponse


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


class QAService:
    async def retrieve(
        self,
        *,
        documents: list[Document],
        question: str,
        mode: str,
        session: AsyncSession,
        knowledge_base_id: UUID | None,
    ) -> tuple[QARetrieval, RetrievalDebugResponse, list[QAEvidence]]:
        retrieval_payload = await RetrievalService().search(
            documents,
            question,
            mode,
            session,
            knowledge_base_id=knowledge_base_id,
        )
        retrieval = RetrievalDebugResponse.model_validate(retrieval_payload)
        evidence = build_qa_evidence(retrieval)
        snapshot = QARetrieval(
            id=uuid.uuid4(),
            question=question,
            retrieval_json=retrieval.model_dump(mode="json"),
            evidence_json=[item.model_dump(mode="json") for item in evidence],
        )
        session.add(snapshot)
        await session.flush()
        return snapshot, retrieval, evidence

    async def answer(self, snapshot: QARetrieval) -> dict:
        evidence = [QAEvidence.model_validate(item) for item in snapshot.evidence_json]
        provider = DeepSeekChatProvider()
        proposal = await provider.answer(snapshot.question, evidence)
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
            "retrieval_id": snapshot.id,
            "answer": proposal.answer,
            "insufficient_evidence": proposal.insufficient_evidence,
            "citations": citations,
            "model": provider.model,
        }
