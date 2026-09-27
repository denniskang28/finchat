import asyncio
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.ingestion.service import IngestionService
from app.models import Document, IngestionJob
from app.retrieval.service import RetrievalService


settings = get_settings()


async def claim_job() -> UUID | None:
    async with SessionLocal() as session:
        job = (
            await session.execute(
                select(IngestionJob)
                .where(
                    IngestionJob.status == "PENDING",
                    IngestionJob.attempts < settings.worker_max_attempts,
                )
                .order_by(IngestionJob.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
        ).scalar_one_or_none()
        if job is None:
            return None
        job.status = "RUNNING"
        is_index_job = job.payload_json.get("job_type") == "INDEX"
        job.stage = "EMBEDDING" if is_index_job else "PARSING"
        job.progress = 80 if is_index_job else 5
        job.attempts += 1
        job.error = None
        job.started_at = datetime.now(UTC)
        await session.commit()
        return job.id


async def process_job(job_id: UUID) -> None:
    async with SessionLocal() as session:
        job = await session.get(IngestionJob, job_id)
        if job is None:
            return
        document = await session.get(Document, job.document_id)
        if document is None:
            job.status = "FAILED"
            job.error = "Document was deleted before ingestion started."
            job.completed_at = datetime.now(UTC)
            await session.commit()
            return
        try:
            job_type = job.payload_json.get("job_type", "INGEST")
            if job_type == "INGEST":
                provider = job.payload_json.get("provider")
                await IngestionService(provider_name=provider).process_document(
                    document, session, final_status="PARSED"
                )
            job.stage = "EMBEDDING"
            job.progress = 85
            document.status = "INDEXING"
            await session.commit()
            await RetrievalService().index_document(document.id, session)
            document.status = "READY"
            job.status = "COMPLETE"
            job.stage = "READY"
            job.progress = 100
            job.completed_at = datetime.now(UTC)
            await session.commit()
        except Exception as exc:
            await session.rollback()
            job = await session.get(IngestionJob, job_id)
            document = await session.get(Document, job.document_id) if job else None
            if job is None:
                return
            job.error = f"{type(exc).__name__}: {exc}"
            if job.attempts < settings.worker_max_attempts:
                job.status = "PENDING"
                job.stage = "RETRY_QUEUED"
                job.progress = 0
                if document:
                    document.status = "READY" if job.payload_json.get("job_type") == "INDEX" else "PROCESSING"
            else:
                job.status = "FAILED"
                job.stage = "FAILED"
                job.completed_at = datetime.now(UTC)
                if document:
                    document.status = "READY" if job.payload_json.get("job_type") == "INDEX" else "FAILED"
            await session.commit()


async def run_worker() -> None:
    await init_db()
    while True:
        job_id = await claim_job()
        if job_id is None:
            await asyncio.sleep(settings.worker_poll_seconds)
            continue
        await process_job(job_id)


if __name__ == "__main__":
    asyncio.run(run_worker())
