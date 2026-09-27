from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models import Base, DEFAULT_KNOWLEDGE_BASE_ID


settings = get_settings()
engine = create_async_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.run_sync(Base.metadata.create_all)
        for ddl in (
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS knowledge_base_id UUID",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_sha256 VARCHAR(64)",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS company TEXT",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS fiscal_year INTEGER",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type VARCHAR(50)",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS language VARCHAR(20)",
            "ALTER TABLE evaluation_cases ADD COLUMN IF NOT EXISTS generation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
        ):
            await connection.execute(text(ddl))
        await connection.execute(
            text(
                "INSERT INTO knowledge_bases (id, name, description) "
                "VALUES (:id, 'Default Knowledge Base', 'Migrated and newly uploaded finance documents') "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"id": DEFAULT_KNOWLEDGE_BASE_ID},
        )
        await connection.execute(
            text("UPDATE documents SET knowledge_base_id = :id WHERE knowledge_base_id IS NULL"),
            {"id": DEFAULT_KNOWLEDGE_BASE_ID},
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_documents_knowledge_base_id "
                "ON documents (knowledge_base_id)"
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_documents_finance_metadata "
                "ON documents (knowledge_base_id, company, fiscal_year, document_type)"
            )
        )
        await connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_kb_source_sha256 "
                "ON documents (knowledge_base_id, source_sha256) "
                "WHERE source_sha256 IS NOT NULL AND status != 'FAILED'"
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_ingestion_jobs_pending "
                "ON ingestion_jobs (status, created_at)"
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_evaluation_datasets_kb "
                "ON evaluation_datasets (knowledge_base_id, created_at)"
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_evaluation_cases_dataset "
                "ON evaluation_cases (dataset_id, status)"
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_evaluation_runs_pending "
                "ON evaluation_runs (status, created_at)"
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_evaluation_results_run "
                "ON evaluation_case_results (run_id, created_at)"
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_search_vector_gin "
                "ON chunks USING gin (search_vector)"
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops) "
                "WHERE embedding IS NOT NULL"
            )
        )


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
