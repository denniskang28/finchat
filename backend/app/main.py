from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router as documents_router
from app.config import get_settings
from app.db import init_db
from app.ingestion.repair_providers import create_table_repair_provider


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Semantic Table RAG POC",
    version="0.1.0",
    description="PDF parsing and semantic table representation inspection only.",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)
app.include_router(documents_router)


@app.get("/api/health")
async def health() -> dict[str, object]:
    provider = create_table_repair_provider(settings)
    return {
        "status": "ok",
        "scope": "ingestion-and-table-debug",
        "table_repair": {
            "enabled": settings.table_repair_enabled,
            "parse_mode": settings.table_parse_mode,
            "page_workers": settings.table_parse_workers,
            "provider": provider.name,
            "model": provider.model,
            "llm_available": provider.available,
        },
    }
