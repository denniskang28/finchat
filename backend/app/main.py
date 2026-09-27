from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router as documents_router
from app.api.evaluation import router as evaluation_router
from app.api.knowledge_bases import jobs_router, router as knowledge_bases_router
from app.api.qa import router as qa_router
from app.api.retrieval import router as retrieval_router
from app.config import get_settings
from app.db import init_db
from app.ingestion.repair_providers import create_table_repair_provider
from app.retrieval.providers import AlibabaRetrievalProvider


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Semantic Table RAG POC",
    version="0.4.0",
    description="PDF ingestion, semantic retrieval, evidence-constrained QA, and evaluation.",
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
app.include_router(knowledge_bases_router)
app.include_router(jobs_router)
app.include_router(retrieval_router)
app.include_router(qa_router)
app.include_router(evaluation_router)


@app.get("/api/health")
async def health() -> dict[str, object]:
    provider = create_table_repair_provider(settings)
    retrieval_provider = AlibabaRetrievalProvider(settings)
    return {
        "status": "ok",
        "scope": "ingestion-retrieval-qa-and-evaluation",
        "table_repair": {
            "enabled": settings.table_repair_enabled,
            "parse_mode": settings.table_parse_mode,
            "page_workers": settings.table_parse_workers,
            "provider": provider.name,
            "model": provider.model,
            "llm_available": provider.available,
        },
        "retrieval": {
            "embedding_model": retrieval_provider.embedding_model,
            "embedding_dimensions": retrieval_provider.embedding_dimensions,
            "rerank_model": retrieval_provider.rerank_model,
            "available": retrieval_provider.available,
        },
        "qa": {
            "provider": "deepseek",
            "model": settings.deepseek_chat_model,
            "available": bool(settings.deepseek_api_key),
        },
        "evaluation": {
            "provider": "alibaba",
            "model": settings.alibaba_evaluation_model,
            "available": bool(settings.dashscope_api_key),
        },
    }


@app.get("/api/providers")
async def providers() -> list[dict[str, object]]:
    default_provider = settings.table_repair_provider.strip().lower()
    return [
        {
            "id": provider_name,
            "name": display_name,
            "model": provider.model,
            "available": provider.available,
            "default": provider_name == default_provider,
        }
        for provider_name, display_name in (
            ("alibaba", "Alibaba Cloud / Qwen"),
            ("deepseek", "DeepSeek"),
        )
        for provider in [create_table_repair_provider(settings, provider_name)]
    ]
