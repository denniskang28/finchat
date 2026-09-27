from __future__ import annotations

import asyncio
import unicodedata
from collections import defaultdict
from typing import Literal
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.ingestion.renderers import render_table_summary
from app.models import Chunk, Document
from app.retrieval.golden import GOLDEN_QUERIES
from app.retrieval.providers import AlibabaRetrievalProvider
from app.schemas import ParsedTable

RetrievalMode = Literal["BASELINE", "SEMANTIC"]


def reciprocal_rank_fusion(
    rankings: list[list[UUID]], *, k: int = 60, limit: int = 20
) -> tuple[list[UUID], dict[UUID, float]]:
    scores: dict[UUID, float] = defaultdict(float)
    best_ranks: dict[UUID, int] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
            best_ranks[chunk_id] = min(rank, best_ranks.get(chunk_id, rank))
    ordered = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], best_ranks[chunk_id], str(chunk_id)),
    )[:limit]
    return ordered, dict(scores)


def grade_targets(
    ranks_by_key: dict[str | None, int], targets: list[str]
) -> tuple[dict[str, int | None], int | None, int | None]:
    target_ranks = {target: ranks_by_key.get(target) for target in targets}
    present = [rank for rank in target_ranks.values() if rank is not None]
    first_relevant_rank = min(present) if present else None
    complete_rank = max(present) if len(present) == len(target_ranks) else None
    return target_ranks, first_relevant_rank, complete_rank


def _candidate_filter(mode: RetrievalMode):
    row_representation = "RAW_ROW" if mode == "BASELINE" else "SEMANTIC_ROW"
    return or_(
        and_(Chunk.chunk_type == "TEXT", Chunk.representation == "TEXT"),
        Chunk.representation == row_representation,
        *([Chunk.representation == "SUMMARY"] if mode == "SEMANTIC" else []),
    )


def _indexable_filter():
    return or_(
        and_(Chunk.chunk_type == "TEXT", Chunk.representation == "TEXT"),
        Chunk.representation.in_(["RAW_ROW", "SEMANTIC_ROW", "SUMMARY"]),
    )


class RetrievalService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = AlibabaRetrievalProvider(self.settings)

    async def index_document(
        self,
        document_id: UUID,
        session: AsyncSession,
        *,
        force: bool = False,
    ) -> dict:
        await self._ensure_table_summaries(document_id, session)
        statement = (
            select(Chunk)
            .where(Chunk.document_id == document_id, _indexable_filter())
            .order_by(Chunk.id)
        )
        chunks = list((await session.execute(statement)).scalars())
        pending = chunks if force else [
            chunk
            for chunk in chunks
            if chunk.embedding is None
            or chunk.metadata_json.get("embedding_model") != self.provider.embedding_model
        ]
        batches = [pending[index:index + 10] for index in range(0, len(pending), 10)]
        semaphore = asyncio.Semaphore(4)

        async def embed_batch(batch: list[Chunk]) -> tuple[list[Chunk], list[list[float]]]:
            async with semaphore:
                vectors = await self.provider.embed(
                    [chunk.content for chunk in batch], text_type="document"
                )
                return batch, vectors

        for batch, vectors in await asyncio.gather(*(embed_batch(batch) for batch in batches)):
            for chunk, vector in zip(batch, vectors, strict=True):
                chunk.embedding = vector
                chunk.metadata_json = {
                    **chunk.metadata_json,
                    "embedding_model": self.provider.embedding_model,
                    "embedding_dimensions": self.provider.embedding_dimensions,
                }
        await session.commit()
        return {
            "document_id": document_id,
            "model": self.provider.embedding_model,
            "dimensions": self.provider.embedding_dimensions,
            "indexed_chunks": len(pending),
            "skipped_chunks": len(chunks) - len(pending),
            "total_indexable_chunks": len(chunks),
        }

    async def _ensure_table_summaries(
        self, document_id: UUID, session: AsyncSession
    ) -> None:
        existing = set(
            (
                await session.execute(
                    select(Chunk.comparison_key).where(
                        Chunk.document_id == document_id,
                        Chunk.representation == "SUMMARY",
                    )
                )
            ).scalars()
        )
        markdown_chunks = list(
            (
                await session.execute(
                    select(Chunk).where(
                        Chunk.document_id == document_id,
                        Chunk.chunk_type == "TABLE_MARKDOWN",
                    )
                )
            ).scalars()
        )
        for markdown in markdown_chunks:
            debug = markdown.metadata_json.get("table_debug") or {}
            parsed_payload = debug.get("parsed_table")
            if not parsed_payload:
                continue
            table = ParsedTable.model_validate(parsed_payload)
            comparison_key = f"{table.table_id}:summary"
            if comparison_key in existing:
                continue
            content = render_table_summary(table)
            session.add(
                Chunk(
                    document_id=document_id,
                    page_number=table.page_number,
                    chunk_type="TABLE_SUMMARY",
                    representation="SUMMARY",
                    comparison_key=comparison_key,
                    content=content,
                    raw_content=markdown.content,
                    semantic_content=content,
                    metadata_json={
                        "element_type": "TABLE_SUMMARY",
                        "table_id": table.table_id,
                        "table_title": table.title,
                        "source_kind": table.source_kind,
                    },
                )
            )
            existing.add(comparison_key)
        await session.flush()

    async def search(
        self,
        document: Document,
        query: str,
        mode: RetrievalMode,
        session: AsyncSession,
    ) -> dict:
        query = unicodedata.normalize("NFC", query.strip())
        if not query:
            raise ValueError("Query cannot be empty.")
        query_vector = (await self.provider.embed([query], text_type="query"))[0]
        lexical_limit = self.settings.retrieval_lexical_top_k
        vector_limit = self.settings.retrieval_vector_top_k

        tsquery = func.websearch_to_tsquery("simple", query)
        lexical_score_expr = func.ts_rank_cd(Chunk.search_vector, tsquery, 32).label(
            "lexical_score"
        )
        lexical_rows = (
            await session.execute(
                select(Chunk, lexical_score_expr)
                .where(
                    Chunk.document_id == document.id,
                    _candidate_filter(mode),
                    Chunk.search_vector.op("@@")(tsquery),
                )
                .order_by(lexical_score_expr.desc(), Chunk.id)
                .limit(lexical_limit)
            )
        ).all()

        distance_expr = Chunk.embedding.cosine_distance(query_vector).label("distance")
        vector_rows = (
            await session.execute(
                select(Chunk, distance_expr)
                .where(
                    Chunk.document_id == document.id,
                    _candidate_filter(mode),
                    Chunk.embedding.is_not(None),
                )
                .order_by(distance_expr, Chunk.id)
                .limit(vector_limit)
            )
        ).all()

        lexical_chunks = [row[0] for row in lexical_rows]
        vector_chunks = [row[0] for row in vector_rows]
        lexical_scores = {chunk.id: float(score) for chunk, score in lexical_rows}
        vector_scores = {chunk.id: 1.0 - float(distance) for chunk, distance in vector_rows}

        chunk_by_id = {chunk.id: chunk for chunk in [*lexical_chunks, *vector_chunks]}
        rrf_ids, rrf_scores = reciprocal_rank_fusion(
            [[chunk.id for chunk in lexical_chunks], [chunk.id for chunk in vector_chunks]],
            k=self.settings.retrieval_rrf_k,
            limit=self.settings.retrieval_rrf_top_k,
        )
        rrf_chunks = [
            chunk_by_id[chunk_id]
            for chunk_id in rrf_ids
        ]

        rerank_pairs = await self.provider.rerank(query, [chunk.content for chunk in rrf_chunks])
        reranked_all = [rrf_chunks[index] for index, _ in rerank_pairs]
        rerank_scores = {
            rrf_chunks[index].id: score for index, score in rerank_pairs
        }
        final_chunks = reranked_all[: self.settings.retrieval_final_top_k]
        final_ranks = {chunk.id: rank for rank, chunk in enumerate(final_chunks, start=1)}

        def hits(chunks: list[Chunk]) -> list[dict]:
            return [
                {
                    "chunk_id": chunk.id,
                    "rank": rank,
                    "final_rank": final_ranks.get(chunk.id),
                    "chunk_type": chunk.chunk_type,
                    "representation": chunk.representation,
                    "comparison_key": chunk.comparison_key,
                    "file": document.filename,
                    "page": chunk.page_number,
                    "table_title": chunk.metadata_json.get("table_title"),
                    "row_label": chunk.metadata_json.get("row_label"),
                    "raw_content": chunk.raw_content,
                    "semantic_content": chunk.semantic_content,
                    "content": chunk.content,
                    "vector_score": vector_scores.get(chunk.id),
                    "lexical_score": lexical_scores.get(chunk.id),
                    "rrf_score": rrf_scores.get(chunk.id),
                    "rerank_score": rerank_scores.get(chunk.id),
                }
                for rank, chunk in enumerate(chunks, start=1)
            ]

        return {
            "original_query": query,
            "retrieval_mode": mode,
            "document_id": document.id,
            "embedding_model": self.provider.embedding_model,
            "rerank_model": self.provider.rerank_model,
            "vector_results": hits(vector_chunks),
            "lexical_results": hits(lexical_chunks),
            "rrf_results": hits(rrf_chunks),
            "reranked_results": hits(final_chunks),
        }

    async def evaluate(self, document: Document, session: AsyncSession) -> dict:
        results = []
        for mode in ("BASELINE", "SEMANTIC"):
            for golden in GOLDEN_QUERIES:
                debug = await self.search(document, golden["query"], mode, session)
                stage_ranks = {}
                for stage in (
                    "vector_results",
                    "lexical_results",
                    "rrf_results",
                    "reranked_results",
                ):
                    ranks_by_key = {
                        hit["comparison_key"]: hit["rank"] for hit in debug[stage]
                    }
                    stage_ranks[stage] = grade_targets(
                        ranks_by_key, golden["targets"]
                    )[0]
                target_ranks, first_relevant_rank, complete_rank = grade_targets(
                    {
                        hit["comparison_key"]: hit["rank"]
                        for hit in debug["reranked_results"]
                    },
                    golden["targets"],
                )
                rrf_complete = all(
                    rank is not None
                    for rank in stage_ranks["rrf_results"].values()
                )
                failure_stage = None
                if complete_rank is None:
                    failure_stage = "RERANK" if rrf_complete else "FIRST_STAGE_OR_RRF"
                results.append(
                    {
                        "query_id": golden["id"],
                        "query": golden["query"],
                        "mode": mode,
                        "expected_target_ids": golden["targets"],
                        "vector_ranks": stage_ranks["vector_results"],
                        "lexical_ranks": stage_ranks["lexical_results"],
                        "rrf_ranks": stage_ranks["rrf_results"],
                        "rerank_ranks": stage_ranks["reranked_results"],
                        "target_ranks": target_ranks,
                        "first_relevant_rank": first_relevant_rank,
                        "first_complete_rank": complete_rank,
                        "hit_at_1": complete_rank is not None and complete_rank <= 1,
                        "hit_at_3": complete_rank is not None and complete_rank <= 3,
                        "hit_at_5": complete_rank is not None and complete_rank <= 5,
                        "reciprocal_rank": (
                            1.0 / first_relevant_rank if first_relevant_rank else 0.0
                        ),
                        "failure_stage": failure_stage,
                    }
                )

        metrics = {}
        for mode in ("BASELINE", "SEMANTIC"):
            mode_results = [result for result in results if result["mode"] == mode]
            denominator = len(mode_results) or 1
            metrics[mode] = {
                "hit_at_1": sum(item["hit_at_1"] for item in mode_results) / denominator,
                "hit_at_3": sum(item["hit_at_3"] for item in mode_results) / denominator,
                "hit_at_5": sum(item["hit_at_5"] for item in mode_results) / denominator,
                "mrr": sum(item["reciprocal_rank"] for item in mode_results) / denominator,
            }
        return {"document_id": document.id, "results": results, "metrics": metrics}
