from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from typing import Literal
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.ingestion.renderers import render_table_summary
from app.models import Chunk, Document
from app.retrieval.context import (
    document_context_metadata,
    enrich_semantic_content,
)
from app.retrieval.golden import GOLDEN_QUERIES
from app.retrieval.providers import AlibabaRetrievalProvider
from app.schemas import ParsedTable

RetrievalMode = Literal["BASELINE", "SEMANTIC", "PRODUCTION"]

logger = logging.getLogger(__name__)

QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "group", "how", "in", "into", "is", "it", "of", "on",
    "or", "report", "results", "the", "to", "was", "were", "what", "which",
    "with", "year", "aia", "annual", "fiscal",
}

RELATIVE_YEAR_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
GROUP_VONB_EXCLUSIONS = {
    "agency", "partnership", "distribution", "margin", "product",
    "singapore", "thailand", "malaysia", "hong kong", "mainland china",
    "other markets", "aia china", "aia thailand", "aia singapore",
    "mcv", "domestic",
}


def _relative_year_count(query: str) -> int | None:
    normalized = query.lower()
    match = re.search(
        r"(?:过去|近|最近)\s*([一二三四五六七八九十]|[0-9]{1,2})\s*年", normalized
    ) or re.search(
        r"(?:past|last|previous)\s+([a-z]+|[0-9]{1,2})\s+years?", normalized
    )
    if match is None:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else RELATIVE_YEAR_WORDS.get(value)


def extract_query_years(
    query: str,
    available_years: set[int],
    *,
    annual_years: set[int] | None = None,
) -> list[int]:
    requested = {int(value) for value in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", query)}
    requested.update(
        2000 + int(value)
        for value in re.findall(
            r"(?i)(?:FY|H[12]|[12]H|Q[1-4])\s*'?([0-9]{2})(?!\d)", query
        )
    )
    explicit_years = sorted(requested & available_years)
    if explicit_years:
        return explicit_years
    relative_count = _relative_year_count(query)
    if relative_count is None:
        return []
    candidate_years = sorted((annual_years or available_years) & available_years)
    return candidate_years[-relative_count:]


def lexical_query_terms(query: str) -> list[str]:
    terms = []
    for value in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", query.lower()):
        if value in QUERY_STOPWORDS or re.fullmatch(r"(?:19|20)\d{2}", value):
            continue
        if value not in terms:
            terms.append(value)
    return terms[:16]


def focused_query(query: str, year: int) -> str:
    terms = lexical_query_terms(query)
    subject = " ".join(terms) if terms else query
    return f"{subject}. Fiscal year {year}."


def is_group_vonb_time_series(query: str, query_years: list[int]) -> bool:
    normalized = query.lower()
    asks_for_change = any(value in normalized for value in ("变化", "趋势", "变动", "change", "trend"))
    group_context = any(value in normalized for value in ("aia", "集团", "group"))
    return "vonb" in normalized and len(query_years) > 1 and (asks_for_change or group_context)


def asks_for_explanation(query: str) -> bool:
    normalized = query.lower()
    return any(value in normalized for value in ("原因", "解释", "驱动", "why", "reason", "driver"))


def is_group_vonb_chunk(chunk: Chunk) -> bool:
    title = str((chunk.metadata_json or {}).get("table_title") or "").lower()
    text = f"{title} {chunk.content}".lower()
    if "vonb" not in text or any(value in text for value in GROUP_VONB_EXCLUSIONS):
        return False
    return (
        "total group vonb" in text
        or "aia group vonb" in text
        or title.strip() in {"vonb ($m)", "vonb"}
        or chunk.chunk_type in {"SECTION", "FACT"}
    )


def semantic_identity_values(chunk: Chunk) -> tuple[str, str] | None:
    metadata = chunk.metadata_json or {}
    title = str(metadata.get("table_title") or "").strip().lower()
    row_label = str(metadata.get("row_label") or "").strip().lower()
    return (title, row_label) if title and row_label else None


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
    if mode == "PRODUCTION":
        row_label = Chunk.metadata_json["row_label"].astext
        return or_(
            and_(Chunk.chunk_type == "TEXT", Chunk.representation == "TEXT"),
            Chunk.representation == "SUMMARY",
            and_(
                Chunk.representation == "SEMANTIC_ROW",
                or_(
                    row_label.is_(None),
                    row_label.op("!~*")(r"^row\s+[0-9]+$"),
                ),
            ),
            and_(
                Chunk.representation == "SEMANTIC",
                Chunk.chunk_type.in_(["SECTION", "FACT"]),
            ),
        )
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
        and_(
            Chunk.representation == "SEMANTIC",
            Chunk.chunk_type.in_(["SECTION", "FACT"]),
        ),
    )


def _embedding_batch_diagnostics(batch: list[Chunk]) -> list[dict[str, object]]:
    """Describe a provider request without writing report content to application logs."""
    return [
        {
            "chunk_id": str(chunk.id),
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type,
            "representation": chunk.representation,
            "content_length": len(chunk.content),
            "content_sha256": hashlib.sha256(chunk.content.encode()).hexdigest()[:16],
        }
        for chunk in batch
    ]


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
        await self._apply_document_context(document_id, session)
        statement = (
            select(Chunk)
            .where(Chunk.document_id == document_id, _indexable_filter())
            .order_by(Chunk.id)
        )
        indexable_chunks = list((await session.execute(statement)).scalars())
        empty_chunks = [chunk for chunk in indexable_chunks if not chunk.content.strip()]
        chunks = [chunk for chunk in indexable_chunks if chunk.content.strip()]
        if empty_chunks:
            logger.warning(
                "Skipping empty embedding chunks: %s",
                json.dumps(
                    {
                        "document_id": str(document_id),
                        "chunks": _embedding_batch_diagnostics(empty_chunks),
                    },
                    sort_keys=True,
                ),
            )
        pending = chunks if force else [
            chunk
            for chunk in chunks
            if chunk.embedding is None
            or chunk.metadata_json.get("embedding_model") != self.provider.embedding_model
        ]
        batches = [pending[index:index + 10] for index in range(0, len(pending), 10)]

        for batch_number, batch in enumerate(batches):
            diagnostics = _embedding_batch_diagnostics(batch)
            log_context = {
                "document_id": str(document_id),
                "batch_number": batch_number,
                "batch_size": len(batch),
                "chunks": diagnostics,
            }
            logger.info("Embedding batch started: %s", json.dumps(log_context, sort_keys=True))
            try:
                vectors = await self.provider.embed(
                    [chunk.content for chunk in batch], text_type="document"
                )
            except Exception:
                logger.exception(
                    "Embedding batch failed: %s", json.dumps(log_context, sort_keys=True)
                )
                raise
            for chunk, vector in zip(batch, vectors, strict=True):
                chunk.embedding = vector
                chunk.metadata_json = {
                    **chunk.metadata_json,
                    "embedding_model": self.provider.embedding_model,
                    "embedding_dimensions": self.provider.embedding_dimensions,
                }
            await session.commit()
            logger.info("Embedding batch completed: %s", json.dumps(log_context, sort_keys=True))
        return {
            "document_id": document_id,
            "model": self.provider.embedding_model,
            "dimensions": self.provider.embedding_dimensions,
            "indexed_chunks": len(pending),
            "skipped_chunks": len(chunks) - len(pending),
            "total_indexable_chunks": len(chunks),
        }

    async def _apply_document_context(
        self, document_id: UUID, session: AsyncSession
    ) -> None:
        document = await session.get(Document, document_id)
        if document is None:
            raise ValueError("Document not found.")
        chunks = list(
            (
                await session.execute(
                    select(Chunk).where(
                        Chunk.document_id == document_id,
                        or_(
                            Chunk.representation.in_(["SEMANTIC_ROW", "SUMMARY"]),
                            and_(
                                Chunk.representation == "SEMANTIC",
                                Chunk.chunk_type.in_(["SECTION", "FACT"]),
                            ),
                        ),
                    )
                )
            ).scalars()
        )
        context_metadata = document_context_metadata(document)
        for chunk in chunks:
            enriched = enrich_semantic_content(
                chunk.semantic_content or chunk.content, document
            )
            if chunk.content != enriched:
                chunk.content = enriched
                chunk.semantic_content = enriched
                chunk.embedding = None
            chunk.metadata_json = {**chunk.metadata_json, **context_metadata}
        await session.flush()

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
        documents: list[Document],
        query: str,
        mode: RetrievalMode,
        session: AsyncSession,
        *,
        knowledge_base_id: UUID | None = None,
    ) -> dict:
        query = unicodedata.normalize("NFC", query.strip())
        if not query:
            raise ValueError("Query cannot be empty.")
        lexical_limit = self.settings.retrieval_lexical_top_k
        vector_limit = self.settings.retrieval_vector_top_k
        available_years = {
            document.fiscal_year for document in documents if document.fiscal_year is not None
        }
        annual_years = {
            document.fiscal_year
            for document in documents
            if document.fiscal_year is not None
            and "annual" in (document.document_type or "").lower()
        }
        query_years = extract_query_years(
            query, available_years, annual_years=annual_years
        )
        group_vonb_time_series = is_group_vonb_time_series(query, query_years)
        explanation_requested = asks_for_explanation(query)
        active_documents = (
            [document for document in documents if document.fiscal_year in query_years]
            if query_years
            else documents
        )
        document_ids = [document.id for document in active_documents]
        document_by_id = {document.id: document for document in active_documents}

        focused_queries = [focused_query(query, year) for year in query_years]
        query_vectors = await self.provider.embed(
            [query, *focused_queries], text_type="query"
        )
        query_vector = query_vectors[0]

        tsquery = func.websearch_to_tsquery("simple", query)
        lexical_score_expr = func.ts_rank_cd(Chunk.search_vector, tsquery, 32).label(
            "lexical_score"
        )
        strict_lexical_rows = (
            await session.execute(
                select(Chunk, lexical_score_expr)
                .where(
                    Chunk.document_id.in_(document_ids),
                    _candidate_filter(mode),
                    Chunk.search_vector.op("@@")(tsquery),
                )
                .order_by(lexical_score_expr.desc(), Chunk.id)
                .limit(lexical_limit)
            )
        ).all()

        terms = lexical_query_terms(query)
        broad_lexical_rows = []
        if terms:
            broad_tsquery = func.to_tsquery("simple", " | ".join(terms))
            broad_score_expr = func.ts_rank_cd(
                Chunk.search_vector, broad_tsquery, 32
            ).label("broad_lexical_score")
            broad_lexical_rows = (
                await session.execute(
                    select(Chunk, broad_score_expr)
                    .where(
                        Chunk.document_id.in_(document_ids),
                        _candidate_filter(mode),
                        Chunk.representation != "TEXT",
                        Chunk.search_vector.op("@@")(broad_tsquery),
                    )
                    .order_by(broad_score_expr.desc(), Chunk.id)
                    .limit(lexical_limit * 2)
                )
            ).all()

        async def vector_search(
            scoped_document_ids: list[UUID], vector: list[float], limit: int
        ) -> list[tuple[Chunk, float]]:
            distance_expr = Chunk.embedding.cosine_distance(vector).label("distance")
            return list(
                (
                    await session.execute(
                        select(Chunk, distance_expr)
                        .where(
                            Chunk.document_id.in_(scoped_document_ids),
                            _candidate_filter(mode),
                            Chunk.embedding.is_not(None),
                        )
                        .order_by(distance_expr, Chunk.id)
                        .limit(limit)
                    )
                ).all()
            )

        vector_ranked_rows = [
            await vector_search(document_ids, query_vector, vector_limit)
        ]
        for year, year_vector in zip(query_years, query_vectors[1:], strict=True):
            year_document_ids = [
                document.id
                for document in active_documents
                if document.fiscal_year == year
            ]
            vector_ranked_rows.append(
                await vector_search(year_document_ids, year_vector, min(10, vector_limit))
            )

        group_vonb_chunks: list[Chunk] = []
        driver_chunks: list[Chunk] = []
        if group_vonb_time_series:
            group_vonb_chunks = [
                chunk
                for chunk in (
                    await session.execute(
                        select(Chunk).where(
                            Chunk.document_id.in_(document_ids),
                            _candidate_filter(mode),
                            Chunk.content.ilike("%VONB%"),
                        )
                    )
                ).scalars()
                if is_group_vonb_chunk(chunk)
            ]
            if explanation_requested:
                driver_vector = (
                    await self.provider.embed(
                        [
                            "AIA Group VONB growth drivers, business reasons, agency productivity, "
                            "distribution, product mix, customer demand, and market performance."
                        ],
                        text_type="query",
                    )
                )[0]
                for year in query_years:
                    year_document_ids = [
                        document.id
                        for document in active_documents
                        if document.fiscal_year == year
                    ]
                    driver_chunks.extend(
                        chunk
                        for chunk, _ in await vector_search(year_document_ids, driver_vector, 3)
                    )

        lexical_chunk_by_id = {
            chunk.id: chunk
            for rows in (strict_lexical_rows, broad_lexical_rows)
            for chunk, _ in rows
        }
        lexical_scores: dict[UUID, float] = {}
        for rows in (strict_lexical_rows, broad_lexical_rows):
            for chunk, score in rows:
                lexical_scores[chunk.id] = max(
                    lexical_scores.get(chunk.id, float("-inf")), float(score)
                )
        lexical_ids, _ = reciprocal_rank_fusion(
            [
                [chunk.id for chunk, _ in strict_lexical_rows],
                [chunk.id for chunk, _ in broad_lexical_rows],
            ],
            k=self.settings.retrieval_rrf_k,
            limit=lexical_limit,
        )
        lexical_chunks = [lexical_chunk_by_id[chunk_id] for chunk_id in lexical_ids]
        vector_chunk_by_id = {
            chunk.id: chunk for rows in vector_ranked_rows for chunk, _ in rows
        }
        vector_scores: dict[UUID, float] = {}
        for rows in vector_ranked_rows:
            for chunk, distance in rows:
                vector_scores[chunk.id] = max(
                    vector_scores.get(chunk.id, float("-inf")),
                    1.0 - float(distance),
                )
        vector_ids, _ = reciprocal_rank_fusion(
            [[chunk.id for chunk, _ in rows] for rows in vector_ranked_rows],
            k=self.settings.retrieval_rrf_k,
            limit=vector_limit,
        )
        vector_chunks = [vector_chunk_by_id[chunk_id] for chunk_id in vector_ids]

        seed_identities = {
            identity
            for chunk in [
                *(chunk for chunk, _ in broad_lexical_rows[:20]),
                *vector_chunks[:10],
                *group_vonb_chunks,
            ]
            if (identity := semantic_identity_values(chunk))
        }
        structured_chunks: list[Chunk] = []
        if seed_identities:
            identity_ranks: dict[tuple[str, str], int] = {}
            ranked_seeds = [
                *(chunk for chunk, _ in broad_lexical_rows),
                *vector_chunks,
            ]
            for rank, chunk in enumerate(ranked_seeds, start=1):
                identity = semantic_identity_values(chunk)
                if identity and identity not in identity_ranks:
                    identity_ranks[identity] = rank
            identity_filters = [
                and_(
                    func.lower(Chunk.metadata_json["table_title"].astext) == title,
                    func.lower(Chunk.metadata_json["row_label"].astext) == row_label,
                )
                for title, row_label in seed_identities
            ]
            structured_chunks = list(
                (
                    await session.execute(
                        select(Chunk)
                        .where(
                            Chunk.document_id.in_(document_ids),
                            _candidate_filter(mode),
                            or_(*identity_filters),
                        )
                        .order_by(Chunk.id)
                        .limit(lexical_limit * 2)
                    )
                ).scalars()
            )
            structured_chunks.sort(
                key=lambda chunk: (
                    identity_ranks.get(semantic_identity_values(chunk), 10_000),
                    document_by_id[chunk.document_id].fiscal_year or 0,
                    str(chunk.id),
                )
            )

        chunk_by_id = {
            chunk.id: chunk
            for chunk in [
                *lexical_chunks,
                *vector_chunks,
                *structured_chunks,
                *group_vonb_chunks,
                *driver_chunks,
            ]
        }
        rrf_ids, rrf_scores = reciprocal_rank_fusion(
            [
                [chunk.id for chunk in lexical_chunks],
                [chunk.id for chunk in vector_chunks],
                [chunk.id for chunk in structured_chunks],
            ],
            k=self.settings.retrieval_rrf_k,
            limit=self.settings.retrieval_rrf_top_k,
        )
        rrf_chunks = [
            chunk_by_id[chunk_id]
            for chunk_id in rrf_ids
        ]
        for chunk in [*group_vonb_chunks, *driver_chunks]:
            if chunk.id not in rrf_ids:
                rrf_ids.append(chunk.id)
                rrf_chunks.append(chunk)

        rerank_pairs = await self.provider.rerank(query, [chunk.content for chunk in rrf_chunks])
        reranked_all = [rrf_chunks[index] for index, _ in rerank_pairs]
        rerank_scores = {
            rrf_chunks[index].id: score for index, score in rerank_pairs
        }
        final_chunks = []
        per_document_counts: dict[UUID, int] = defaultdict(int)
        final_limit = (
            max(self.settings.retrieval_final_top_k, len(query_years) * 2)
            if group_vonb_time_series
            else self.settings.retrieval_final_top_k
        )
        per_document_limit = (
            3 if len(active_documents) > 1 else final_limit
        )
        selected_ids: set[UUID] = set()
        rerank_positions = {
            chunk.id: rank for rank, chunk in enumerate(reranked_all, start=1)
        }
        rrf_positions = {chunk.id: rank for rank, chunk in enumerate(rrf_chunks, start=1)}
        identity_year_chunks: dict[
            tuple[str, str], dict[int, list[Chunk]]
        ] = defaultdict(lambda: defaultdict(list))
        for chunk in rrf_chunks:
            identity = semantic_identity_values(chunk)
            year = document_by_id[chunk.document_id].fiscal_year
            if identity and year:
                identity_year_chunks[identity][year].append(chunk)
        if group_vonb_time_series:
            for year in query_years:
                candidates = [
                    chunk
                    for chunk in group_vonb_chunks
                    if document_by_id[chunk.document_id].fiscal_year == year
                ]
                if not candidates:
                    continue
                series_chunk = min(
                    candidates,
                    key=lambda chunk: (
                        0 if "total group vonb" in chunk.content.lower() else 1,
                        0
                        if (
                            chunk.representation == "SEMANTIC_ROW"
                            and re.search(rf"(?<!\d){year}(?!\d)\s*:", chunk.content)
                        )
                        else 1,
                        rerank_positions.get(chunk.id, 10_000),
                    ),
                )
                final_chunks.append(series_chunk)
                selected_ids.add(series_chunk.id)
                per_document_counts[series_chunk.document_id] += 1
            if explanation_requested:
                driver_ids = {chunk.id for chunk in driver_chunks}
                for year in query_years:
                    driver_chunk = next(
                        (
                            chunk
                            for chunk in reranked_all
                            if chunk.id in driver_ids
                            and document_by_id[chunk.document_id].fiscal_year == year
                            and chunk.id not in selected_ids
                            and per_document_counts[chunk.document_id] < per_document_limit
                        ),
                        None,
                    )
                    if driver_chunk is not None:
                        final_chunks.append(driver_chunk)
                        selected_ids.add(driver_chunk.id)
                        per_document_counts[driver_chunk.document_id] += 1
        complete_identities = [
            (identity, by_year)
            for identity, by_year in identity_year_chunks.items()
            if query_years and all(year in by_year for year in query_years)
        ]
        if complete_identities:
            _, paired_by_year = min(
                complete_identities,
                key=lambda item: sum(
                    min(rrf_positions[chunk.id] for chunk in item[1][year])
                    for year in query_years
                ),
            )
            for year in query_years:
                if len(final_chunks) >= final_limit:
                    break
                paired_chunk = min(
                    paired_by_year[year],
                    key=lambda chunk: rerank_positions.get(chunk.id, 10_000),
                )
                final_chunks.append(paired_chunk)
                selected_ids.add(paired_chunk.id)
                per_document_counts[paired_chunk.document_id] += 1
        for year in query_years:
            if len(final_chunks) >= final_limit:
                break
            if any(
                document_by_id[chunk.document_id].fiscal_year == year
                for chunk in final_chunks
            ):
                continue
            year_chunk = next(
                (
                    chunk
                    for chunk in reranked_all
                    if document_by_id[chunk.document_id].fiscal_year == year
                    and per_document_counts[chunk.document_id] < per_document_limit
                ),
                None,
            )
            if year_chunk:
                final_chunks.append(year_chunk)
                selected_ids.add(year_chunk.id)
                per_document_counts[year_chunk.document_id] += 1
        for chunk in reranked_all:
            if len(final_chunks) >= final_limit:
                break
            if chunk.id in selected_ids:
                continue
            if per_document_counts[chunk.document_id] >= per_document_limit:
                continue
            final_chunks.append(chunk)
            per_document_counts[chunk.document_id] += 1
            if len(final_chunks) >= final_limit:
                break
        final_ranks = {chunk.id: rank for rank, chunk in enumerate(final_chunks, start=1)}

        parent_keys = {
            (chunk.document_id, chunk.metadata_json.get("parent_key"))
            for chunk in chunk_by_id.values()
            if chunk.metadata_json.get("parent_key")
        }
        parent_content: dict[tuple[UUID, str], str] = {}
        if parent_keys:
            keys = {key for _, key in parent_keys}
            parents = (
                await session.execute(
                    select(Chunk).where(
                        Chunk.document_id.in_(document_ids),
                        Chunk.comparison_key.in_(keys),
                    )
                )
            ).scalars()
            parent_content = {
                (parent.document_id, parent.comparison_key): parent.content
                for parent in parents
                if parent.comparison_key
            }

        def hits(chunks: list[Chunk]) -> list[dict]:
            return [
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "rank": rank,
                    "final_rank": final_ranks.get(chunk.id),
                    "chunk_type": chunk.chunk_type,
                    "representation": chunk.representation,
                    "comparison_key": chunk.comparison_key,
                    "parent_key": chunk.metadata_json.get("parent_key"),
                    "file": document_by_id[chunk.document_id].filename,
                    "fiscal_year": document_by_id[chunk.document_id].fiscal_year,
                    "document_type": document_by_id[chunk.document_id].document_type,
                    "page": chunk.page_number,
                    "table_title": chunk.metadata_json.get("table_title"),
                    "row_label": chunk.metadata_json.get("row_label"),
                    "raw_content": chunk.raw_content,
                    "semantic_content": chunk.semantic_content,
                    "parent_content": parent_content.get(
                        (chunk.document_id, chunk.metadata_json.get("parent_key"))
                    ),
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
            "document_ids": document_ids,
            "knowledge_base_id": knowledge_base_id,
            "query_years": query_years,
            "embedding_model": self.provider.embedding_model,
            "rerank_model": self.provider.rerank_model,
            "vector_results": hits(vector_chunks),
            "lexical_results": hits(lexical_chunks),
            "structured_results": hits(structured_chunks),
            "rrf_results": hits(rrf_chunks),
            "reranked_results": hits(final_chunks),
        }

    async def evaluate(self, document: Document, session: AsyncSession) -> dict:
        results = []
        for mode in ("BASELINE", "SEMANTIC"):
            for golden in GOLDEN_QUERIES:
                debug = await self.search([document], golden["query"], mode, session)
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
