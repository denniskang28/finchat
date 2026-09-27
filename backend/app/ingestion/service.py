import asyncio
import time
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.ingestion.parser import PdfParser
from app.ingestion.repair_providers import create_table_repair_provider
from app.ingestion.renderers import build_table_debug
from app.ingestion.table_quality import TableRepairPipeline
from app.models import Chunk, Document


class IngestionService:
    def __init__(self, provider_name: str | None = None) -> None:
        self.settings = get_settings()
        self.provider = create_table_repair_provider(self.settings, provider_name)
        if provider_name is not None and not self.provider.available:
            raise ValueError(f"Provider '{provider_name}' is not configured with an API key.")
        self.parser = PdfParser(
            TableRepairPipeline(
                self.provider,
                llm_primary=self.settings.table_parse_mode == "llm_primary",
            ),
            page_workers=self.settings.table_parse_workers,
        )

    async def ingest(self, upload: UploadFile, session: AsyncSession) -> Document:
        filename = Path(upload.filename or "document.pdf").name
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF uploads are supported.")
        payload = await upload.read()
        if not payload.startswith(b"%PDF"):
            raise ValueError("Uploaded file is not a valid PDF.")

        document_id = uuid.uuid4()
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        stored_path = self.settings.upload_dir / f"{document_id}.pdf"
        stored_path.write_bytes(payload)

        document = Document(
            id=document_id,
            filename=filename,
            title=Path(filename).stem,
            file_path=str(stored_path),
            status="PROCESSING",
            metadata_json={},
        )
        session.add(document)
        await session.commit()

        try:
            parse_started_at = time.perf_counter()
            pages, tables, page_contents = await asyncio.to_thread(
                self.parser.parse, stored_path, document_id
            )
            parse_duration_seconds = time.perf_counter() - parse_started_at
            for page_number, page_text in enumerate(pages, start=1):
                session.add(
                    Chunk(
                        document_id=document_id,
                        page_number=page_number,
                        chunk_type="TEXT",
                        representation="TEXT",
                        comparison_key=f"p{page_number}:text",
                        content=page_text,
                        raw_content=page_text,
                        metadata_json={"element_type": "TEXT", "page": page_number},
                    )
                )

            content_section_count = 0
            content_fact_count = 0
            content_warning_count = 0
            low_coverage_pages: list[int] = []
            coverage_total = 0.0
            for page_content in page_contents:
                coverage = page_content.coverage
                coverage_total += coverage.coverage_ratio
                content_warning_count += len(page_content.parse_warnings) + len(coverage.warnings)
                if coverage.coverage_ratio < 0.8 or coverage.warnings:
                    low_coverage_pages.append(page_content.page_number)
                session.add(
                    Chunk(
                        document_id=document_id,
                        page_number=page_content.page_number,
                        chunk_type="PAGE_SUMMARY",
                        representation="SEMANTIC",
                        comparison_key=f"p{page_content.page_number}:summary",
                        content=page_content.summary,
                        raw_content=pages[page_content.page_number - 1],
                        semantic_content=page_content.summary,
                        metadata_json={
                            "element_type": "PAGE_SUMMARY",
                            "title": page_content.title,
                            "section_count": len(page_content.sections),
                            "coverage": coverage.model_dump(mode="json"),
                            "parse_warnings": page_content.parse_warnings,
                        },
                    )
                )
                for section in page_content.sections:
                    content_section_count += 1
                    session.add(
                        Chunk(
                            document_id=document_id,
                            page_number=page_content.page_number,
                            chunk_type="SECTION",
                            representation="SEMANTIC",
                            comparison_key=section.section_id,
                            content=section.content,
                            raw_content=section.raw_content,
                            semantic_content=section.content,
                            metadata_json={
                                "element_type": "SECTION",
                                "heading_path": section.heading_path,
                                "bbox": section.bbox,
                                "source_token_ids": section.source_token_ids,
                                "parent_key": f"p{page_content.page_number}:summary",
                                "fact_count": len(section.facts),
                            },
                        )
                    )
                    for fact in section.facts:
                        content_fact_count += 1
                        session.add(
                            Chunk(
                                document_id=document_id,
                                page_number=page_content.page_number,
                                chunk_type="FACT",
                                representation="SEMANTIC",
                                comparison_key=fact.fact_id,
                                content=fact.content,
                                raw_content=f"{fact.label}{f' | {fact.value}' if fact.value else ''}",
                                semantic_content=fact.content,
                                metadata_json={
                                    "element_type": "FACT",
                                    "label": fact.label,
                                    "value": fact.value,
                                    "bbox": fact.bbox,
                                    "source_token_ids": fact.source_token_ids,
                                    "footnotes": fact.footnotes,
                                    "parent_key": section.section_id,
                                    "page_summary_key": f"p{page_content.page_number}:summary",
                                },
                            )
                        )

            raw_row_count = 0
            semantic_row_count = 0
            warning_count = 0
            repaired_table_count = 0
            review_table_count = 0
            repair_count = 0
            llm_primary_pages: set[int] = set()
            fallback_pages: set[int] = set()
            llm_primary_structure_count = 0
            fallback_structure_count = 0
            for artifact in tables:
                table = artifact.canonical_table
                if table.extraction_method.startswith("llm.page_structure"):
                    llm_primary_structure_count += 1
                    llm_primary_pages.add(table.page_number)
                if any(
                    warning.startswith("LLM-primary fallback:")
                    for warning in table.parse_warnings
                ):
                    fallback_structure_count += 1
                    fallback_pages.add(table.page_number)
                debug = build_table_debug(artifact)
                warning_count += len(table.parse_warnings) + len(artifact.quality.remaining_issues)
                repair_count += len(artifact.repairs)
                repaired_table_count += artifact.quality.status == "REPAIRED"
                review_table_count += artifact.quality.status == "NEEDS_REVIEW"
                debug_json = debug.model_dump(mode="json")
                session.add(
                    Chunk(
                        document_id=document_id,
                        page_number=table.page_number,
                        chunk_type="TABLE_MARKDOWN",
                        representation="MARKDOWN",
                        comparison_key=f"{table.table_id}:table",
                        content=debug.markdown,
                        raw_content=None,
                        semantic_content=None,
                        metadata_json={
                            "element_type": "TABLE_MARKDOWN",
                            "table_id": table.table_id,
                            "table_title": table.title,
                            "table_debug": debug_json,
                        },
                    )
                )
                semantic_by_key = {row.comparison_key: row.content for row in debug.semantic_rows}
                raw_by_key = {row.comparison_key: row.content for row in debug.raw_rows}
                for raw_row in debug.raw_rows:
                    raw_row_count += 1
                    session.add(
                        Chunk(
                            document_id=document_id,
                            page_number=table.page_number,
                            chunk_type="CHART_SERIES" if table.source_kind == "CHART" else "TABLE_ROW",
                            representation="RAW_ROW",
                            comparison_key=raw_row.comparison_key,
                            content=raw_row.content,
                            raw_content=raw_row.content,
                            semantic_content=semantic_by_key.get(raw_row.comparison_key),
                            metadata_json={
                                "table_id": table.table_id,
                                "table_title": table.title,
                                "row_label": raw_row.row_label,
                                "source_kind": table.source_kind,
                            },
                        )
                    )
                for semantic_row in debug.semantic_rows:
                    semantic_row_count += 1
                    session.add(
                        Chunk(
                            document_id=document_id,
                            page_number=table.page_number,
                            chunk_type="CHART_SERIES" if table.source_kind == "CHART" else "TABLE_ROW",
                            representation="SEMANTIC_ROW",
                            comparison_key=semantic_row.comparison_key,
                            content=semantic_row.content,
                            raw_content=raw_by_key.get(semantic_row.comparison_key),
                            semantic_content=semantic_row.content,
                            metadata_json={
                                "table_id": table.table_id,
                                "table_title": table.title,
                                "row_label": semantic_row.row_label,
                                "source_kind": table.source_kind,
                            },
                        )
                    )

            document.page_count = len(pages)
            document.status = "READY"
            document.metadata_json = {
                "text_pages": len(pages),
                "tables": len(tables),
                "raw_rows": raw_row_count,
                "semantic_rows": semantic_row_count,
                "parse_warnings": warning_count + content_warning_count,
                "table_parse_warnings": warning_count,
                "content_parse_warnings": content_warning_count,
                "content_sections": content_section_count,
                "content_facts": content_fact_count,
                "content_coverage": round(coverage_total / len(page_contents), 4) if page_contents else 0.0,
                "low_content_coverage_pages": low_coverage_pages,
                "table_repairs": repair_count,
                "repaired_tables": repaired_table_count,
                "needs_review_tables": review_table_count,
                "table_repair_provider": self.provider.name,
                "table_repair_model": self.provider.model,
                "parse_duration_seconds": round(parse_duration_seconds, 3),
                "table_parse_mode": self.settings.table_parse_mode,
                "table_parse_workers": self.settings.table_parse_workers,
                "llm_primary_attempted_pages": (
                    len(pages)
                    if self.settings.table_parse_mode == "llm_primary"
                    and self.provider.available
                    else 0
                ),
                "llm_primary_structures": llm_primary_structure_count,
                "llm_primary_structure_pages": sorted(llm_primary_pages),
                "fallback_structures": fallback_structure_count,
                "fallback_structure_pages": sorted(fallback_pages),
                "llm_repair_available": self.provider.available,
            }
            await session.commit()
            await session.refresh(document)
            return document
        except Exception as exc:
            await session.rollback()
            document.status = "FAILED"
            document.metadata_json = {"error": str(exc)}
            session.add(document)
            await session.commit()
            raise
