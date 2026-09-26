import asyncio
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.ingestion.parser import AiaPdfParser
from app.ingestion.renderers import build_table_debug
from app.models import Chunk, Document


class IngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.parser = AiaPdfParser()

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
            pages, tables = await asyncio.to_thread(self.parser.parse, stored_path, document_id)
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

            raw_row_count = 0
            semantic_row_count = 0
            warning_count = 0
            for table in tables:
                debug = build_table_debug(table)
                warning_count += len(table.parse_warnings)
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
                "parse_warnings": warning_count,
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

