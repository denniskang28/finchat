import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.documents import get_chunk, list_chunks, update_document_metadata
from app.schemas import DocumentMetadataUpdate


class _ScalarResult:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return self.values

    def scalar_one_or_none(self):
        return self.values[0] if self.values else None


class _Session:
    def __init__(self, document, chunks):
        self.document = document
        self.chunks = chunks
        self.statements = []

    async def get(self, _model, document_id):
        return self.document if document_id == self.document.id else None

    async def execute(self, statement):
        self.statements.append(statement)
        return _ScalarResult(self.chunks)

    async def commit(self):
        pass

    async def refresh(self, _value):
        pass


def _chunk(document_id):
    return SimpleNamespace(
        id=uuid4(),
        document_id=document_id,
        page_number=91,
        chunk_type="TABLE_ROW",
        representation="SEMANTIC_ROW",
        comparison_key="p91_t1:total",
        content="The total corporate bond allocation is 100%.",
        raw_content="Total | 100%",
        semantic_content="The total corporate bond allocation is 100%.",
        metadata_json={"table_id": "p91_t1", "row_label": "Total"},
        embedding=None,
        created_at=datetime.now(UTC),
    )


def test_list_chunks_returns_lightweight_summaries():
    document = SimpleNamespace(id=uuid4())
    chunk = _chunk(document.id)
    session = _Session(document, [chunk])

    result = asyncio.run(list_chunks(document.id, session))

    assert result == [
        {
            "id": chunk.id,
            "page_number": 91,
            "chunk_type": "TABLE_ROW",
            "representation": "SEMANTIC_ROW",
            "comparison_key": "p91_t1:total",
            "content_preview": "The total corporate bond allocation is 100%.",
            "created_at": chunk.created_at,
        }
    ]
    assert "raw_content" not in result[0]
    assert "metadata" not in result[0]


def test_get_chunk_returns_full_debug_payload_and_is_document_scoped():
    document = SimpleNamespace(id=uuid4())
    chunk = _chunk(document.id)
    session = _Session(document, [chunk])

    result = asyncio.run(get_chunk(document.id, chunk.id, session))

    assert result["raw_content"] == "Total | 100%"
    assert result["semantic_content"] == chunk.content
    assert result["metadata"]["table_id"] == "p91_t1"
    assert result["embedding_dimensions"] is None
    query = str(session.statements[-1])
    assert "chunks.id" in query
    assert "chunks.document_id" in query


def test_get_chunk_returns_404_when_chunk_is_not_in_document():
    document = SimpleNamespace(id=uuid4())
    session = _Session(document, [])

    with pytest.raises(HTTPException) as error:
        asyncio.run(get_chunk(document.id, uuid4(), session))

    assert error.value.status_code == 404
    assert error.value.detail == "Chunk not found"


def test_metadata_patch_preserves_fields_that_were_not_sent():
    document = SimpleNamespace(
        id=uuid4(),
        knowledge_base_id=uuid4(),
        filename="report.pdf",
        title="report",
        status="READY",
        page_count=10,
        company="Old Company",
        fiscal_year=2025,
        document_type="annual_results",
        language="en",
        metadata_json={},
        created_at=datetime.now(UTC),
    )
    session = _Session(document, [])

    result = asyncio.run(
        update_document_metadata(
            document.id,
            DocumentMetadataUpdate(company="New Company"),
            session,
        )
    )

    assert result["company"] == "New Company"
    assert result["fiscal_year"] == 2025
    assert result["document_type"] == "annual_results"
    assert result["language"] == "en"
