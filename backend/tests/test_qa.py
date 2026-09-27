import asyncio
from uuid import uuid4

from app.api.qa import build_qa_evidence
from app.config import Settings
from app.qa.provider import DeepSeekChatProvider
from app.schemas import QAEvidence, RetrievalDebugResponse, RetrievalHit


def _hit(*, content: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        rank=rank,
        final_rank=rank,
        chunk_type="TABLE_ROW",
        representation="SEMANTIC_ROW",
        comparison_key=f"row-{rank}",
        file="report.pdf",
        page=91,
        content=content,
    )


def test_qa_evidence_uses_only_final_reranked_results():
    vector_only = _hit(content="vector candidate", rank=1)
    final = _hit(content="final evidence", rank=1)
    retrieval = RetrievalDebugResponse(
        original_query="question",
        retrieval_mode="PRODUCTION",
        document_ids=[final.document_id],
        embedding_model="embedding-model",
        rerank_model="rerank-model",
        vector_results=[vector_only],
        lexical_results=[],
        rrf_results=[vector_only],
        reranked_results=[final],
    )

    evidence = build_qa_evidence(retrieval)

    assert [item.content for item in evidence] == ["final evidence"]
    assert evidence[0].chunk_id == final.chunk_id
    assert evidence[0].evidence_number == 1


def test_deepseek_retries_invalid_structured_answer(monkeypatch):
    class FakeResponse:
        status_code = 200
        is_error = False
        text = ""

        def __init__(self, content):
            self.content = content

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    class FakeClient:
        def __init__(self):
            self.responses = [
                FakeResponse("not-json"),
                FakeResponse(
                    '{"answer":"USD 6.2 billion [E1]","citation_numbers":[1],'
                    '"insufficient_evidence":false}'
                ),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, *_args, **_kwargs):
            return self.responses.pop(0)

    fake_client = FakeClient()
    monkeypatch.setattr(
        "app.qa.provider.httpx.AsyncClient", lambda **_kwargs: fake_client
    )
    provider = DeepSeekChatProvider(
        Settings(
            deepseek_api_key="test-key",
            deepseek_base_url="https://example.test",
            deepseek_chat_model="deepseek-chat",
        )
    )
    evidence = [
        QAEvidence(
            evidence_number=1,
            chunk_id=uuid4(),
            filename="report.pdf",
            page=91,
            chunk_type="TABLE_ROW",
            content="USD 6.2 billion",
        )
    ]

    proposal = asyncio.run(provider.answer("How much?", evidence))

    assert proposal.answer == "USD 6.2 billion [E1]"
    assert proposal.citation_numbers == [1]
    assert fake_client.responses == []


def test_deepseek_retries_citation_outside_supplied_evidence(monkeypatch):
    class FakeResponse:
        status_code = 200
        is_error = False
        text = ""

        def __init__(self, content):
            self.content = content

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    class FakeClient:
        def __init__(self):
            self.responses = [
                FakeResponse(
                    '{"answer":"Unsupported [E99]","citation_numbers":[99],'
                    '"insufficient_evidence":false}'
                ),
                FakeResponse(
                    '{"answer":"Supported [E1]","citation_numbers":[1],'
                    '"insufficient_evidence":false}'
                ),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, *_args, **_kwargs):
            return self.responses.pop(0)

    fake_client = FakeClient()
    monkeypatch.setattr(
        "app.qa.provider.httpx.AsyncClient", lambda **_kwargs: fake_client
    )
    provider = DeepSeekChatProvider(
        Settings(
            deepseek_api_key="test-key",
            deepseek_base_url="https://example.test",
            deepseek_chat_model="deepseek-chat",
        )
    )
    evidence = [
        QAEvidence(
            evidence_number=1,
            chunk_id=uuid4(),
            filename="report.pdf",
            page=91,
            chunk_type="TABLE_ROW",
            content="Supported",
        )
    ]

    proposal = asyncio.run(provider.answer("What is supported?", evidence))

    assert proposal.answer == "Supported [E1]"
    assert proposal.citation_numbers == [1]
    assert fake_client.responses == []
