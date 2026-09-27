from __future__ import annotations

import asyncio

import httpx

from app.config import Settings


class AlibabaRetrievalProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.dashscope_api_key
        self.embedding_model = settings.alibaba_embedding_model
        self.embedding_dimensions = settings.alibaba_embedding_dimensions
        self.embedding_url = settings.alibaba_embedding_url
        self.rerank_model = settings.alibaba_rerank_model
        self.rerank_url = settings.alibaba_rerank_url
        self.timeout = settings.table_repair_timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _post(self, url: str, payload: dict) -> httpx.Response:
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(3):
                try:
                    response = await client.post(url, headers=self._headers(), json=payload)
                except httpx.HTTPError as exc:
                    last_error = exc
                else:
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        return response
                    last_error = RuntimeError(
                        f"Alibaba request failed ({response.status_code}): {response.text[:500]}"
                    )
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Alibaba request failed after 3 attempts: {last_error}")

    async def embed(
        self, texts: list[str], *, text_type: str = "document"
    ) -> list[list[float]]:
        if not self.available:
            raise ValueError("DASHSCOPE_API_KEY is required for embedding.")
        if not texts:
            return []
        parameters = {
            "dimension": self.embedding_dimensions,
            "text_type": text_type,
        }
        if text_type == "query":
            parameters["instruct"] = (
                "Given a query, retrieve the financial-report passage that directly "
                "matches every entity, geography, measure, period, and qualifier."
            )
        payload = {
            "model": self.embedding_model,
            "input": {"texts": texts},
            "parameters": parameters,
        }
        response = await self._post(self.embedding_url, payload)
        if response.is_error:
            raise RuntimeError(
                f"Alibaba embedding request failed ({response.status_code}): {response.text[:500]}"
            )
        data = response.json().get("output", {}).get("embeddings", [])
        ordered = sorted(data, key=lambda item: item["text_index"])
        embeddings = [item["embedding"] for item in ordered]
        if len(embeddings) != len(texts):
            raise RuntimeError("Alibaba embedding response count did not match the request.")
        if any(len(vector) != self.embedding_dimensions for vector in embeddings):
            raise RuntimeError("Alibaba embedding response had an unexpected dimension.")
        return embeddings

    async def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not self.available:
            raise ValueError("DASHSCOPE_API_KEY is required for reranking.")
        if not documents:
            return []
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "instruct": (
                "Given a query, find the financial-report passage that directly "
                "answers every entity, geography, measure, period, and qualifier."
            ),
        }
        response = await self._post(self.rerank_url, payload)
        if response.is_error:
            raise RuntimeError(
                f"Alibaba rerank request failed ({response.status_code}): {response.text[:500]}"
            )
        results = [
            (int(item["index"]), float(item["relevance_score"]))
            for item in response.json().get("results", [])
        ]
        indices = [index for index, _ in results]
        if len(indices) != len(documents):
            raise RuntimeError("Alibaba rerank response omitted candidate documents.")
        if len(set(indices)) != len(indices):
            raise RuntimeError("Alibaba rerank response contained duplicate indices.")
        if any(index < 0 or index >= len(documents) for index in indices):
            raise RuntimeError("Alibaba rerank response contained an out-of-range index.")
        return results
