from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


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

    async def _post(
        self,
        url: str,
        payload: dict,
        *,
        max_attempts: int = 3,
        retryable_response: Callable[[httpx.Response], bool] | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        last_response: httpx.Response | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(max_attempts):
                try:
                    response = await client.post(url, headers=self._headers(), json=payload)
                except httpx.HTTPError as exc:
                    last_error = exc
                else:
                    should_retry = response.status_code in {429, 500, 502, 503, 504}
                    if retryable_response is not None:
                        should_retry = should_retry or retryable_response(response)
                    if not should_retry:
                        return response
                    last_response = response
                    last_error = RuntimeError(
                        f"Alibaba request failed ({response.status_code}): {response.text[:500]}"
                    )
                if attempt < max_attempts - 1:
                    delay_seconds = min(8.0, 0.75 * (2**attempt))
                    logger.warning(
                        "Retrying Alibaba request after attempt %s/%s in %.2fs.",
                        attempt + 1,
                        max_attempts,
                        delay_seconds,
                    )
                    await asyncio.sleep(delay_seconds)
        if last_response is not None:
            return last_response
        raise RuntimeError(f"Alibaba request failed after {max_attempts} attempts: {last_error}")

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
        response = await self._post(
            self.embedding_url,
            payload,
            max_attempts=6,
            retryable_response=lambda candidate: (
                candidate.status_code == 400
                and "already running" in candidate.text.lower()
            ),
        )
        if response.is_error:
            raise RuntimeError(
                f"Alibaba embedding request failed ({response.status_code}): {response.text[:500]}"
            )
        data = response.json().get("output", {}).get("embeddings", [])
        try:
            ordered = sorted(
                data,
                key=lambda item: item.get("text_index", item.get("index")),
            )
        except TypeError as exc:
            raise RuntimeError("Alibaba embedding response omitted item indices.") from exc
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
