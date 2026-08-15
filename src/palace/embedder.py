"""Ollama-backed text embedding."""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from palace.models.config import EmbeddingConfig


logger = logging.getLogger("palace.embedder")


class EmbeddingError(RuntimeError):
    """Raised when embedding generation permanently fails."""


class OllamaEmbedder:
    """Production embedder backed by a local Ollama server.

    Uses httpx for HTTP and tenacity for retries with exponential backoff.
    All HTTP code is confined to this class — nothing else in the package
    speaks to Ollama directly.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)
        self._endpoint = f"{config.host.rstrip('/')}/api/embed"
        self._dimensionality: int | None = None

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def dimensionality(self) -> int:
        if self._dimensionality is None:
            probe = self.embed_one("dimensionality probe")
            self._dimensionality = len(probe)
        return self._dimensionality

    def health_check(self) -> bool:
        """Probe Ollama connectivity via the /api/tags endpoint."""
        try:
            resp = self._client.get(
                f"{self._config.host.rstrip('/')}/api/tags",
                timeout=min(self._config.timeout_seconds, 10.0),
            )
            return resp.status_code == 200
        except Exception:
            return False

    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in sub-batches, retrying transient failures.

        Raises:
            EmbeddingError: if any sub-batch fails after all retries.
        """
        if not texts:
            return []
        results: list[list[float]] = []
        batch_size = max(1, self._config.batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = self._embed_batch_with_retry(batch)
            results.extend(vectors)
        return results

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        retryer = retry(
            stop=stop_after_attempt(self._config.max_retries),
            wait=wait_exponential(
                multiplier=self._config.retry_initial_wait_seconds,
                max=self._config.retry_max_wait_seconds,
            ),
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )(self._post_embed)
        try:
            return retryer(batch)
        except (httpx.HTTPError, RetryError) as exc:
            raise EmbeddingError(
                f"Ollama embedding failed after {self._config.max_retries} retries: {exc}"
            ) from exc

    def _post_embed(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self._config.model, "input": batch}
        resp = self._client.post(self._endpoint, json=payload)
        if resp.status_code >= 400:
            try:
                resp.raise_for_status()
            except Exception as exc:
                raise EmbeddingError(
                    f"Ollama request failed with status {resp.status_code}: {exc}"
                ) from exc
        data = resp.json()
        # Ollama's /api/embed returns {"embeddings": [[...], ...]}.
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings or [])} embeddings for {len(batch)} inputs"
            )
        return [list(map(float, vec)) for vec in embeddings]
