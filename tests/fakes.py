"""Test-only fakes shared across the suite."""

from __future__ import annotations

import hashlib


class FakeEmbedder:
    """Deterministic lexical embedder for tests."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimensionality(self) -> int:
        return self._dim

    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dim
            for word in text.lower().split():
                digest = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                vector[digest % self._dim] += 1.0
            norm = sum(value * value for value in vector) ** 0.5
            vectors.append(
                [value / norm if norm else value for value in vector]
                if norm
                else vector
            )
        return vectors

    def health_check(self) -> bool:
        return True
