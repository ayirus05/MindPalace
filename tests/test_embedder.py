"""Tests for the embedding manager."""

from __future__ import annotations

import pytest

from palace.embeddings.manager import EmbeddingError, FakeEmbedder
from palace.models.config import EmbeddingConfig


class TestFakeEmbedder:
    def test_dimensionality(self) -> None:
        e = FakeEmbedder(dim=128)
        assert e.dimensionality == 128

    def test_embed_one_returns_vector(self) -> None:
        e = FakeEmbedder(dim=32)
        v = e.embed_one("hello world")
        assert len(v) == 32

    def test_embed_batch_preserves_order(self) -> None:
        e = FakeEmbedder(dim=16)
        texts = ["alpha", "beta", "gamma"]
        vecs = e.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 16 for v in vecs)

    def test_embed_empty_batch(self) -> None:
        e = FakeEmbedder()
        assert e.embed_batch([]) == []

    def test_same_text_same_vector(self) -> None:
        e = FakeEmbedder(dim=32)
        assert e.embed_one("test") == e.embed_one("test")

    def test_different_text_different_vector(self) -> None:
        e = FakeEmbedder(dim=32)
        assert e.embed_one("apple") != e.embed_one("orange")

    def test_health_check(self) -> None:
        assert FakeEmbedder().health_check() is True

    def test_model_name(self) -> None:
        assert FakeEmbedder().model_name == "fake"


class TestOllamaEmbedderUnit:
    """Unit tests for OllamaEmbedder using a mock HTTP client."""

    def test_model_name(self) -> None:
        from palace.embeddings.manager import OllamaEmbedder

        cfg = EmbeddingConfig(model="nomic-embed-text")
        emb = OllamaEmbedder(cfg, client=_FakeClient())
        assert emb.model_name == "nomic-embed-text"

    def test_health_check_ok(self) -> None:
        import httpx
        from palace.embeddings.manager import OllamaEmbedder

        client = _FakeClient(
            responses={
                "/api/tags": httpx.Response(200, json={"models": []}),
            }
        )
        emb = OllamaEmbedder(EmbeddingConfig(), client=client)
        assert emb.health_check() is True

    def test_health_check_connection_error(self) -> None:
        from palace.embeddings.manager import OllamaEmbedder

        emb = OllamaEmbedder(EmbeddingConfig(), client=_FailingClient())
        assert emb.health_check() is False

    def test_embed_batch_parses_response(self) -> None:
        import httpx
        from palace.embeddings.manager import OllamaEmbedder

        client = _FakeClient(
            responses={
                "/api/embed": httpx.Response(
                    200,
                    json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
                ),
            }
        )
        emb = OllamaEmbedder(EmbeddingConfig(batch_size=8), client=client)
        vecs = emb.embed_batch(["text one", "text two"])
        assert len(vecs) == 2
        assert vecs[0] == [0.1, 0.2]
        assert vecs[1] == [0.3, 0.4]

    def test_embed_batch_chunked(self) -> None:
        import httpx
        from palace.embeddings.manager import OllamaEmbedder

        # batch_size=1 forces one request per text.
        client = _FakeClient(
            responses={
                "/api/embed": httpx.Response(
                    200, json={"embeddings": [[0.1, 0.2]]},
                ),
            }
        )
        emb = OllamaEmbedder(EmbeddingConfig(batch_size=1, max_retries=1), client=client)
        vecs = emb.embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        assert client.call_count == 3

    def test_embed_batch_mismatched_count(self) -> None:
        import httpx
        from palace.embeddings.manager import OllamaEmbedder

        client = _FakeClient(
            responses={
                "/api/embed": httpx.Response(200, json={"embeddings": [[0.1]]}),
            }
        )
        emb = OllamaEmbedder(EmbeddingConfig(max_retries=1), client=client)
        with pytest.raises(EmbeddingError):
            emb.embed_batch(["a", "b"])  # Ollama returns 1 but we sent 2

    def test_embed_batch_retries_then_succeeds(self) -> None:
        import httpx
        from palace.embeddings.manager import OllamaEmbedder

        client = _FakeClient(
            responses={
                "/api/embed": [
                    httpx.ConnectError("transient"),
                    httpx.Response(200, json={"embeddings": [[0.1, 0.2]]}),
                ],
            }
        )
        emb = OllamaEmbedder(
            EmbeddingConfig(max_retries=3, retry_initial_wait_seconds=0.01, retry_max_wait_seconds=0.05),
            client=client,
        )
        vecs = emb.embed_batch(["text"])
        assert vecs == [[0.1, 0.2]]

    def test_embed_batch_retries_exhausted_raises(self) -> None:
        import httpx
        from palace.embeddings.manager import OllamaEmbedder

        client = _FailingClient()
        emb = OllamaEmbedder(
            EmbeddingConfig(max_retries=2, retry_initial_wait_seconds=0.01, retry_max_wait_seconds=0.02),
            client=client,
        )
        with pytest.raises(EmbeddingError):
            emb.embed_batch(["text"])


# ---- mock HTTP clients ----------------------------------------------------


import httpx


class _FakeClient(httpx.Client):
    """A mock httpx.Client that returns canned responses."""

    def __init__(self, responses: dict | None = None) -> None:
        # Don't call super().__init__ — we don't want real HTTP.
        self._responses = responses or {}
        self.call_count = 0
        self.calls: list[str] = []

    def _normalize_response(self, resp, method: str, url: str):
        if isinstance(resp, httpx.Response):
            if getattr(resp, "_request", None) is None:
                return httpx.Response(
                    status_code=resp.status_code,
                    content=resp.content,
                    headers=resp.headers,
                    request=httpx.Request(method, url),
                )
        return resp

    def get(self, url, **kwargs):
        self.calls.append(url)
        self.call_count += 1
        path = url.split("localhost:11434")[-1] if "localhost" in url else url
        resp = self._responses.get(path)
        if isinstance(resp, list):
            resp = resp.pop(0)
        if isinstance(resp, Exception):
            raise resp
        resp = self._normalize_response(resp, "GET", url)
        return resp or httpx.Response(404, request=httpx.Request("GET", url))

    def post(self, url, json=None, **kwargs):
        self.calls.append(url)
        self.call_count += 1
        from urllib.parse import urlparse

        path = urlparse(url).path
        resp = self._responses.get(path)
        if isinstance(resp, list):
            resp = resp.pop(0)
        if isinstance(resp, Exception):
            raise resp
        resp = self._normalize_response(resp, "POST", url)
        return resp or httpx.Response(404, request=httpx.Request("POST", url))


class _FailingClient(httpx.Client):
    def __init__(self) -> None:
        pass

    def get(self, *args, **kwargs):
        raise httpx.ConnectError("unreachable")

    def post(self, *args, **kwargs):
        raise httpx.ConnectError("unreachable")
