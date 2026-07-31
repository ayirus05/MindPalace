"""Tests for hashing utilities."""

from __future__ import annotations

from pathlib import Path

from palace.utils.hashing import TextEstimator, hash_file, sha256_bytes, sha256_text


class TestHashing:
    def test_sha256_bytes_deterministic(self) -> None:
        assert sha256_bytes(b"hello") == sha256_bytes(b"hello")
        assert sha256_bytes(b"hello") != sha256_bytes(b"world")

    def test_sha256_text_deterministic(self) -> None:
        assert sha256_text("hello") == sha256_text("hello")
        assert sha256_text("hello") != sha256_text("world")

    def test_sha256_text_and_bytes_agree(self) -> None:
        assert sha256_text("hello") == sha256_bytes(b"hello")

    def test_hash_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.txt"
        p.write_text("hello world", encoding="utf-8")
        assert hash_file(p) == sha256_text("hello world")

    def test_hash_file_large(self, tmp_path: Path) -> None:
        p = tmp_path / "big.txt"
        content = "x" * 100_000
        p.write_text(content, encoding="utf-8")
        assert hash_file(p) == sha256_text(content)

    def test_hash_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            hash_file(tmp_path / "missing.txt")


class TestTextEstimator:
    def test_word_count(self) -> None:
        est = TextEstimator()
        assert est.word_count("one two three") == 3
        assert est.word_count("") == 0

    def test_token_estimate(self) -> None:
        est = TextEstimator(tokens_per_word=1.3)
        assert est.token_estimate("one two three") == 4  # 3 * 1.3 = 3.9 -> 4

    def test_split_to_token_budget(self) -> None:
        est = TextEstimator(tokens_per_word=1.0)
        text = " ".join(f"word{i}" for i in range(10))
        chunks = est.split_to_token_budget(text, target_tokens=3)
        assert len(chunks) == 4  # 3 + 3 + 3 + 1
        assert "word0" in chunks[0]
        assert "word9" in chunks[-1]

    def test_split_empty_text(self) -> None:
        est = TextEstimator()
        assert est.split_to_token_budget("", target_tokens=100) == []
