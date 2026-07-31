"""Small hashing and text-estimation utilities.

Centralized so the indexer and repository never duplicate this logic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of a string (UTF-8 encoded)."""
    return sha256_bytes(text.encode("utf-8"))


def hash_file(path: Path) -> str:
    """SHA-256 of a file's contents, read in streaming fashion."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


class TextEstimator(BaseModel):
    """Lightweight word/token estimator shared by the chunker and indexer.

    Token counts are approximations based on words.  Avoids pulling in a
    tokenizer dependency for what is a sizing heuristic, not a hard limit.
    """

    tokens_per_word: float = 1.3

    def word_count(self, text: str) -> int:
        return len(text.split())

    def token_estimate(self, text: str) -> int:
        return int(round(self.word_count(text) * self.tokens_per_word))

    def split_to_token_budget(self, text: str, target_tokens: int) -> list[str]:
        """Split text into word-list segments whose estimated token count
        is at or below ``target_tokens``.  Used as a fallback splitter.
        """
        words = text.split()
        if not words:
            return []
        out: list[str] = []
        cur: list[str] = []
        cur_tokens = 0.0
        for w in words:
            w_tokens = self.tokens_per_word
            if cur_tokens + w_tokens > target_tokens and cur:
                out.append(" ".join(cur))
                cur, cur_tokens = [], 0.0
            cur.append(w)
            cur_tokens += w_tokens
        if cur:
            out.append(" ".join(cur))
        return out
