"""Chunk local Git history into commit-level technical documents."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from palace.metadata.keywords import extract_keywords
from palace.models.chunk import Chunk, ChunkMetadata, DocumentDomain
from palace.utils.hashing import TextEstimator, sha256_text


_RECORD_SEPARATOR = "\x1e"
_FIELD_SEPARATOR = "\x1f"
_LOG_FORMAT = f"{_RECORD_SEPARATOR}%H{_FIELD_SEPARATOR}%aI{_FIELD_SEPARATOR}%B"


class GitLogChunker:
    """Create one chunk for each commit in a local Git repository."""

    def chunk_repo(self, repo_path: Path) -> list[Chunk]:
        """Read ``repo_path`` history and return commit-message/stat chunks."""
        result = subprocess.run(
            ["git", "log", f"--format={_LOG_FORMAT}", "--stat"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        estimator = TextEstimator()
        chunks: list[Chunk] = []
        for record in result.stdout.split(_RECORD_SEPARATOR):
            record = record.strip()
            if not record:
                continue

            fields = record.split(_FIELD_SEPARATOR, maxsplit=2)
            if len(fields) != 3:
                continue

            commit_hash, author_date_text, content = fields
            content = content.strip()
            author_date = datetime.fromisoformat(author_date_text.strip())
            metadata = ChunkMetadata(
                source_file=str(repo_path),
                source_file_hash=commit_hash.strip(),
                domain=DocumentDomain.TECHNICAL,
                date=author_date.date(),
                tags=[],
                keywords=extract_keywords(content),
                content_hash=sha256_text(content),
                word_count=estimator.word_count(content),
                token_estimate=estimator.token_estimate(content),
                created_at=author_date,
                updated_at=author_date,
            )
            chunks.append(Chunk(metadata=metadata, content=content))
        return chunks


__all__ = ["GitLogChunker"]
