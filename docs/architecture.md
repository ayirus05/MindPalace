# Architecture

## Overview

MindPalace Tier 2 is a local-first semantic memory engine. It indexes unstructured text (journals, notes, transcripts) into a vector store and provides ranked semantic search with metadata filtering. The system is designed to scale to years of data while keeping incremental indexing cheap.

## Design principles

**Layered architecture with one-directional dependencies.** Models depend on nothing. Metadata extraction depends on models. The embedder depends on models. The chunker depends on models + metadata. The repository depends on models + LanceDB. The indexer depends on all of the above. The search engine depends on the embedder + repository + ranker. The CLI depends on everything. No layer reaches downward past its own scope.

**Purposeful concrete implementations.** Embedding uses `OllamaEmbedder`, persistence uses `LanceDBRepository`, and chunking uses `MarkdownASTChunker`.

**No raw database code outside the repository.** LanceDB is imported in exactly one file: `repository.py`. If you ever want to swap LanceDB for Qdrant, Pinecone, or pgvector, you replace one file.

**No global state.** Configuration flows through `PalaceConfig` instances passed via dependency injection. Loggers are named per-module, not global singletons. The hash cache is an instance, not a module-level variable.

**Idempotent writes.** `update_chunks(source_file, chunks)` deletes existing chunks for that source, then inserts new ones. Re-indexing the same file produces the same end state. The incremental indexer's hash cache makes this cheap — unchanged files are skipped entirely.

## Data flow

### Indexing pipeline

```
Source files → discover_files() → hash each file
    → if hash unchanged: skip (no embedding work)
    → if hash changed or new:
        → chunk_file() → [Chunk, Chunk, ...]
        → embedder.embed_batch([chunk.content for chunk in chunks])
        → repository.update_chunks(source_file, populated_chunks)
        → hash_cache.upsert(source_file, hash, mtime, chunk_count)
    → if file vanished from disk:
        → repository.delete_chunks(source_file)
        → hash_cache.remove(source_file)
```

### Search pipeline

```
query string
    → embedder.embed_one(query) → query_vector
    → repository.search(query_vector, top_k=prefilter, domain/date/tags filters)
        → LanceDB pre-filters on metadata (SQL WHERE)
        → vector similarity over filtered set
        → [RawHit, RawHit, ...]
    → SearchRanker.rank(hits)
        → semantic score (base)
        → + recency boost (if within N days)
        → × domain weight (per-domain multiplier)
        → filter below minimum_score
        → deduplicate by source_file (keep best per file)
        → top_k limit
        → [SearchResult, SearchResult, ...]
```

## Chunking strategy

`MarkdownASTChunker` handles every document domain. It parses Markdown headings, emits one chunk per heading-delimited section, and carries the heading hierarchy into keyword extraction as a context path. Headerless documents become a single chunk. Each result is a validated `Chunk` populated through `MetadataExtractor`.

## Metadata extraction

The `MetadataExtractor` composes three strategies:

- **Date extraction** — regex chain: ISO → long-form → slash dates. Returns the first match. Slash dates use a heuristic: if the first number > 12, it's day-first (`15/4/2026`); otherwise month-first (`4/15/2026`).
- **Domain inference** — based on the source file's path components. `/journals/` → JOURNAL, `/notes/` → NOTES, `/transcripts/` → TRANSCRIPT, etc. Unknown paths get UNKNOWN.
- **Tag extraction** — two sources: YAML front-matter `tags:` field and inline `#tag` syntax. Duplicates are deduplicated. Pure-numeric tags are ignored.

## Ranking

The ranker combines four signals:

1. **Semantic similarity** — the base score from LanceDB's cosine similarity.
2. **Recency boost** — results within `recency_boost_days` get a bonus that decays linearly to zero at the boundary.
3. **Domain weighting** — per-domain multipliers (e.g., journal: 1.1, notes: 1.0, transcript: 0.9). Domains not listed default to 1.0.
4. **Duplicate suppression** — if enabled, only the best-scoring hit per source file is kept.

Results below `minimum_score` are dropped before ranking. The final list is truncated to `default_top_k` (or the caller's `top_k` override).

## Configuration

`PalaceConfig` is a Pydantic model loaded from `config.yaml`. All paths are resolved relative to `project_root` (the directory containing the config file) unless absolute. The config object is passed by injection to every component that needs it — no component reads `config.yaml` directly.

## Embedding implementation

`OllamaEmbedder` uses httpx for HTTP and tenacity for exponential-backoff retries. It batches texts at `batch_size`, and its health check probes `/api/tags`. All Ollama HTTP code is confined to `embedder.py`. Tests keep their deterministic fake under `tests/`, outside the production package.

## Scaling considerations

- **10,000+ entries:** LanceDB is file-based and uses disk-backed approximation. The prefilter (metadata SQL pushdown) narrows the candidate set before vector comparison, keeping search fast even with large corpora.
- **Minimal RAM:** chunks are embedded in batches (default 32), not all at once. The hash cache skips unchanged files entirely — re-indexing after touching 5 entries out of 10,000 only embeds those 5.
- **Fast startup:** LanceDB opens lazily; the table is created on first write. No daemon process to start.
- **Incremental indexing:** content-hash deltas mean the cost of `palace index` is proportional to what changed, not to the corpus size.
