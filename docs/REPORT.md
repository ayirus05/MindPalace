# MindPalace Tier 2 — Implementation Report

## Architecture summary

MindPalace Tier 2 is a local-first semantic memory engine that indexes unstructured text (journals, notes, transcripts) into a LanceDB vector store using Ollama's `nomic-embed-text` embedding model. It provides ranked semantic search with metadata pre-filtering, recency weighting, domain weighting, and duplicate suppression.

The system is layered with one-directional dependencies:

```
CLI → Search Engine / Indexer → Repository + Embedder → LanceDB / Ollama
                                     ↑
                              Chunker + Metadata Extractor
                                     ↑
                                   Models
```

External service boundaries use protocols (`Embedder` and `ChunkRepository`), while all chunking uses the concrete `MarkdownASTChunker`. LanceDB code is confined to a single file (`repository.py`); Ollama HTTP code is confined to a single class (`OllamaEmbedder`).

The system integrates with the existing Tier 1 (structured markdown palace) without modifying it. Tier 1 and Tier 2 share the same MindPalace directory but are independent: Tier 1 is markdown files read directly by Claude; Tier 2 is a vector index queried via the CLI or a future MCP server. The planned skills layer will fuse the two by reading Tier 1 deterministically and calling Tier 2's `semantic_search()` for unstructured recall.

## Package structure

```
src/palace/
├── models/
│   ├── chunk.py         — Chunk, ChunkMetadata, SearchResult, RawHit, IndexStats, DocumentDomain
│   └── config.py        — PalaceConfig + EmbeddingConfig, ChunkerConfig, DatabaseConfig, IndexerConfig, SearchConfig, LoggingConfig
├── embeddings/
│   └── manager.py       — Embedder protocol, OllamaEmbedder (httpx + tenacity), FakeEmbedder
├── metadata/
│   └── extractor.py     — MetadataExtractor, extract_date, infer_domain, extract_tags
├── indexing/
│   ├── chunker.py       — MarkdownASTChunker
│   ├── hash_cache.py    — HashCache (JSON-backed), FileEntry
│   ├── indexer.py       — IncrementalIndexer (delta pipeline), IndexResult
│   └── repository.py    — ChunkRepository protocol, LanceDBRepository, _escape_sql_string
├── search/
│   ├── engine.py        — SemanticSearchEngine (public API), SearchOutcome
│   └── ranker.py        — SearchRanker (recency, domain, dedup, threshold)
├── cli/
│   └── app.py           — Typer CLI (index, search, stats, verify, doctor, inspect, rebuild)
└── utils/
    ├── hashing.py        — sha256_text/hash_file, TextEstimator
    └── logging.py        — configure_logging (Rich handler)

tests/                    — 124 test functions across 9 files
docs/                     — 7 documentation files
```

## Implementation decisions

### 1. Focused abstraction boundaries
`Embedder` and `ChunkRepository` are `typing.Protocol` classes because they have external implementations and test doubles. Chunking uses the concrete `MarkdownASTChunker` directly.

### 2. Frozen Pydantic models
All domain objects use `ConfigDict(frozen=True)`. Chunks, metadata, and search results are immutable as they flow through the pipeline. The indexer creates new `Chunk` instances with embeddings attached rather than mutating existing ones.

### 3. Content-hash incremental indexing
File content is SHA-256 hashed; the hash is persisted in a JSON cache. On each `palace index` run, only files whose hash changed are re-chunked and re-embedded. Unchanged files are skipped entirely — no embedding work, no DB writes. This makes incremental indexing proportional to what changed, not to corpus size.

### 4. Semantic Markdown chunking
`MarkdownASTChunker` handles all document domains. Markdown headings define section boundaries, heading hierarchy supplies context for YAKE keyword extraction, and headerless documents remain a single chunk.

### 5. Metadata pre-filtering in LanceDB
Domain, date range, and tags are pushed down to LanceDB as a SQL WHERE clause before vector search. This narrows the candidate set before the vector comparison runs, keeping search fast over large corpora. All SQL string values are escaped via `_escape_sql_string()` to prevent injection.

### 6. Multi-signal ranking
The ranker combines: semantic similarity (base), recency boost (linear decay within N days), domain weight (per-domain multiplier), and duplicate suppression (one result per source file). Results below `minimum_score` are dropped. This surfaces the most useful results rather than only the highest cosine similarity.

### 7. FakeEmbedder for testing
A deterministic, hash-based embedder produces reproducible vectors without any network calls or model downloads. It creates weak-but-usable lexical similarity (shared words → shared vector dimensions) sufficient for unit testing the pipeline. Set via `PALACE_EMBEDDER=fake` environment variable.

### 8. SQL injection prevention
All user-provided strings (source files, tags, domains) passed to LanceDB SQL filters are escaped by doubling single quotes. The `delete_chunks`, `_build_filter_sql`, and `filter_by_metadata` methods all use `_escape_sql_string()`.

### 9. MCP-ready search API
`SemanticSearchEngine.semantic_search()` takes plain Python types (strings, dates, lists) and returns plain Pydantic models (`list[SearchResult]`). An MCP server wrapping this is a thin adapter — it calls the method and serializes results. No engine changes are needed.

### 10. Configuration via injection
`PalaceConfig` is loaded once (from `config.yaml`) and passed to every component via constructor injection. No component reads the config file directly. Paths are resolved relative to `project_root`. This makes testing trivial — construct a `PalaceConfig.default_for(tmp_path)` and pass it in.

## Test coverage

124 test functions across 9 test files:

| Test file | Tests | Coverage area |
|---|---|---|
| `test_config.py` | 8 | YAML loading, path resolution, validation |
| `test_hashing.py` | 10 | SHA-256, file hashing, text estimation, token budget splitting |
| `test_metadata.py` | 22 | Date extraction (ISO/long/slash), domain inference, tag extraction (front-matter + inline) |
| `test_chunker.py` | 14 | Journal splitting, note chunking with overlap, hash overrides, front-matter preservation |
| `test_hash_cache.py` | 8 | Load/save, corruption handling, upsert/remove, parent dir creation |
| `test_repository.py` | 13 | Insert/delete/update/search, domain/date/tag filters, statistics, vacuum |
| `test_indexer.py` | 14 | Discovery, incremental skip/reindex/delete, hash cache persistence, verify |
| `test_embedder.py` | 16 | FakeEmbedder, OllamaEmbedder with mock HTTP (retry, batch, mismatch, health check) |
| `test_search.py` | 19 | Ranker (score/recency/domain/dedup/threshold), engine (filters, overrides, diagnostics) |

**Note:** Tests could not be executed in the build sandbox (no network to install pydantic, lancedb, etc.). All 28 Python files are verified to parse cleanly, and pure-logic tests (hashing, chunking, date/tag extraction) were validated manually. Run `./run_tests.sh` on a machine with network access to execute the full suite with coverage.

## Future improvements

1. **MCP server** — Wrap `SemanticSearchEngine.semantic_search()` in an MCP tool. The interface is already clean Python; only a thin serialization adapter is needed.

2. **Async embedding** — The current embedder is synchronous. For very large initial indexes (10,000+ entries), async httpx with connection pooling would speed up batch embedding.

3. **Embedding cache** — Currently, re-indexing a changed file re-embeds all its chunks. A content-hash-to-embedding cache would skip re-embedding for chunks whose text hasn't changed within a modified file (e.g., if you appended one entry to a journal, the existing entries wouldn't need re-embedding).

4. **Hybrid search** — Add BM25/keyword search alongside vector search and fuse the results. This improves recall for exact-match queries (names, medications, specific terms) that semantic search handles poorly.

5. **Chunk-level diffing** — Instead of per-file re-indexing, diff at the chunk level. When a journal entry is edited, only that entry's chunk is re-embedded; all other entries in the file are untouched.

6. **Streaming search** — For the MCP server use case, yield results as they're ranked rather than waiting for the full ranking pass.

7. **Multi-model support** — Allow different embedding models for different domains (e.g., a domain-specific model for medical journals).

8. **Tier 1 integration via skills** — Write Claude skills that read Tier 1 markdown deterministically and call Tier 2's search for unstructured recall. This is the planned fusion layer.

## Known limitations

1. **Ollama dependency** — The production embedder requires a running Ollama instance with `nomic-embed-text` pulled. The `FakeEmbedder` is available for testing but produces poor search quality.

2. **LanceDB SQL dialect** — The filter SQL uses LanceDB's specific SQL dialect. Migrating to a different vector store would require rewriting the filter builder in `repository.py`.

3. **Single-user, single-process** — No concurrency control. Running two `palace index` processes simultaneously could corrupt the hash cache or LanceDB table. This is acceptable for a personal system but would need locking for multi-user deployment.

4. **Token estimation is approximate** — The `TextEstimator` uses `words × 1.3` rather than a real tokenizer. Chunk sizes are approximate, not exact. This is intentional (avoids a tokenizer dependency) but means chunks may vary in actual token count.

5. **No incremental embedding cache** — When a file changes, all its chunks are re-embedded even if most are unchanged. The hash cache works at file granularity, not chunk granularity.

6. **Date parsing is Western-centric** — Slash dates assume either US (`M/D/Y`) or day-first (`D/M/Y` when first number > 12). Ambiguous dates like `4/5/2026` default to M/D/Y. Users in other locales may need to use ISO dates.

7. **No full-text search fallback** — Search is purely semantic. If the embedding model is poor at matching specific terms (medication names, numbers), there's no BM25 fallback to catch exact matches.

8. **Tests not executed in build environment** — The sandbox has no network access, so dependencies (pydantic, lancedb, typer, etc.) could not be installed. The test suite is written and all files parse cleanly, but the user must run `./run_tests.sh` on their machine to verify.
