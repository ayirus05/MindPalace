# MindPalace Tier 2 — Semantic Memory Engine

A local-first semantic search engine that sits alongside the Tier 1 structured markdown palace. Tier 2 indexes journals, notes, transcripts, and conversations into a LanceDB vector store using Ollama's `nomic-embed-text` model, then provides semantic search with metadata filtering, recency weighting, and duplicate suppression.

The entire system runs on your machine. No data leaves your computer.

## Quick start

```bash
# 1. Install Ollama and pull the embedding model
ollama pull nomic-embed-text

# 2. Install MindPalace
uv pip install -e ".[dev]"

# 3. Add journals/notes to the journals/ directory

# 4. Index
palace index

# 5. Search
palace search "energy fatigue patterns last month"

# 6. Check status
palace stats
palace doctor
```

## What Tier 2 does

Tier 1 (already built) stores structured facts as markdown files — personality, values, goals, communication style. Claude reads these deterministically for exact, current-state grounding.

Tier 2 handles the unstructured stuff: daily journals spanning months or years, conversation transcripts, raw notes that don't fit a clean schema. These are too voluminous and too messy for deterministic file reads, so they get embedded into a vector store and retrieved via semantic search.

A future skills layer will fuse the two: read the structured tier for exact facts (current goals, communication preferences), call semantic search for recall across journals, and synthesize grounded answers.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CLI (Typer + Rich)                    │
│         index · search · stats · verify · doctor          │
│              inspect · rebuild                             │
├─────────────┬──────────────────────┬─────────────────────┤
│   Indexer   │   Semantic Search   │   Configuration      │
│  (incremental│     Engine          │    (config.yaml)     │
│   delta)     │  (rank + filter)    │                     │
├─────────────┼──────────────────────┤                     │
│   Chunker   │   Search Ranker      │                     │
│  (journal / │  (recency, domain,   │                     │
│   note)     │   dedup, threshold)  │                     │
├─────────────┼──────────────────────┤                     │
│  Metadata   │                      │                     │
│  Extractor  │                      │                     │
├─────────────┴──────────────────────┴─────────────────────┤
│              Embedding Manager (Ollama)                   │
│          batch · retry · timeout · FakeEmbedder          │
├───────────────────────────────────────────────────────────┤
│           LanceDB Repository (the only DB layer)           │
│        insert · delete · update · search · vacuum          │
├───────────────────────────────────────────────────────────┤
│              LanceDB (local, file-based)                   │
│                    index/vector.lancedb                    │
└───────────────────────────────────────────────────────────┘
```

Key design constraint: LanceDB code lives exclusively in `repository.py`. Everything else talks to the repository through the `ChunkRepository` protocol, making the system testable without a real database and swappable to a different vector store.

## Package structure

```
MindPalace/
├── config.yaml              # all runtime configuration
├── pyproject.toml           # dependencies + build config
├── run_tests.sh             # test runner
├── journals/                # source: journal entries to index
├── index/
│   └── vector.lancedb       # LanceDB vector store (created on first index)
├── cache/
│   ├── hash_cache.json      # incremental indexing state
│   └── palace.log           # structured log
├── src/palace/
│   ├── models/
│   │   ├── chunk.py         # Chunk, ChunkMetadata, SearchResult, RawHit, IndexStats
│   │   └── config.py        # PalaceConfig + nested config models
│   ├── embeddings/
│   │   └── manager.py       # Embedder protocol + OllamaEmbedder + FakeEmbedder
│   ├── metadata/
│   │   └── extractor.py     # date/domain/tag extraction strategies
│   ├── indexing/
│   │   ├── chunker.py       # MarkdownASTChunker
│   │   ├── hash_cache.py    # JSON-backed incremental cache
│   │   ├── indexer.py       # IncrementalIndexer (delta pipeline)
│   │   └── repository.py    # LanceDBRepository (the only DB layer)
│   ├── search/
│   │   ├── engine.py        # SemanticSearchEngine (public API)
│   │   └── ranker.py        # SearchRanker (recency, domain, dedup)
│   ├── cli/
│   │   └── app.py           # Typer CLI with Rich output
│   └── utils/
│       ├── hashing.py       # SHA-256 + TextEstimator
│       └── logging.py       # Rich logging setup
├── tests/                   # pytest suite (80+ tests)
└── docs/                    # documentation
```

## CLI reference

| Command | Description |
|---|---|
| `palace index` | Index new and changed files (incremental). Use `--reindex` to force. |
| `palace search "query"` | Semantic search. Supports `--domain`, `--from`, `--to`, `--tag`, `--top-k`, `--min-score`, `--json`. |
| `palace stats` | Show index statistics (chunk count, domain breakdown, date range). |
| `palace verify` | Check the hash cache against the repository and filesystem. |
| `palace doctor` | Diagnose environment: Python, config, source dirs, LanceDB, Ollama. |
| `palace inspect` | Browse raw chunks. Filter by `--source` or a chunk ID argument. |
| `palace rebuild` | Compact the database and re-index everything from scratch. |

All commands accept `--config /path/to/config.yaml` to override the config file location.

## Configuration

All settings live in `config.yaml`. Key sections:

- **embedding** — Ollama model, host, batch size, retry policy
- **chunker** — target token size, overlap, tokens-per-word ratio
- **database** — LanceDB path, table name, distance metric
- **indexer** — source directories, file extensions, hash cache path
- **search** — prefilter and final top-K, minimum score, recency boost, domain weights, dedup
- **logging** — log level, Rich output, log file path

See `config.yaml` for full documentation of each field.

## Testing

```bash
./run_tests.sh
```

Tests use a `FakeEmbedder` (deterministic, no Ollama required) and real LanceDB in temporary directories. Coverage target: 90%.

To run without Ollama for smoke testing the CLI:
```bash
PALACE_EMBEDDER=fake palace index
PALACE_EMBEDDER=fake palace search "test query"
```

## MCP readiness

The `SemanticSearchEngine.semantic_search()` method exposes a clean Python interface:

```python
engine.semantic_search(
    query="energy fatigue last month",
    domain="journal",
    date_from="2026-04-01",
    date_to="2026-07-31",
    tags=["health"],
    top_k=8,
    minimum_score=0.15,
)
```

An MCP server wrapping this is a thin adapter — no changes to the engine needed. The engine holds no per-call state and is safe to reuse.
