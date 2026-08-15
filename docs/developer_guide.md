# Developer Guide

## Development setup

```bash
# Clone and install with dev dependencies
uv pip install -e ".[dev]"

# Run the test suite with coverage
./run_tests.sh

# Or run individual test files
python -m pytest tests/test_chunker.py -v
```

## Project conventions

### Type hints
Every function has type hints on all parameters and return values. The codebase targets Python 3.10+, so `from __future__ import annotations` is used in every file for deferred evaluation (allowing `X | None` syntax everywhere).

### Pydantic models
All domain objects are frozen Pydantic v2 models (`model_config = ConfigDict(frozen=True)`). This prevents accidental mutation of chunks, metadata, and search results as they flow through the pipeline.

### Concrete production components
Embedding, persistence, and chunking use `OllamaEmbedder`, `LanceDBRepository`, and `MarkdownASTChunker` directly. Test doubles live under `tests/`, outside the production package.

### Dependency injection
No global state. Every class receives its dependencies via the constructor:
- `IncrementalIndexer(config, embedder, repository, hash_cache)`
- `SemanticSearchEngine(config, embedder, repository, ranker)`

This means tests construct components with test doubles directly, no monkeypatching of module-level globals.

### Logging
Each module gets its own logger: `logging.getLogger("palace.<module>")`. The CLI configures logging centrally via `configure_logging()` which uses Rich's console handler. Loggers never call `logging.basicConfig()` — they propagate to the palace root logger.

### No raw DB code outside repository.py
LanceDB is imported in exactly one file. The indexer and search engine use `LanceDBRepository`; don't import `lancedb` elsewhere.

## Changing chunking behavior

1. Update `MarkdownASTChunker` in `indexing/chunker.py`.
2. Preserve the `chunk_file` and `chunk_text` public methods.
3. Add tests in `tests/test_chunker.py`.

## Adding a new metadata field

1. Add the field to `ChunkMetadata` in `models/chunk.py`.
2. Add it to the LanceDB schema in `repository.py` (`SCHEMA` and `_chunk_to_row`).
3. Populate it in `MetadataExtractor.build_metadata()` or `MarkdownASTChunker.chunk_text()`.
4. Update `statistics()` if it's worth aggregating.
5. Add it to `RawHit` and `SearchResult` if it should surface in search.

## Adding a search filter

1. Add the parameter to `SemanticSearchEngine.search()`.
2. Pass it to `repository.search()` and add a clause in `_build_filter_sql()`.
3. Test it in `tests/test_search.py`.

## Adding a CLI command

1. Add a function to `cli/app.py` decorated with `@app.command()`.
2. Use Rich tables for output and `typer.Option` for flags.
3. Wire up config loading via `_load_config()` and construct components via the `_make_*` helpers.

## Writing an MCP server (future)

The `SemanticSearchEngine` is designed to be wrapped by an MCP server with no changes. The interface is plain Python types:

```python
from palace.search.engine import SemanticSearchEngine

engine = SemanticSearchEngine(config, embedder, repository)

# This is the method an MCP tool would call:
outcome = engine.search(
    query="energy fatigue last month",
    domain="journal",
    date_from="2026-04-01",
    date_to="2026-07-31",
    tags=["health"],
    top_k=8,
    minimum_score=0.15,
)
# outcome.results: list[SearchResult] — each has .content, .score, .source_file, .date, .domain, .tags
```

The MCP server is a thin adapter: take the tool arguments, call `search`, and serialize `outcome.results` to the MCP response format. No engine changes required.

## Test strategy

- **Test-only embedder** — deterministic and defined in `tests/fakes.py`, so unit tests do not require Ollama.
- **Real LanceDB** — tests use real LanceDB in `tmp_path` directories (via the `repository` fixture). This validates actual DB operations.
- **Mock HTTP** — `test_embedder.py` uses a `_FakeClient` (subclassing `httpx.Client`) to test retry logic and response parsing without network calls.
- **`tmp_path` fixtures** — `conftest.py` provides a `tmp_project` fixture that creates the standard directory structure in a temp directory, plus `sample_journal` and `sample_note` fixtures with realistic content.

## Code review checklist

Before submitting changes:
- [ ] All functions have type hints
- [ ] No raw LanceDB code outside `repository.py`
- [ ] No global mutable state
- [ ] New config fields documented in `config.yaml` and `docs/`
- [ ] Tests added or updated
- [ ] `palace doctor` passes
- [ ] `./run_tests.sh` passes with ≥90% coverage
