# CLI Reference

All commands accept `--config` / `-c` to specify an alternate config file path.

## palace index

Index new and changed source files into the vector store.

```bash
palace index [--reindex] [--config PATH]
```

**Options:**
- `--reindex` — Force re-indexing of all files, ignoring the hash cache.
- `--config PATH` — Path to config.yaml (default: auto-detected in cwd).

**Behavior:**
- Walks configured source directories (`indexer.source_dirs`).
- Hashes every file. Unchanged files are skipped (no embedding work).
- Changed/new files are chunked, embedded in batches, and upserted.
- Files removed from disk are deleted from the index.
- Reports: files scanned, indexed, skipped, deleted; chunks created; errors.

**Example:**
```bash
palace index
palace index --reindex
```

---

## palace search

Semantic search across the index.

```bash
palace search QUERY [OPTIONS]
```

**Arguments:**
- `QUERY` — Natural language query (required).

**Options:**
- `--domain` / `-d` — Filter by domain (journal, notes, transcript, conversation).
- `--from` — Earliest date (ISO format: `2026-04-01`).
- `--to` — Latest date (ISO format).
- `--tag` / `-t` — Filter by tag (can be repeated for multiple tags).
- `--top-k` / `-k` — Number of results (default: 8).
- `--min-score` — Minimum similarity score (default: 0.15).
- `--json` — Output results as JSON.
- `--config PATH` — Config file path.

**Output:**
Rich table with score, date, domain, source file, and content excerpt. Also shows result count, search duration, and candidate count scanned.

**Examples:**
```bash
palace search "energy and sleep patterns"
palace search "supplement protocol" --domain journal --from 2026-04-01
palace search "rebalancing" --tag finance --top-k 3 --json
```

---

## palace stats

Show database statistics.

```bash
palace stats [--config PATH]
```

**Output:**
- Total chunks
- Unique source files
- Earliest and latest entry dates
- Database path
- Breakdown by domain

---

## palace verify

Check the index against the hash cache and filesystem.

```bash
palace verify [--config PATH]
```

**Behavior:**
- Loads the hash cache.
- Re-hashes every cached file on disk.
- Reports: cached file count, repository chunk count, healthy files (hash matches), stale files (hash changed), missing files (deleted from disk).

**Does not modify the index.** Use `palace index` to repair drift.

---

## palace doctor

Diagnose the MindPalace environment.

```bash
palace doctor [--config PATH]
```

**Checks:**
1. Python version and executable path
2. config.yaml found and readable
3. Each configured source directory exists
4. Database path accessible
5. LanceDB Python package installed
6. Ollama reachable at configured host
7. Embedding model available in Ollama

**Exit:** Non-zero if critical components are missing.

---

## palace inspect

Browse raw chunks in the index.

```bash
palace inspect [CHUNK_ID] [--source PATH] [--limit N] [--config PATH]
```

**Arguments:**
- `CHUNK_ID` — Optional chunk ID to inspect a specific chunk.

**Options:**
- `--source` / `-s` — Filter by source file path.
- `--limit` / `-n` — Max chunks to display (default: 10).
- `--config PATH` — Config file path.

**Output:**
Rich table with chunk ID, source file, date, word count, and content excerpt.

**Examples:**
```bash
palace inspect                              # browse all chunks
palace inspect --source journals/2026-04.md # chunks from one file
palace inspect abc123def456                 # one specific chunk
```

---

## palace rebuild

Drop the index and rebuild from scratch.

```bash
palace rebuild [--config PATH]
```

**Behavior:**
- Compacts the existing table (vacuum).
- Clears the hash cache.
- Re-indexes every source file.

Use this when the embedding model changed, the index is corrupt, or you want to reclaim disk space.

---

## Environment variables

| Variable | Description |
|---|---|
| `PALACE_EMBEDDER` | Set to `fake` to use the deterministic test embedder instead of Ollama. Useful for smoke testing without Ollama running. |
