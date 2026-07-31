# Troubleshooting

## `palace index` fails with "Ollama is not reachable"

**Cause:** Ollama isn't running, or the embedding model isn't pulled.

**Fix:**
```bash
# Start Ollama
ollama serve

# Pull the model
ollama pull nomic-embed-text

# Verify
ollama list
```

If Ollama is running on a non-default host, update `config.yaml`:
```yaml
embedding:
  host: "http://your-host:11434"
```

**Workaround for testing without Ollama:**
```bash
export PALACE_EMBEDDER=fake
palace index
```

---

## `palace doctor` shows "model nomic-embed-text not pulled"

The model needs to be downloaded:
```bash
ollama pull nomic-embed-text
```

---

## `palace doctor` shows "lancedb not installed"

Install dependencies:
```bash
uv pip install -e ".[dev]"
```

---

## `palace index` finds zero files

Check that:
1. Your source files are in the directory configured in `config.yaml` under `indexer.source_dirs` (default: `journals/`).
2. Files have a `.md` or `.txt` extension (configured in `indexer.extensions`).
3. The source directory exists relative to `config.yaml`'s location.

```bash
palace doctor
```
This will show whether each source directory exists.

---

## `palace search` returns no results

1. **Index is empty:** Run `palace stats` to confirm chunks exist.
2. **Minimum score too high:** Try `palace search "query" --min-score 0.0`.
3. **No matching domain:** If you're filtering by `--domain`, make sure files are in the right directory (e.g., `journals/` for the `journal` domain).
4. **Date range too narrow:** Widen `--from` and `--to`.

---

## `palace search` results are irrelevant

1. **Check your embedding model:** Results depend on `nomic-embed-text` quality. Using `PALACE_EMBEDDER=fake` produces poor results — only use that for smoke testing.
2. **Adjust domain weights:** In `config.yaml`, under `search.domain_weights`, boost domains you care about.
3. **Tune minimum_score:** Lower it for more recall, raise it for more precision.
4. **Increase prefilter_top_k:** Higher values give the ranker more candidates to work with.

---

## Hash cache is corrupt

If `cache/hash_cache.json` is corrupt, it's automatically treated as empty on next load — all files will be re-indexed. To force a clean rebuild:

```bash
rm cache/hash_cache.json
palace index
```

---

## Embedding errors after upgrading Ollama

If you update Ollama or change the embedding model, existing vectors may have incompatible dimensions. Rebuild from scratch:

```bash
palace rebuild
```

---

## LanceDB table is locked

If a previous `palace index` was interrupted, the LanceDB table might be locked. Delete the index and rebuild:

```bash
rm -rf index/vector.lancedb
palace index
```

---

## Performance is slow

1. **Incremental indexing:** Make sure you're not running `palace index --reindex` every time. Plain `palace index` only re-embeds changed files.
2. **Batch size:** Increase `embedding.batch_size` in `config.yaml` if your machine has ample RAM.
3. **Vacuum:** Run `palace stats` → if chunk count is high, run `palace rebuild` periodically to compact the table.

---

## Permission denied errors

Ensure the `index/`, `cache/`, and `journals/` directories are writable by the user running `palace`.

---

## Still stuck

Run `palace doctor` and check the output. If that doesn't identify the problem, check the log file at `cache/palace.log` for detailed error messages.
