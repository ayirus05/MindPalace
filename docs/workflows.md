# Example Workflows

## Workflow 1: Daily journal indexing

You keep a monthly journal file in `journals/2026-04.md`. Each day you append a dated entry. At the end of the week, you index:

```bash
palace index
```

Only the new entries get embedded (the file hash changed, so the whole file is re-chunked and re-embedded — but unchanged files are skipped). The hash cache persists between runs.

Search for patterns:
```bash
palace search "energy levels and sleep quality" --from 2026-04-01 --to 2026-04-30
```

## Workflow 2: Research notes

You keep research notes in `notes/` with YAML front-matter tags:

```markdown
---
tags: [finance, quarterly-review]
---
# Q2 Portfolio Review

Tech allocation at 32% vs 25% ceiling. Need to rebalance before
quarter end. Healthcare and energy are underweight...
```

Index and search by tag:
```bash
palace index
palace search "rebalancing strategy" --tag finance
```

## Workflow 3: Conversation transcripts

Transcripts go in `transcripts/` (add to `config.yaml` → `indexer.source_dirs`):

```yaml
indexer:
  source_dirs:
    - "journals"
    - "notes"
    - "transcripts"
```

Search across all domains or filter to transcripts:
```bash
palace search "discussion about risk tolerance" --domain transcript
```

## Workflow 4: Combined search with ranking

Search across everything, weighting journals higher:

```yaml
# config.yaml
search:
  domain_weights:
    journal: 1.2
    notes: 1.0
    transcript: 0.8
```

```bash
palace search "why I felt burned out in March" --top-k 5
```

The ranker combines semantic similarity, recency (recent entries boosted), domain weight, and deduplication (one result per source file) to surface the most useful hits.

## Workflow 5: Verify and maintain

After editing many files, verify the index is consistent:
```bash
palace verify
```

This re-hashes every cached file and reports: healthy (hash matches), stale (hash changed, needs reindex), missing (file deleted).

To repair drift:
```bash
palace index        # re-indexes only stale/missing files
# or for a full rebuild:
palace rebuild
```

Check statistics:
```bash
palace stats
```

## Workflow 6: Inspect specific chunks

Find a chunk and inspect its full content:
```bash
# List chunks from a specific file
palace inspect --source journals/2026-04.md

# Inspect by chunk ID
palace inspect abc123def456
```

## Workflow 7: JSON output for scripting

Pipe search results to another tool:
```bash
palace search "supplement protocol" --json | jq '.[].content'
```

## Workflow 8: Diagnose environment issues

```bash
palace doctor
```

Checks Python version, config file, source directories, LanceDB installation, and Ollama connectivity in one pass.
