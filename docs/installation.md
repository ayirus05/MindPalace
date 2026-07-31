# Installation Guide

## Prerequisites

- **Python 3.10+** (3.12+ recommended)
- **Ollama** — local LLM/embedding server
- **uv** (recommended) or pip

## Step 1: Install Ollama

### macOS
```bash
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download the installer from [ollama.com](https://ollama.com).

## Step 2: Pull the embedding model

```bash
ollama pull nomic-embed-text
```

This downloads ~270 MB. Verify it's available:
```bash
ollama list
```

## Step 3: Install MindPalace

From the MindPalace directory:

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, pytest-cov, and respx for tests.

## Step 4: Verify the installation

```bash
palace doctor
```

This checks:
- Python version and path
- config.yaml found
- Source directories exist
- LanceDB installed
- Ollama reachable and model pulled

If everything is green, you're ready to index.

## Step 5: Add source files

Place journal entries, notes, and transcripts in the directories configured in `config.yaml` (default: `journals/`). Supported formats: `.md`, `.txt`.

Journal files should have dated entries:
```markdown
2026-04-15

Energy has been low this week. Started a new supplement protocol
on Monday and I'm tracking sleep metrics to see if there's a pattern.

2026-04-22

Sleep metrics dipped again. Going to pause the evening dose.
```

## Step 6: Index and search

```bash
palace index
palace search "energy and sleep patterns"
```

## Running without Ollama (smoke testing)

If Ollama isn't available but you want to test the pipeline:

```bash
export PALACE_EMBEDDER=fake
palace index
palace search "test query"
```

This uses a deterministic hash-based embedder that produces weak but usable similarity for testing. No data leaves your machine.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues.
