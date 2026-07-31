# MindPalace

A personal knowledge dossier — a structured, file-based store of information about one person (you) that Claude can query on demand to produce personalized answers grounded in your actual profile, rather than relying on whatever happens to fit in a context window.

## Architecture

The MindPalace is a hybrid two-tier system:

### Tier 1 — Structured core (deterministic, exact)
Small markdown files keyed by domain, each with YAML frontmatter, fronted by an always-loaded `INDEX.md`. Claude reads these directly with the Read tool. This is where current-state facts live: personality profile, health routines, finance snapshot, etc. Nothing here is ever embedded — you want exact records, not semantic neighbors.

### Tier 2 — Unstructured index (planned, not yet built)
A local vector store (LanceDB + `nomic-embed-text` via Ollama) for voluminous, messy text: journals, conversation logs, raw notes. Claude reaches it via an MCP `semantic_search` tool. This is where embeddings earn their keep — recall across volume where structure doesn't exist.

### Skills layer (planned)
Reusable workflow instructions that tell Claude, for a given query type, exactly which files to read, when to invoke semantic search, and how to synthesize the two. The skill is what fuses the tiers and produces grounded answers instead of generic advice.

## Folder structure

```
MindPalace/
  INDEX.md                # always-loaded orientation map
  README.md               # this file
  personality/            # Tier 1 — personality domain records
    profile.md            # identity, roles, life stage
    personality.md        # Big Five, temperament, dispositions
    values.md             # core values, money/health mindset
    communication.md      # tone, formality, directness
    motivation.md         # what energizes, goals, demotivators
    cognition.md          # decision-making, learning style
    relationships.md      # social energy, attachment, conflict style
    stress-response.md    # stress signals, coping, recovery
    life-context.md       # daily routine, energy patterns, season of life
    goals.md              # short/long-term goals, success definition
  questionnaire/          # the HTML app that populates the personality tier
    mindpalace_questionnaire.html
  skills/                 # workflow skills (planned)
```

## How to populate the personality tier

1. Open `questionnaire/mindpalace_questionnaire.html` in any browser.
2. Work through the eight categories. Each starts broad, then branches based on your answers to capture nuance.
3. Export the generated markdown (per-file download or copy-to-clipboard).
4. Save each file into `personality/` over the corresponding stub.
5. Update `INDEX.md`'s "At a glance" section with the headline facts.
6. Ask Claude questions — it reads `INDEX.md` automatically, then pulls the relevant domain file(s) and synthesizes grounded answers.

## Design principles

- **Files are the database.** No server, no cloud, no daemon for Tier 1. You own your data.
- **Minimal data, maximal synthesis logic.** One person's data is small — the differentiator isn't seeing more, it's the skill telling Claude exactly which three files to read and how to combine them.
- **Structured before embedded.** Embeddings deliberately trade precision for recall. For current-state facts you want the exact record, not a semantic neighbor. So the structured tier handles precision; embeddings only supplement where structure doesn't exist.
- **Local-first for privacy.** Health, finance, and personality data is sensitive. Keep it on-device.
- **Self-maintaining with `updated:` dates.** Every record carries an `updated:` field so skills can flag stale data before relying on it.

## Privacy

Everything in this palace runs locally. No data leaves your machine in Tier 1. When Tier 2 is added, the embedding model (Ollama) and vector store (LanceDB) also run locally.
