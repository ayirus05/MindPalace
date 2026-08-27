# MindPalace and Skill Router Roadmap

## Purpose

This roadmap describes the long-term evolution of MindPalace from a local semantic
memory engine into a production-ready retrieval and skill-execution platform. The
target architecture preserves the project's local-first, deterministic, and
model-agnostic principles while adding better retrieval quality, broader ingestion,
safer automation, and reliable unattended operation.

The phases express implementation order and dependency, not fixed release dates:

| Priority | Goal | Exit condition |
| --- | --- | --- |
| **Phase 1 — Foundations** | Establish stable contracts for retrieval, tools, filtering, and asynchronous operation. | Core interfaces are versioned, tested, observable, and usable from both the CLI and API. |
| **Phase 2 — Intelligence and safety** | Improve result quality and introduce controlled tool selection and execution. | Routing and ranking can be evaluated offline; destructive operations cannot run without an explicit policy decision. |
| **Phase 3 — Scale and autonomy** | Support heterogeneous sources, tiered memory, and reliable headless operation. | Long-running deployments can index incrementally, recover from failure, and operate within configured resource limits. |

Cross-cutting requirements for every phase are local-first defaults, typed boundaries,
dependency injection, idempotent operations, structured telemetry, and deterministic
fallback behavior when an optional model is unavailable.

---

## 1. MindPalace Retrieval Engine

The retrieval engine should combine high-recall lexical and semantic discovery with
strict filters and a high-precision final ranking stage. Retrieval remains independent
from any specific LLM provider so the same engine can serve the CLI, API, MCP tools,
and the Skill Router.

### 1.1 Hybrid BM25 + LanceDB vector search

**Priority: Phase 1**

**Architectural breakdown**

- Introduce a `Retriever` contract with lexical and vector implementations. The
  existing LanceDB repository remains the owner of vector persistence and search;
  a BM25 index stores normalized terms, document frequencies, and chunk identifiers.
- Feed both indexes from the same canonical chunk stream. A successful indexing
  transaction records the content hash, embedding version, lexical-index version,
  and shared `chunk_id`, preventing drift between the two result sets.
- Execute lexical and vector retrieval concurrently after applying the same metadata
  filter plan. Each retriever returns a larger candidate pool plus its native score.
- Fuse candidates by stable `chunk_id` using Reciprocal Rank Fusion initially. Add
  configurable lexical/vector weights only after an evaluation corpus demonstrates
  that weighted score normalization improves quality.
- Preserve source attribution and parent-document links through fusion, then pass the
  fused candidate list to the re-ranking stage. If Ollama or the vector index is
  unavailable, return BM25 results with a degraded-mode marker rather than failing the
  entire query.

```text
Query + filters
      ├── BM25 retriever ────────┐
      └── LanceDB retriever ─────┤
                                 v
                     reciprocal-rank fusion
                                 v
                    metadata-safe candidates
```

**Rationale**

Vector search handles paraphrases and conceptual similarity, while BM25 is stronger
for exact names, identifiers, uncommon phrases, and error codes. Combining them
reduces the main failure modes of either technique without requiring an LLM for
first-stage retrieval. A lexical fallback also keeps the local knowledge base useful
when the embedding service is offline.

**Key deliverables**

- Versioned lexical-index format and rebuild command.
- Shared candidate/result models with per-retriever score provenance.
- Fusion configuration, latency metrics, and degraded-mode reporting.
- Golden-query evaluation set measuring recall at *k*, mean reciprocal rank, and
  filter correctness against vector-only and BM25-only baselines.

### 1.2 Deterministic metadata filtering in tool schemas

**Priority: Phase 1**

**Architectural breakdown**

- Define one versioned Pydantic filter model and derive JSON Schema from it for CLI,
  FastAPI, MCP, and Skill Router tool declarations. Supported fields should include
  domain, source, date range, tags, content type, sensitivity, and ingestion adapter.
- Parse and validate filters before retrieval. Reject unknown fields, invalid enum
  values, inverted date ranges, and ambiguous dates instead of silently broadening a
  search.
- Compile the validated model into a backend-neutral filter expression, then translate
  that expression inside each repository adapter. Parameterize or safely escape every
  predicate; no caller may assemble LanceDB SQL fragments directly.
- Apply identical filter semantics to BM25 and vector candidate generation. Re-check
  the predicate on fused results as a defense-in-depth invariant and include the
  normalized filter plan in trace output.
- Version filter schemas and advertise supported versions through tool metadata so an
  older client fails explicitly when it requests unsupported behavior.

**Rationale**

Metadata constraints often encode correctness and privacy boundaries, not merely
ranking preferences. A model must not be able to invent ad hoc query syntax or turn an
invalid filter into an unrestricted search. One generated schema eliminates semantic
drift among the Python API, command line, FastAPI, and model-facing tools.

**Key deliverables**

- `SearchFilter` model, JSON Schema snapshot, and backend-neutral filter AST.
- Contract tests proving equivalent filtering across retrievers and interfaces.
- Property-based tests for quoting, empty sets, date boundaries, and invalid input.
- Structured audit fields for requested, normalized, and enforced filters.

### 1.3 Semantic re-ranking via cross-encoders

**Priority: Phase 2**

**Architectural breakdown**

- Add a `Reranker` interface after hybrid fusion. Its primary implementation scores
  `(query, candidate text)` pairs with a local cross-encoder; a no-op implementation
  preserves fused order when the model is disabled or unavailable.
- Re-rank only a bounded candidate window (for example, the top 30–100 fused hits),
  batch inference, and apply a strict latency budget. The final `top_k` is selected
  after re-ranking.
- Combine cross-encoder relevance with deterministic policy signals such as recency,
  domain weighting, and deduplication. Keep each component score in result provenance
  so ranking decisions can be inspected rather than collapsed into an opaque number.
- Store the model identifier, revision, tokenizer settings, and scoring configuration
  with evaluation runs. Cache query/candidate scores by query hash, chunk hash, and
  model version where this provides a measurable latency benefit.
- Make CPU-friendly local inference the default. GPU execution is an optional adapter,
  and timeouts fall back to fused ranking without dropping the query.

**Rationale**

Bi-encoder embeddings are efficient for broad candidate generation but score the query
and document independently. Cross-encoders jointly inspect each pair and are better at
fine distinctions, negation, and contextual relevance. Restricting them to a small
candidate set captures that precision without putting model inference on every stored
chunk.

**Key deliverables**

- Pluggable cross-encoder and no-op re-ranker implementations.
- Configurable candidate window, batch size, timeout, and model revision.
- Offline comparison against the Phase 1 hybrid baseline using nDCG at *k*, MRR,
  latency percentiles, and memory usage.
- Explainable result provenance and timeout/fallback telemetry.

### 1.4 Polymorphic ingestion adapters for non-Markdown sources

**Priority: Phase 3**

**Architectural breakdown**

- Define an `IngestionAdapter` protocol responsible for source discovery, extraction,
  normalization, metadata mapping, and stable source identity. Markdown becomes the
  reference adapter instead of special-case logic embedded in the indexer.
- Normalize every input into a `SourceDocument` containing canonical text, structural
  segments, source metadata, timestamps, access labels, attachments, and provenance.
  Existing chunkers consume this intermediate form rather than raw files.
- Select adapters through a MIME type/URI registry. Initial adapters should target
  high-value local formats such as plain text, PDF, HTML, JSON exports, and chat or
  email archives; remote connectors remain optional and explicitly configured.
- Give each adapter a checkpoint cursor and deterministic content fingerprint so
  incremental indexing can resume after interruption and can distinguish updates,
  moves, and deletions.
- Isolate parsers with resource limits. Preserve the original URI and extraction
  warnings on every chunk, quarantine malformed or encrypted inputs, and never claim
  successful indexing for a partial extraction without surfacing that state.

**Rationale**

Personal knowledge is spread across documents, exports, transcripts, and application
archives. A normalized adapter boundary broadens coverage without coupling retrieval
to every source format, while stable identities and provenance retain the deterministic
incremental behavior that makes MindPalace inexpensive to maintain.

**Key deliverables**

- Adapter protocol, registry, normalized document model, and capability metadata.
- Markdown reference adapter plus prioritized non-Markdown adapters.
- Fixture corpus covering encoding errors, duplicates, updates, deletions, and partial
  extraction.
- Per-adapter health, throughput, checkpoint, and quarantine reporting.

---

## 2. Skill Router Control Plane

The Skill Router is the policy and orchestration layer between a user's intent and
available capabilities. It should decide which tools are eligible, make execution
constraints visible, and maintain only the memory required for the current task.

### 2.1 Model-agnostic tool registry using JSON Schema

**Priority: Phase 1**

**Architectural breakdown**

- Create a central `ToolRegistry` whose entries contain a stable name, semantic
  version, description, JSON Schema input/output contracts, capability tags, safety
  classification, timeout, idempotency declaration, and executor reference.
- Generate schemas from typed Python models where possible, while allowing externally
  supplied standards-compliant schemas. Validate registration at startup and validate
  every invocation before dispatch and every result before returning it.
- Add provider adapters that translate the canonical registry entry into each model
  vendor's tool/function format. Provider-specific names, limits, and serialization
  rules remain outside tool implementations.
- Support scoped registry views. The router exposes only tools authorized for the
  current user, environment, and task, limiting both accidental selection and prompt
  surface area.
- Record registry version and tool version on execution traces. Reject incompatible
  versions explicitly rather than attempting an unsafe best-effort conversion.

**Rationale**

A canonical registry prevents the control plane from becoming coupled to one model's
function-calling format. JSON Schema provides a portable contract for validation,
documentation, test generation, and discovery, while scoped views support least
privilege.

**Key deliverables**

- Registry, registration lifecycle, schema validator, and versioning rules.
- Adapters for the initially supported model providers plus an MCP-facing projection.
- CLI commands to list, inspect, validate, and dry-run registered tools.
- Contract tests using valid, boundary, and malformed payloads.

### 2.2 Intent-based pre-routing classifier

**Priority: Phase 2**

**Architectural breakdown**

- Insert a lightweight classifier before model tool selection. It maps a request to a
  hierarchical intent, confidence, required capabilities, risk class, and whether
  clarification is necessary.
- Use a deterministic rules layer for explicit commands and high-risk keywords, then a
  small local classifier or embedding-based classifier for ambiguous natural language.
  The full reasoning model remains a fallback, not the default classifier.
- Query the registry by capability and policy to produce a short eligible-tool set.
  Low-confidence classifications broaden the set conservatively or ask the user for
  clarification; they never bypass safety policy.
- Allow multi-intent plans, but bound their size and retain the classifier decision,
  selected candidates, confidence, and final tool choice in an auditable route trace.
- Train and calibrate against a versioned, privacy-safe intent corpus with explicit
  out-of-domain examples and adversarial prompts.

```text
User request → rules/classifier → intent + risk + confidence
                                      ↓
                              policy-filtered registry
                                      ↓
                             eligible tools or clarify
```

**Rationale**

Presenting every tool to a model increases latency, token use, and the chance of an
irrelevant or unsafe selection. Pre-routing narrows the decision space while retaining
a measurable path for ambiguity and out-of-domain requests.

**Key deliverables**

- Versioned intent taxonomy and labeled evaluation corpus.
- Rules, classifier adapter, confidence calibration, and out-of-domain handling.
- Route-trace model and metrics for top-*k* accuracy, unsafe false negatives,
  clarification rate, latency, and tool-set reduction.
- Shadow mode for comparing classifier decisions with existing routing before rollout.

### 2.3 Sandboxed execution and human-in-the-loop CLI prompts for destructive actions

**Priority: Phase 2**

**Architectural breakdown**

- Classify tools and concrete invocations as read-only, reversible write, destructive,
  privileged, or external side effect. Evaluate policy after arguments are validated,
  because risk depends on resolved targets as well as the tool name.
- Run tools through an execution broker that applies filesystem/network scopes,
  environment allowlists, time and memory limits, output caps, and cancellation. The
  broker emits an immutable execution receipt containing the policy and result.
- For destructive actions, first produce a dry-run preview with exact targets and
  effects. Interactive CLI sessions require an explicit confirmation token tied to the
  normalized invocation; changing an argument invalidates the approval.
- In headless mode, deny confirmation-required actions by default. Permit them only
  through a narrowly scoped, expiring policy grant supplied out of band, never by
  interpreting natural-language consent in the prompt.
- Define recovery hooks for reversible changes and clearly label operations that
  cannot be rolled back. Redact secrets from previews, logs, and model-visible output.

**Rationale**

Correct routing does not guarantee safe execution. Sandboxing limits the blast radius
of defects and prompt injection, while a preview-bound approval ensures a person
consents to the actual action rather than a vague description of it.

**Key deliverables**

- Policy engine, risk taxonomy, execution broker, and platform-specific sandbox
  capability detection.
- Rich CLI previews and confirm/deny flow with non-interactive fail-closed behavior.
- Approval receipts, redaction rules, cancellation, and reversible-action hooks.
- Security tests for path traversal, argument mutation, privilege escalation, timeout,
  prompt injection, and approval replay.

### 2.4 Multi-tiered working memory management

**Priority: Phase 3**

**Architectural breakdown**

- Separate memory into four explicit tiers: turn-local scratch state, active-task
  working memory, durable episodic summaries, and long-term MindPalace knowledge.
- Assign each tier a retention policy, token/storage budget, sensitivity label, and
  promotion rule. Raw tool output stays ephemeral unless a policy promotes a minimal,
  sourced summary.
- Add a `MemoryManager` that selects context by relevance, recency, task identity, and
  budget. It compacts older task events into structured summaries while retaining
  pointers to immutable source records for verification.
- Require provenance, timestamps, and confidence for promoted facts. Contradictory
  durable memories are retained as conflicts for resolution rather than silently
  overwritten.
- Support task completion, user-requested forget, retention expiry, and crash recovery.
  Sensitive tiers are encrypted at rest where configured and are excluded from remote
  models unless policy explicitly permits disclosure.

| Tier | Typical contents | Lifetime |
| --- | --- | --- |
| Turn-local | Intermediate reasoning state and transient tool output | One interaction |
| Active task | Decisions, open steps, selected evidence, execution receipts | Until completion or expiry |
| Episodic | Compact summaries of completed work with source pointers | Policy-controlled |
| Long-term | User-approved knowledge indexed by MindPalace | Durable until removed |

**Rationale**

Treating all history as one context window is expensive, noisy, and privacy-unfriendly.
Explicit tiers provide predictable retention and bounded prompts, while provenance and
promotion rules prevent transient model output from becoming unverified personal fact.

**Key deliverables**

- Memory data model, budgets, promotion/expiry policies, and persistence adapters.
- Deterministic context assembly with token estimates and truncation traces.
- Compaction quality tests, contradiction handling, deletion guarantees, and recovery
  tests.
- User-facing inspection and forget commands.

---

## 3. Daemon & Production Deployment

The production layer should expose the same application services used by the CLI,
without duplicating business logic. It must remain practical for a single-user local
deployment while defining the lifecycle, observability, and failure behavior needed
for an always-on service.

### 3.1 Asynchronous FastAPI event loop

**Priority: Phase 1**

**Architectural breakdown**

- Use FastAPI lifespan hooks to initialize configuration, repositories, registries,
  model clients, and bounded executors once, then close them cleanly during shutdown.
  Request handlers receive these services through dependency injection.
- Convert I/O paths to native async where supported. Place blocking LanceDB, parser,
  and local inference calls in bounded worker pools so they cannot stall the event
  loop.
- Represent indexing and other long operations as jobs. Endpoints enqueue work and
  return a job identifier; clients inspect status or consume progress events instead
  of holding a request open indefinitely.
- Apply concurrency limits and backpressure separately to search, indexing, embedding,
  and tool execution. Propagate cancellation and define timeouts at each boundary.
- Provide health, readiness, and metrics endpoints. Structured logs and traces include
  request/job IDs, phase timings, queue depth, model/index versions, and degraded-mode
  state without exposing private content.

**Rationale**

FastAPI alone does not make blocking storage or model calls asynchronous. Explicit
lifecycles, worker boundaries, and backpressure keep interactive search responsive
while indexing or inference is active and make daemon failures diagnosable.

**Key deliverables**

- Application factory and lifespan-managed service container.
- Async search API, bounded worker pools, job API, cancellation, and graceful shutdown.
- Health/readiness/metrics surfaces and structured request/job tracing.
- Concurrency, saturation, disconnect, restart, and shutdown integration tests.

### 3.2 Headless cron-triggered background indexing

**Priority: Phase 3**

**Architectural breakdown**

- Provide a non-interactive `palace index --headless` entry point with stable exit
  codes, structured logs, explicit config selection, and no prompts. Cron or a system
  scheduler remains responsible for timing; the daemon exposes the same indexing job
  service for deployments that prefer an API trigger.
- Acquire a per-vault lease before indexing to prevent overlapping cron runs. Include
  owner identity, heartbeat, expiry, and safe stale-lease recovery.
- Persist job checkpoints and per-adapter cursors. Reuse content-hash incremental
  indexing and idempotent repository updates so interrupted runs can resume without
  duplicating chunks.
- Stage index updates and publish them atomically where backend capabilities allow.
  Keep the previous readable generation until the new generation passes integrity
  checks; isolate individual source failures rather than discarding a healthy batch.
- Emit a machine-readable run summary with scanned, changed, indexed, deleted,
  quarantined, skipped, and failed counts. Integrate configurable notifications only
  after repeated failure or staleness thresholds, not for every successful run.

```text
cron/system timer → headless index command → acquire vault lease
                                            ↓
                                  discover + checkpoint + index
                                            ↓
                                 verify → publish → run summary
```

**Rationale**

Scheduled incremental indexing keeps retrieval current without an always-running file
watcher and fits the project's local-first deployment model. Leases, checkpoints, and
atomic publication make unattended runs predictable across crashes, slow parsers, and
machine restarts.

**Key deliverables**

- Headless command contract, documented cron/systemd examples, and stable exit codes.
- Vault lease, checkpoint persistence, retry/backoff policy, and stale-job recovery.
- Atomic or generation-based publishing strategy with integrity verification.
- Run-history inspection, stale-index detection, retention cleanup, and failure
  notification hooks.

---

## Dependency sequence and release gates

The roadmap should be delivered as vertical slices rather than as isolated subsystems:

1. **Phase 1:** stabilize filter and tool schemas, add hybrid retrieval, and make the
   FastAPI lifecycle safe under concurrency. The release gate is deterministic
   contract behavior across CLI, API, MCP, and tests.
2. **Phase 2:** add cross-encoder re-ranking, intent pre-routing, and the sandboxed
   execution broker. The release gate is measured retrieval/routing improvement plus
   fail-closed safety behavior under adversarial tests.
3. **Phase 3:** add polymorphic ingestion, tiered memory, and scheduled headless
   indexing. The release gate is restart-safe, bounded, observable operation on a
   representative multi-format vault over repeated unattended runs.

Before enabling a later phase by default, preserve the preceding phase as a fallback:
BM25 remains available without embeddings, fused retrieval remains available without a
cross-encoder, direct registry lookup remains available if the classifier is disabled,
and interactive operation remains available if the daemon is not running.
