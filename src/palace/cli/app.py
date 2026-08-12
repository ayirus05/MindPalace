"""MindPalace Tier 2 command-line interface.

Built with Typer + Rich.  Every command loads :class:`PalaceConfig` and
wires up the real embedder/repository/indexer (or their test doubles when
``PALACE_EMBEDDER=fake`` is set — useful for smoke tests without Ollama).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from palace.embeddings.manager import Embedder, FakeEmbedder, OllamaEmbedder
from palace.indexing.indexer import IncrementalIndexer
from palace.indexing.repository import ChunkRepository, LanceDBRepository
from palace.models.config import PalaceConfig
from palace.search.engine import SemanticSearchEngine
from palace.skills.loader import SkillRegistry
from palace.skills.memory_tools import configure_memory_tools
from palace.skills.router import MemoryAgent
from palace.utils.logging import configure_logging
from palace.vault.manager import FileVaultManager


app = typer.Typer(
    name="palace",
    help="MindPalace Tier 2 — local-first semantic memory engine.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
vault_app = typer.Typer(name="vault", help="Manage Core Vault memory lockers.")
app.add_typer(vault_app, name="vault")
console = Console()


# ---- helpers --------------------------------------------------------------


def _load_config(config_path: Path | None) -> PalaceConfig:
    path = config_path or _find_config()
    if not path or not path.exists():
        # Fall back to a default config rooted at the cwd.
        console.print(
            "[dim]No config.yaml found; using defaults rooted at cwd.[/dim]"
        )
        return PalaceConfig.default_for(Path.cwd())
    cfg = PalaceConfig.from_yaml(path)
    configure_logging(
        level=cfg.logging.level,
        rich_output=cfg.logging.rich_output,
        file_path=cfg.logging.file_path,
        force=True,
    )
    return cfg


def _find_config() -> Path | None:
    cwd = Path.cwd()
    candidates = [cwd / "config.yaml", cwd / "MindPalace" / "config.yaml"]
    for c in candidates:
        if c.exists():
            return c
    return None


def _make_embedder(config: PalaceConfig) -> Embedder:
    """Inject the right embedder based on env / config."""
    if os.environ.get("PALACE_EMBEDDER", "").lower() == "fake":
        return FakeEmbedder()
    return OllamaEmbedder(config.embedding)


def _make_repository(config: PalaceConfig) -> ChunkRepository:
    return LanceDBRepository(
        db_path=config.resolve(config.database.path),
        table_name=config.database.table_name,
    )


# ---- commands -------------------------------------------------------------


ConfigOption = Annotated[
    Optional[Path],
    typer.Option("--config", "-c", help="Path to config.yaml"),
]


class VaultLockerType(str, Enum):
    """Supported Core Vault locker scopes."""

    all = "all"
    system = "system"
    user = "user"


@vault_app.command("list")
def list_vault_lockers(
    locker_type: Annotated[
        VaultLockerType,
        typer.Option("--type", help="Locker scope to list."),
    ] = VaultLockerType.all,
    config_path: ConfigOption = None,
) -> None:
    """List active Core Vault lockers."""
    config = _load_config(config_path)
    lockers = FileVaultManager(config).list_lockers(locker_type.value)

    if not lockers:
        console.print("[yellow]No active lockers.[/yellow]")
        return

    table = Table(title=f"Core Vault lockers ({locker_type.value})")
    table.add_column("Locker", style="cyan")
    for locker in lockers:
        table.add_row(locker)
    console.print(table)


@vault_app.command("read")
def read_vault_field(
    locker_name: Annotated[str, typer.Argument(help="Locker name.")],
    field_key: Annotated[str, typer.Argument(help="Field key.")],
    config_path: ConfigOption = None,
) -> None:
    """Read a field from a Core Vault locker."""
    config = _load_config(config_path)
    value = FileVaultManager(config).read_field(locker_name, field_key)
    console.print(value)


@vault_app.command("write")
def write_vault_field(
    locker_name: Annotated[str, typer.Argument(help="Locker name.")],
    field_key: Annotated[str, typer.Argument(help="Field key.")],
    value: Annotated[str, typer.Argument(help="Field value.")],
    config_path: ConfigOption = None,
) -> None:
    """Write a field to a Core Vault locker."""
    config = _load_config(config_path)
    FileVaultManager(config).write_field(locker_name, field_key, value)
    console.print(
        f"[green]Updated[/green] {locker_name}.{field_key}"
    )


@app.command()
def chat(
    config_path: ConfigOption = None,
    skill_names: Annotated[
        Optional[list[str]],
        typer.Option(
            "--skill",
            "-s",
            help="Markdown skill to activate; repeat for multiple skills.",
        ),
    ] = None,
) -> None:
    """Chat with the memory agent."""
    cfg = _load_config(config_path)
    embedder = _make_embedder(cfg)
    repo = _make_repository(cfg)
    vault = FileVaultManager(cfg)
    search_engine = SemanticSearchEngine(cfg, embedder, repo)
    configure_memory_tools(
        vault_manager=vault,
        search_engine=search_engine,
    )
    registry = SkillRegistry()
    try:
        agent = MemoryAgent(
            model="llama3.1",
            registry=registry,
            active_skills=skill_names,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(
        "[dim]Commands: /skills, /skill <name[,name...]>, /skill off, exit[/dim]"
    )

    while True:
        user_input = console.input("[bold green]You:[/bold green]\n")
        command = user_input.strip()

        if not command.startswith("/"):
            response = agent.chat(user_input)
            console.print(f"[bold blue]Palace:[/bold blue] {response}")
            continue

        if command.lower() in {"exit", "quit"}:
            break
        if command == "/skills":
            names = sorted(registry.skills)
            console.print(
                "[cyan]Available skills:[/cyan] " + (", ".join(names) or "none")
            )
            continue
        if command == "/skill":
            active = ", ".join(skill.name for skill in agent.active_skills) or "none"
            console.print(f"[cyan]Active skills:[/cyan] {active}")
            continue
        if command.startswith("/skill "):
            requested_skill = command.removeprefix("/skill ").strip()
            if requested_skill.lower() in {"off", "none"}:
                agent.set_active_skills(None)
                console.print("[yellow]Active skills cleared.[/yellow]")
                continue
            requested_skills = [
                name.strip() for name in requested_skill.split(",") if name.strip()
            ]
            try:
                agent.set_active_skills(requested_skills)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
            else:
                console.print(
                    f"[green]Active skills:[/green] {', '.join(requested_skills)}"
                )
            continue
        console.print("[yellow]Unknown command.[/yellow]")


@app.command()
def index(
    config_path: ConfigOption = None,
    reindex: Annotated[bool, typer.Option("--reindex", help="Force re-index all files")] = False,
) -> None:
    """Index new and changed source files into the vector store."""
    cfg = _load_config(config_path)
    embedder = _make_embedder(cfg)
    repo = _make_repository(cfg)
    indexer = IncrementalIndexer(cfg, embedder, repo)

    if not embedder.health_check() and not isinstance(embedder, FakeEmbedder):
        console.print(
            Panel(
                "[red]Ollama is not reachable.[/red]\n"
                "Start it with [cyan]ollama serve[/cyan] and ensure the "
                "[cyan]nomic-embed-text[/cyan] model is pulled\n"
                "([cyan]ollama pull nomic-embed-text[/cyan]).\n\n"
                "To run without Ollama (for smoke testing), set "
                "[cyan]PALACE_EMBEDDER=fake[/cyan].",
                title="Embedding backend unavailable",
                border_style="red",
            )
        )
        raise typer.Exit(code=2)

    console.print(f"[bold]Indexing[/bold] from {', '.join(cfg.indexer.source_dirs)} ...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description="Indexing files...", total=None)
        result = indexer.reindex_all() if reindex else indexer.index()

    table = Table(title="Index result", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Files scanned", str(result.scanned))
    table.add_row("Files indexed", str(result.indexed))
    table.add_row("Files skipped", str(result.skipped))
    table.add_row("Files deleted", str(result.deleted))
    table.add_row("Chunks created", str(result.chunks_created))
    table.add_row("Errors", str(len(result.errors)))
    table.add_row("Duration", f"{result.duration_seconds:.2f}s")
    console.print(table)

    for err in result.errors[:10]:
        console.print(f"  [red]error:[/red] {err}")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural language query")],
    config_path: ConfigOption = None,
    domain: Annotated[Optional[str], typer.Option("--domain", "-d")] = None,
    date_from: Annotated[Optional[str], typer.Option("--from")] = None,
    date_to: Annotated[Optional[str], typer.Option("--to")] = None,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 8,
    minimum_score: Annotated[float, typer.Option("--min-score")] = 0.15,
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
) -> None:
    """Semantic search across journals, notes, and transcripts."""
    cfg = _load_config(config_path)
    embedder = _make_embedder(cfg)
    repo = _make_repository(cfg)
    engine = SemanticSearchEngine(cfg, embedder, repo)

    outcome = engine.search_with_diagnostics(
        query=query,
        domain=domain,
        date_from=date_from,
        date_to=date_to,
        tags=tag,
        top_k=top_k,
        minimum_score=minimum_score,
    )

    if json_out:
        import json
        console.print_json(json.dumps([r.model_dump(mode="json") for r in outcome.results]))
        return

    if not outcome.results:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(title=f"Search: {query}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", style="green")
    table.add_column("Date", style="cyan")
    table.add_column("Domain", style="magenta")
    table.add_column("Source", style="dim")
    table.add_column("Excerpt", style="white")
    for i, r in enumerate(outcome.results, 1):
        excerpt = r.content[:120].replace("\n", " ") + ("…" if len(r.content) > 120 else "")
        table.add_row(
            str(i),
            f"{r.score:.3f}",
            r.date.isoformat() if r.date else "—",
            r.domain.value,
            Path(r.source_file).name,
            excerpt,
        )
    console.print(table)
    console.print(
        f"[dim]{len(outcome.results)} results · "
        f"{outcome.duration_ms:.0f}ms · "
        f"{outcome.candidate_count} candidates scanned[/dim]"
    )


@app.command()
def stats(
    config_path: ConfigOption = None,
) -> None:
    """Show database statistics."""
    cfg = _load_config(config_path)
    repo = _make_repository(cfg)
    s = repo.statistics()
    table = Table(title="MindPalace index statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total chunks", str(s.get("total_chunks", 0)))
    table.add_row("Unique source files", str(s.get("unique_source_files", 0)))
    table.add_row("Earliest date", str(s.get("date_range_earliest") or "—"))
    table.add_row("Latest date", str(s.get("date_range_latest") or "—"))
    table.add_row("DB path", s.get("db_path", "—"))
    by_domain = s.get("by_domain", {})
    if by_domain:
        console.print(table)
        dt = Table(title="By domain")
        dt.add_column("Domain", style="magenta")
        dt.add_column("Chunks", justify="right")
        for d, n in sorted(by_domain.items()):
            dt.add_row(str(d), str(n))
        console.print(dt)
    else:
        console.print(table)


@app.command()
def verify(
    config_path: ConfigOption = None,
) -> None:
    """Verify the index against the hash cache."""
    cfg = _load_config(config_path)
    embedder = _make_embedder(cfg)
    repo = _make_repository(cfg)
    from palace.indexing.hash_cache import HashCache
    cache = HashCache(cfg.resolve(cfg.indexer.hash_cache_path))
    cache.load()
    repo_stats = repo.statistics()
    cached_files = cache.all_keys()
    console.print(
        f"Cached files: {len(cached_files)} · "
        f"Repository chunks: {repo_stats.get('total_chunks', 0)}"
    )
    # Check each cached file still exists and hash matches.
    from palace.utils.hashing import hash_file
    missing = 0
    changed = 0
    for key in cached_files:
        p = Path(key)
        if not p.exists():
            missing += 1
            continue
        try:
            current = hash_file(p)
            entry = cache.get(key)
            if entry and entry.content_hash != current:
                changed += 1
        except OSError:
            missing += 1
    table = Table(title="Verify result")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="white")
    table.add_row("Cached files", str(len(cached_files)))
    table.add_row("Missing on disk", str(missing))
    table.add_row("Changed on disk", str(changed))
    table.add_row("Repository chunks", str(repo_stats.get("total_chunks", 0)))
    status = "[green]OK[/green]" if missing == 0 and changed == 0 else "[yellow]drift[/yellow]"
    table.add_row("Overall", status)
    console.print(table)


@app.command()
def doctor(
    config_path: ConfigOption = None,
) -> None:
    """Diagnose the MindPalace environment."""
    cfg = _load_config(config_path)

    table = Table(title="Environment diagnostics")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Detail", style="dim")

    # Python version
    table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", sys.executable)

    # Config
    cfg_path = _find_config()
    table.add_row(
        "config.yaml",
        "[green]found[/green]" if cfg_path else "[red]missing[/red]",
        str(cfg_path or "—"),
    )

    # Source dirs
    for d in cfg.indexer.source_dirs:
        p = cfg.resolve(d)
        table.add_row(
            f"source: {d}",
            "[green]ok[/green]" if p.exists() else "[red]missing[/red]",
            str(p),
        )

    # DB path
    db = cfg.resolve(cfg.database.path)
    table.add_row("database path", "[green]exists[/green]" if db.parent.exists() else "[yellow]will create[/yellow]", str(db))

    # LanceDB import
    try:
        import lancedb  # noqa: F401
        table.add_row("lancedb", "[green]installed[/green]", "")
    except ImportError:
        table.add_row("lancedb", "[red]not installed[/red]", "pip install lancedb")

    # Ollama
    import httpx
    try:
        resp = httpx.get(f"{cfg.embedding.host.rstrip('/')}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            has_model = any(cfg.embedding.model in m for m in models)
            status = "[green]ready[/green]" if has_model else f"[yellow]model {cfg.embedding.model} not pulled[/yellow]"
            table.add_row("ollama", status, f"available: {len(models)} models")
        else:
            table.add_row("ollama", f"[red]HTTP {resp.status_code}[/red]", "")
    except Exception:
        table.add_row("ollama", "[red]not reachable[/red]",
                      f"start with 'ollama serve' at {cfg.embedding.host}")

    console.print(table)


@app.command()
def inspect(
    chunk_id: Annotated[Optional[str], typer.Argument(help="Inspect a specific chunk by ID")] = None,
    config_path: ConfigOption = None,
    by_source: Annotated[Optional[str], typer.Option("--source", help="Filter by source file")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
) -> None:
    """Inspect raw chunks in the index."""
    cfg = _load_config(config_path)
    repo = _make_repository(cfg)
    if chunk_id:
        rows = repo.filter_by_metadata(chunk_id=chunk_id)
    elif by_source:
        rows = repo.filter_by_metadata(source_file=by_source)
    else:
        rows = repo.filter_by_metadata()
    if not rows:
        console.print("[yellow]No chunks found.[/yellow]")
        return
    table = Table(title=f"Inspecting {len(rows)} chunk(s)", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Chunk ID", style="cyan")
    table.add_column("Source", style="dim")
    table.add_column("Date", style="cyan")
    table.add_column("Words", justify="right")
    table.add_column("Excerpt", style="white")
    for i, row in enumerate(rows[:limit], 1):
        content = row.get("content", "")
        excerpt = content[:100].replace("\n", " ") + ("…" if len(content) > 100 else "")
        table.add_row(
            str(i),
            row.get("chunk_id", "")[:12] + "…",
            Path(row.get("source_file", "")).name,
            str(row.get("date") or "—"),
            str(row.get("word_count", 0)),
            excerpt,
        )
    console.print(table)


@app.command()
def rebuild(
    config_path: ConfigOption = None,
) -> None:
    """Drop the index and rebuild it from scratch."""
    cfg = _load_config(config_path)
    embedder = _make_embedder(cfg)
    repo = _make_repository(cfg)
    try:
        repo.vacuum()
    except Exception:
        pass
    indexer = IncrementalIndexer(cfg, embedder, repo)
    result = indexer.reindex_all()
    console.print(
        f"[green]Rebuilt[/green]: {result.indexed} files, "
        f"{result.chunks_created} chunks in {result.duration_seconds:.2f}s"
    )


if __name__ == "__main__":
    app()
