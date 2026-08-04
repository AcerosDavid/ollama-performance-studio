"""
Command-line interface for ollama-performance-studio.

Built with Typer. Exposes commands: run, report, dashboard, list, compare.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table

from ollama_benchmark.benchmark_runner import BenchmarkRunner
from ollama_benchmark.config import load_config
from ollama_benchmark.database import Database
from ollama_benchmark.dashboard import start_dashboard
from ollama_benchmark.models import ConfigNotFoundError, ConfigValidationError, OllamaUnavailableError
from ollama_benchmark.report_generator import ReportGenerator

app = typer.Typer(
    help="ollama-performance-studio — benchmark and compare local LLM models",
    add_completion=False,
)
console = Console()
logger = logging.getLogger(__name__)

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    console.print("\n[yellow]  Shutdown requested — finishing current model...[/yellow]")


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Logging — file only, console stays completely clean
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool, log_file: Path) -> None:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(ch)

    for noisy in ("httpx", "httpcore", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Worker thread that runs the benchmark and pushes events into a queue.
# Events: ("phase"|"dl"|"prompt"|"done"|"finished", payload_dict)
# ---------------------------------------------------------------------------

def _run_worker(
    runner: BenchmarkRunner,
    prompt_count: int,
    evt_q: "queue.Queue",
    session_result_holder: list,
    worker_error_holder: list,
) -> None:

    # Track which model each ModelManager belongs to
    _mm_to_model: dict = {}

    # Wire pull-progress callback: called from inside mm.pull() on every chunk
    def _on_mm_created(mm) -> None:
        # Figure out which model this MM is for by inspecting the call stack
        # Actually, we can't reliably do that. Instead, we patch _pull_model
        # to set a thread-local variable with the model name before creating MM.
        import threading
        tls = threading.current_thread()
        model_name = getattr(tls, "_kiro_current_model", "")
        _mm_to_model[id(mm)] = model_name

        def _pull_progress_cb(completed: int, total: int, status: str) -> None:
            evt_q.put(("dl", {
                "model":     model_name,
                "completed": completed,
                "total":     total,
                "status":    status,
            }))
        mm.on_pull_progress = _pull_progress_cb

    runner.on_mm_created = _on_mm_created  # type: ignore[attr-defined]

    # Patch _pull_model to set thread-local model name before calling original
    original_pull = runner._pull_model

    def _tracked_pull(model_name: str):
        import threading
        tls = threading.current_thread()
        tls._kiro_current_model = model_name  # type: ignore[attr-defined]
        return original_pull(model_name)

    runner._pull_model = _tracked_pull  # type: ignore[method-assign]

    _current_model: list = [""]

    # Phase callback
    def _phase_cb(model: str, phase: str) -> None:
        _current_model[0] = model
        evt_q.put(("phase", {"model": model, "phase": phase}))
    runner._phase_callback = _phase_cb  # type: ignore[attr-defined]

    # Prompt callback
    def _prompt_cb(model: str, idx: int, category: str) -> None:
        evt_q.put(("prompt", {"model": model, "idx": idx, "category": category}))
    runner._prompt_callback = _prompt_cb  # type: ignore[attr-defined]

    original_evaluate = runner._evaluate_model

    def _patched_evaluate(model_name: str, session_id: int, prefetched_pull=None):
        _current_model[0] = model_name
        result = original_evaluate(model_name, session_id, prefetched_pull=prefetched_pull)
        evt_q.put(("done", {
            "model":        model_name,
            "status":       result.status,
            "tps":          result.avg_tps,
            "quality":      result.quality_score,
            "download_s":   result.download_time_s,
            "cold_start_s": result.cold_start_s,
        }))
        return result

    runner._evaluate_model = _patched_evaluate  # type: ignore[method-assign]

    try:
        result = runner.run()
        session_result_holder.append(result)
    except OllamaUnavailableError as e:
        worker_error_holder.append(("ollama", str(e)))
    except Exception as e:
        logger.exception("Benchmark worker failed")
        worker_error_holder.append(("error", str(e)))
    finally:
        evt_q.put(("finished", {}))


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@app.command()
def run(
    config: str = typer.Argument("config.yaml", help="Path to YAML configuration file"),
    output: str = typer.Argument(".", help="Directory where the HTML report will be saved"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Write DEBUG-level detail to the log file"),
) -> None:
    """Run a benchmark session and generate an HTML report."""
    config_path = Path(config)
    output_path = Path(output)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"benchmark_{timestamp}.log"
    _setup_logging(verbose, log_file)

    console.print()
    console.print(Panel.fit(
        "[bold cyan]ollama-performance-studio[/bold cyan]\n"
        "[dim]Local LLM Benchmark Suite[/dim]",
        border_style="cyan",
    ))
    console.print()

    try:
        with console.status(f"[cyan]Loading config: [bold]{config_path}[/bold]"):
            benchmark_config = load_config(config_path)

        model_count  = len(benchmark_config.models)
        prompt_count = sum(len(v) for v in benchmark_config.prompts.values())
        cat_count    = len(benchmark_config.prompts)

        cfg = Table.grid(padding=(0, 2))
        cfg.add_column(style="dim")
        cfg.add_column()
        cfg.add_row("Models",
                    f"[bold]{model_count}[/bold]  " +
                    "  ".join(f"[cyan]{m}[/cyan]" for m in benchmark_config.models[:3]) +
                    (" …" if model_count > 3 else ""))
        cfg.add_row("Prompts",  f"[bold]{prompt_count}[/bold] across [bold]{cat_count}[/bold] categories")
        cfg.add_row("Output",   f"[bold]{output_path}[/bold]")
        cfg.add_row("Log file", f"[dim]{log_file}[/dim]")
        console.print(cfg)
        console.print()

        db     = Database(benchmark_config.database_path)
        runner = BenchmarkRunner(benchmark_config, db)

        evt_q: queue.Queue = queue.Queue()
        session_result_holder: list = []
        worker_error_holder:   list = []

        worker = threading.Thread(
            target=_run_worker,
            args=(runner, prompt_count, evt_q, session_result_holder, worker_error_holder),
            daemon=True,
        )

        console.print(Rule("[bold]Benchmark Session[/bold]", style="cyan"))
        console.print()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=24),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
            expand=False,
        )

        task_ids: dict[str, dict] = {}

        with progress:
            overall = progress.add_task(
                f"[bold]Overall[/bold] [dim](0/{model_count})[/dim]",
                total=model_count,
            )
            completed_count = 0
            worker.start()

            while True:
                try:
                    evt_name, payload = evt_q.get(timeout=0.1)
                except queue.Empty:
                    if not worker.is_alive() and evt_q.empty():
                        break
                    continue

                model = payload.get("model", "")

                if model and model not in task_ids and evt_name not in ("finished", "done"):
                    dl_t = progress.add_task(
                        f"  [yellow]↓[/yellow] Downloading  [cyan]{model}[/cyan]",
                        total=100,
                    )
                    ph_t = progress.add_task("  [dim]…[/dim]", total=1, visible=False)
                    pr_t = progress.add_task("  [dim]prompts[/dim]", total=prompt_count, visible=False)
                    task_ids[model] = {"dl": dl_t, "ph": ph_t, "pr": pr_t}

                ids = task_ids.get(model, {})
                if not ids:
                    if evt_name == "finished":
                        break
                    continue

                if evt_name == "phase":
                    phase = payload["phase"]
                    if phase == "pull":
                        progress.update(ids["dl"],
                            description=f"  [yellow]↓[/yellow] Downloading  [cyan]{model}[/cyan]",
                            completed=0, visible=True)
                    elif phase == "queued":
                        progress.update(ids["dl"], completed=100,
                            description=f"  [dim]⏸ Ready       {model}  waiting for evaluation…[/dim]")
                        progress.refresh()
                    elif phase == "start":
                        progress.update(ids["dl"], completed=100,
                            description=f"  [green]✔[/green] Downloaded   [cyan]{model}[/cyan]")
                        progress.update(ids["ph"], visible=True, completed=0,
                            description=f"  [blue]▶[/blue] Starting     [cyan]{model}[/cyan]  [dim](cold start…)[/dim]")
                        progress.refresh()
                    elif phase == "infer":
                        progress.update(ids["ph"], visible=False)
                        progress.update(ids["pr"], visible=True, completed=0,
                            description=f"  [green]◉[/green] Evaluating   [cyan]{model}[/cyan]  [dim]0 / {prompt_count}[/dim]")
                        progress.refresh()
                    elif phase == "score":
                        progress.update(ids["pr"],
                            description=f"  [green]◉[/green] Evaluating   [cyan]{model}[/cyan]  [dim]{prompt_count} / {prompt_count}  ✔[/dim]",
                            completed=prompt_count)
                        progress.update(ids["ph"], visible=True,
                            description=f"  [magenta]★[/magenta] Scoring      [cyan]{model}[/cyan]")
                        progress.refresh()

                elif evt_name == "dl":
                    total_b = payload["total"]
                    done_b  = payload["completed"]
                    status  = payload["status"]
                    if total_b > 0:
                        pct  = min(99, int(done_b / total_b * 100))
                        mb_d = done_b  / 1_048_576
                        mb_t = total_b / 1_048_576
                        progress.update(ids["dl"], completed=pct,
                            description=f"  [yellow]↓[/yellow] Downloading  [cyan]{model}[/cyan]  [dim]{mb_d:.0f} / {mb_t:.0f} MB ({pct}%)[/dim]")
                    elif "verif" in status:
                        progress.update(ids["dl"], completed=99,
                            description=f"  [yellow]⟳[/yellow] Verifying    [cyan]{model}[/cyan]")

                elif evt_name == "prompt":
                    idx = payload["idx"]
                    cat = payload["category"]
                    progress.update(ids["pr"], completed=idx,
                        description=f"  [green]◉[/green] Evaluating   [cyan]{model}[/cyan]  [dim]{cat}  {idx} / {prompt_count}[/dim]")

                elif evt_name == "done":
                    st      = payload["status"]
                    tps     = payload.get("tps")
                    quality = payload.get("quality")
                    dl_s    = payload.get("download_s")
                    cs_s    = payload.get("cold_start_s")
                    icon    = "[green]✔[/green]" if st == "completed" else "[red]✘[/red]"
                    parts   = []
                    if tps:     parts.append(f"{tps:.1f} tps")
                    if quality: parts.append(f"quality {quality:.2f}")
                    if dl_s:    parts.append(f"dl {dl_s:.0f}s")
                    if cs_s:    parts.append(f"start {cs_s:.1f}s")
                    summary = "  [dim]" + "  ·  ".join(parts) + "[/dim]" if parts else ""
                    progress.update(ids["dl"], completed=100,
                        description=f"  {icon} [bold]{model}[/bold]{summary}")
                    for t in ("ph", "pr"):
                        progress.update(ids[t], visible=False)
                    progress.stop_task(ids["dl"])
                    completed_count += 1
                    progress.update(overall, advance=1,
                        description=f"[bold]Overall[/bold] [dim]({completed_count}/{model_count})[/dim]")

                elif evt_name == "finished":
                    break

        worker.join()

        if worker_error_holder:
            kind, msg = worker_error_holder[0]
            if kind == "ollama":
                console.print(f"\n[bold red]✘  Ollama not available:[/bold red] {msg}")
                console.print("[dim]  Start it with: [bold]ollama serve[/bold][/dim]")
            else:
                console.print(f"\n[bold red]✘  Error:[/bold red] {msg}")
            raise typer.Exit(code=1)

        session_result = session_result_holder[0] if session_result_holder else None

        console.print()
        if session_result and session_result.session_id:
            with console.status("[cyan]Generating HTML report…"):
                report_gen  = ReportGenerator(db)
                report_path = report_gen.generate(session_result.session_id, output)
            console.print(f"[green]✔[/green]  Report saved  →  [bold]{report_path}[/bold]")
            console.print()
            _print_summary(session_result, db)
        else:
            console.print("[yellow]  No session was created.[/yellow]")

    except ConfigNotFoundError as e:
        console.print(f"\n[bold red]✘  Config not found:[/bold red] {e}")
        raise typer.Exit(code=1)
    except ConfigValidationError as e:
        console.print(f"\n[bold red]✘  Config error:[/bold red]")
        for field, value in e.errors:
            console.print(f"   [red]• {field}:[/red] {value}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]✘  Error:[/bold red] {e}")
        logger.exception("CLI run command failed")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Other commands
# ---------------------------------------------------------------------------

@app.command()
def report(
    session: int = typer.Option(..., "--session", "-s", help="Session ID to generate report for"),
    output: Path = typer.Option(Path("."), "--output", "-o", help="Directory where the HTML report will be saved"),
    db: Path = typer.Option(Path("benchmark.db"), "--db", help="Path to SQLite database file"),
) -> None:
    """Regenerate an HTML report from previously collected data."""
    try:
        with console.status(f"[cyan]Generating report for session [bold]{session}[/bold]…"):
            database = Database(str(db))
            report_gen = ReportGenerator(database)
            report_path = report_gen.generate(session, output)
        console.print(f"[green]✔[/green]  Report saved  →  [bold]{report_path}[/bold]")
    except Exception as e:
        console.print(f"[bold red]✘  Error:[/bold red] {e}")
        logger.exception("CLI report command failed")
        raise typer.Exit(code=1)


@app.command()
def dashboard(
    port: int = typer.Option(8080, "--port", "-p", help="Port to serve the dashboard on"),
    db: Path = typer.Option(Path("benchmark.db"), "--db", help="Path to SQLite database file"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
) -> None:
    """Launch the interactive web dashboard."""
    try:
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Dashboard[/bold cyan]  →  http://{host}:{port}\n"
            f"[dim]Database: {db}[/dim]\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            border_style="cyan",
        ))
        console.print()
        start_dashboard(str(db), port=port, host=host)
    except KeyboardInterrupt:
        console.print("\n[yellow]  Dashboard stopped.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]✘  Error:[/bold red] {e}")
        logger.exception("CLI dashboard command failed")
        raise typer.Exit(code=1)


@app.command(name="list")
def list_sessions(
    db: Path = typer.Option(Path("benchmark.db"), "--db", help="Path to SQLite database file"),
) -> None:
    """List all benchmark sessions stored in the database."""
    try:
        database = Database(str(db))
        sessions = database.get_sessions()
        if not sessions:
            console.print("[yellow]  No sessions found.[/yellow]")
            return
        table = Table(title="Benchmark Sessions", border_style="dim", header_style="bold cyan")
        table.add_column("ID",       justify="right",  style="bold", width=4)
        table.add_column("Status",   justify="center", width=14)
        table.add_column("Started",  width=20)
        table.add_column("Finished", width=20)
        table.add_column("Models",   justify="right", width=7)
        status_map = {
            "completed":   "[green]✔  completed[/green]",
            "running":     "[yellow]⟳  running[/yellow]",
            "interrupted": "[red]✘  interrupted[/red]",
        }
        for s in sessions:
            table.add_row(
                str(s.id),
                status_map.get(s.status, s.status),
                str(s.started_at)[:19],
                str(s.finished_at)[:19] if s.finished_at else "[dim]—[/dim]",
                str(s.model_count),
            )
        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[bold red]✘  Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def compare(
    session_ids: list[int] = typer.Argument(..., help="Session IDs to compare"),
    db: Path = typer.Option(Path("benchmark.db"), "--db", help="Path to SQLite database file"),
) -> None:
    """Compare two or more sessions side by side."""
    if len(session_ids) < 2:
        console.print("[bold red]✘[/bold red]  At least 2 session IDs are required.")
        raise typer.Exit(code=1)
    try:
        database   = Database(str(db))
        comparison = database.compare_sessions(session_ids)
        table = Table(title=f"Session Comparison  {session_ids}", border_style="dim", header_style="bold cyan")
        table.add_column("Session",      justify="right", width=8)
        table.add_column("Status",       width=14)
        table.add_column("Models",       justify="right", width=7)
        table.add_column("Best Quality", width=30)
        table.add_column("Fastest",      width=30)
        table.add_column("Lightest",     width=30)
        for s in comparison.sessions:
            table.add_row(
                str(s.session_id), s.status, str(s.model_count),
                s.best_quality_model or "[dim]—[/dim]",
                s.fastest_model      or "[dim]—[/dim]",
                s.lowest_ram_model   or "[dim]—[/dim]",
            )
        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[bold red]✘  Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(session_result, db: Database) -> None:
    model_results  = db.get_model_results(session_result.session_id)
    completed      = [m for m in model_results if m.status == "completed"]

    console.print(Rule("[bold]Results[/bold]", style="cyan"))
    console.print()

    if not completed:
        console.print("[yellow]  No models completed successfully.[/yellow]")
        return

    completed_sorted = sorted(completed, key=lambda x: x.overall_rank or 999)

    table = Table(border_style="dim", header_style="bold cyan", show_lines=False)
    table.add_column("Rank",    justify="center", width=6)
    table.add_column("Model",   width=34)
    table.add_column("Quality", justify="right", width=9)
    table.add_column("TPS",     justify="right", width=8)
    table.add_column("TTFT ms", justify="right", width=9)
    table.add_column("RAM MB",  justify="right", width=9)
    table.add_column("Status",  justify="center", width=12)

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for i, m in enumerate(completed_sorted, 1):
        table.add_row(
            medals.get(i, f"  {i}"),
            m.model_name,
            f"{m.quality_score:.3f}" if m.quality_score is not None else "[dim]—[/dim]",
            f"{m.avg_tps:.1f}"       if m.avg_tps       is not None else "[dim]—[/dim]",
            f"{m.avg_ttft_ms:.0f}"   if m.avg_ttft_ms   is not None else "[dim]—[/dim]",
            f"{m.avg_ram_mb:.0f}"    if m.avg_ram_mb    is not None else "[dim]—[/dim]",
            f"[green]{m.status}[/green]" if m.status == "completed" else f"[red]{m.status}[/red]",
        )
    console.print(table)
    console.print()

    recs = db.get_recommendations(session_result.session_id)
    if recs:
        rec_table = Table.grid(padding=(0, 2))
        rec_table.add_column(style="bold magenta", width=18)
        rec_table.add_column(style="bold white",   width=36)
        rec_table.add_column(style="dim")
        icons = {"best_quality": "★  Best quality", "fastest": "⚡  Fastest", "lightweight": "🪶  Lightest"}
        for r in recs:
            rec_table.add_row(icons.get(r.profile, r.profile), r.model_name, r.justification)
        console.print(Panel(rec_table, title="[bold]Recommendations[/bold]", border_style="magenta"))
        console.print()

    console.print(
        f"[dim]Session ID: {session_result.session_id}   •   "
        f"ops report --session {session_result.session_id}[/dim]"
    )
    console.print()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
