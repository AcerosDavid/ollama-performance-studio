"""
Command-line interface for ollama-performance-studio.

Built with Typer. Exposes commands: run, report, dashboard, list, compare.
"""

from __future__ import annotations

import logging
import signal
from datetime import datetime
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from ollama_benchmark.benchmark_runner import BenchmarkRunner
from ollama_benchmark.config import BenchmarkConfig, load_config
from ollama_benchmark.database import Database
from ollama_benchmark.dashboard import start_dashboard
from ollama_benchmark.models import ConfigNotFoundError, ConfigValidationError, OllamaUnavailableError
from ollama_benchmark.report_generator import ReportGenerator

app = typer.Typer(help="Professional benchmark tool for Ollama LLM models")
console = Console()
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    console.print("\n[yellow]Shutdown requested. Completing current model...[/yellow]")


signal.signal(signal.SIGINT, _signal_handler)


@app.command()
def run(
    config: str = typer.Argument("config.yaml", help="Path to YAML configuration file"),
    output: str = typer.Argument(".", help="Directory where the HTML report will be saved"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode to show all commands, prompts, responses, and resource metrics"),
) -> None:
    """Start a benchmark session.

    CONFIG: Path to YAML configuration file (default: config.yaml)
    OUTPUT: Directory where the HTML report will be saved (default: current directory)
    """
    global _shutdown_requested

    config_path = Path(config)
    output_path = Path(output)

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"benchmark_{timestamp}.log"

    # Configure logging with both console and file handlers
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Remove any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(console_handler)

    # File handler with full timestamp
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    root_logger.setLevel(logging.DEBUG)

    if verbose:
        console.print(f"[bold yellow]Verbose mode enabled - showing all commands, prompts, and responses[/bold yellow]\n")
    console.print(f"[bold blue]Logging to file: {log_file}[/bold blue]\n")

    try:
        # Load configuration
        console.print(f"[bold blue]Loading configuration from {config_path}...[/bold blue]")
        benchmark_config = load_config(config_path)

        # Initialize database
        db = Database(benchmark_config.database_path)

        # Create benchmark runner
        runner = BenchmarkRunner(benchmark_config, db)

        # Run benchmark with live progress
        console.print("[bold green]Starting benchmark session...[/bold green]")

        with Live(console=console, refresh_per_second=4) as live:
            # This is a simplified progress display
            # In production, this would show real-time metrics
            live.update(
                Panel(
                    "[bold]Benchmark in progress...[/bold]\n"
                    "Models will be evaluated sequentially.\n"
                    "Press Ctrl+C to gracefully shutdown.",
                    title="Benchmark Progress",
                )
            )

            try:
                session_result = runner.run()
                _shutdown_requested = True  # Mark as complete
            except OllamaUnavailableError as e:
                console.print(f"[red]Error: {e}[/red]")
                console.print("[yellow]Make sure Ollama is running with 'ollama serve'[/yellow]")
                raise typer.Exit(code=1)
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")
                logger.exception("Benchmark failed")
                raise typer.Exit(code=1)

        # Generate report
        if session_result and session_result.session_id:
            console.print(f"[bold green]Benchmark completed! Session ID: {session_result.session_id}[/bold green]")
            console.print("[bold blue]Generating report...[/bold blue]")

            report_gen = ReportGenerator(db)
            report_path = report_gen.generate(session_result.session_id, output)

            console.print(f"[bold green]Report generated: {report_path}[/bold green]")

            # Show summary
            _print_summary(session_result, db)
        else:
            console.print("[yellow]Benchmark completed but no session was created.[/yellow]")

    except ConfigNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise typer.Exit(code=1)
    except ConfigValidationError as e:
        console.print(f"[red]Configuration validation error:[/red]")
        for field, value in e.errors:
            console.print(f"  [red]- {field}: {value}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("CLI run command failed")
        raise typer.Exit(code=1)


@app.command()
def report(
    session: int = typer.Option(..., "--session", "-s", help="Session ID to generate report for"),
    output: Path = typer.Option(
        Path("."),
        "--output",
        "-o",
        help="Directory where the HTML report will be saved",
    ),
    db: Path = typer.Option(
        Path("benchmark.db"),
        "--db",
        help="Path to SQLite database file",
    ),
) -> None:
    """Regenerate an HTML report from previously collected data."""
    try:
        console.print(f"[bold blue]Generating report for session {session}...[/bold blue]")

        database = Database(str(db))
        report_gen = ReportGenerator(database)
        report_path = report_gen.generate(session, output)

        console.print(f"[bold green]Report generated: {report_path}[/bold green]")
    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        logger.exception("CLI report command failed")
        raise typer.Exit(code=1)


@app.command()
def dashboard(
    port: int = typer.Option(8080, "--port", "-p", help="Port to serve the dashboard on"),
    db: Path = typer.Option(
        Path("benchmark.db"),
        "--db",
        help="Path to SQLite database file",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
) -> None:
    """Launch the interactive web dashboard."""
    try:
        console.print(f"[bold blue]Starting dashboard on http://{host}:{port}[/bold blue]")
        console.print(f"[dim]Database: {db}[/dim]")
        console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")

        start_dashboard(str(db), port=port, host=host)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting dashboard: {e}[/red]")
        logger.exception("CLI dashboard command failed")
        raise typer.Exit(code=1)


@app.command()
def list_sessions(
    db: Path = typer.Option(
        Path("benchmark.db"),
        "--db",
        help="Path to SQLite database file",
    ),
) -> None:
    """List all benchmark sessions stored in the database."""
    try:
        database = Database(str(db))
        sessions = database.get_sessions()

        if not sessions:
            console.print("[yellow]No sessions found in database.[/yellow]")
            return

        console.print(f"[bold]Found {len(sessions)} session(s):[/bold]\n")

        for session in sessions:
            status_color = {
                "completed": "green",
                "running": "yellow",
                "interrupted": "red",
            }.get(session.status, "white")

            console.print(
                f"[bold]Session {session.id}[/bold] - "
                f"[{status_color}]{session.status}[/{status_color}]"
            )
            console.print(f"  Started: {session.started_at}")
            if session.finished_at:
                console.print(f"  Finished: {session.finished_at}")
            console.print(f"  Models: {session.model_count}")
            console.print()

    except Exception as e:
        console.print(f"[red]Error listing sessions: {e}[/red]")
        logger.exception("CLI list command failed")
        raise typer.Exit(code=1)


@app.command()
def compare(
    session_ids: list[int] = typer.Argument(..., help="Session IDs to compare"),
    db: Path = typer.Option(
        Path("benchmark.db"),
        "--db",
        help="Path to SQLite database file",
    ),
) -> None:
    """Compare two or more sessions side by side."""
    if len(session_ids) < 2:
        console.print("[red]At least 2 session IDs are required for comparison.[/red]")
        raise typer.Exit(code=1)

    try:
        console.print(f"[bold blue]Comparing sessions: {session_ids}[/bold blue]")

        database = Database(str(db))
        comparison = database.compare_sessions(session_ids)

        # Print comparison summary
        console.print(f"\n[bold]Comparison Summary[/bold]\n")

        for session_comp in comparison.sessions:
            console.print(f"[bold]Session {session_comp.session_id}:[/bold]")
            console.print(f"  Status: {session_comp.status}")
            console.print(f"  Models evaluated: {session_comp.model_count}")
            console.print(f"  Best overall model: {session_comp.best_model}")
            console.print()

        # Print side-by-side metrics
        console.print("[bold]Side-by-Side Metrics[/bold]\n")
        console.print(f"{'Session':<10} {'Best Quality':<20} {'Fastest TPS':<20} {'Lowest RAM':<20}")
        console.print("-" * 70)

        for session_comp in comparison.sessions:
            console.print(
                f"{session_comp.session_id:<10} "
                f"{session_comp.best_quality_model or 'N/A':<20} "
                f"{session_comp.fastest_model or 'N/A':<20} "
                f"{session_comp.lowest_ram_model or 'N/A':<20}"
            )

    except Exception as e:
        console.print(f"[red]Error comparing sessions: {e}[/red]")
        logger.exception("CLI compare command failed")
        raise typer.Exit(code=1)


def _print_summary(session_result, db: Database) -> None:
    """Print a summary of benchmark results."""
    console.print("\n[bold]📊 Benchmark Summary[/bold]\n")

    model_results = db.get_model_results(session_result.session_id)
    completed = [m for m in model_results if m.status == "completed"]

    if not completed:
        console.print("[yellow]No models completed successfully.[/yellow]")
        return

    # Sort by overall rank
    completed_sorted = sorted(
        completed, key=lambda x: x.overall_rank or 0.0, reverse=True
    )

    console.print("[bold]Top 3 Models:[/bold]\n")
    for i, model in enumerate(completed_sorted[:3], 1):
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "  ")
        overall_str = f"{model.overall_rank:.3f}" if model.overall_rank else "N/A"
        quality_str = f"{model.quality_score:.3f}" if model.quality_score else "N/A"
        tps_str = f"{model.avg_tps:.2f}" if model.avg_tps else "N/A"
        ram_str = f"{model.avg_ram_mb:.1f}" if model.avg_ram_mb else "N/A"
        console.print(
            f"{rank_emoji} [bold]{model.model_name}[/bold] - "
            f"Overall: {overall_str}"
        )
        console.print(
            f"    Quality: {quality_str} | "
            f"TPS: {tps_str} | "
            f"RAM: {ram_str} MB"
        )

    # Show primary recommendation
    recommendations = db.get_recommendations(session_result.session_id)
    if recommendations:
        primary = next((r for r in recommendations if "best" in r.profile.lower()), None)
        if primary:
            console.print(f"\n[bold green]💡 Primary Recommendation:[/bold green]")
            console.print(f"  {primary.profile}: {primary.model_name}")
            console.print(f"  {primary.justification}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
