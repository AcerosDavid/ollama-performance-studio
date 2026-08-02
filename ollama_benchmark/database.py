"""
Database layer for ollama-performance-studio.

Implements the SQLite persistence layer using SQLAlchemy Core (not ORM).
The Database class manages schema creation, write operations (task 4.2) and
query operations (task 4.3) — all through a single engine backed by the
SQLite file at the path given to the constructor.

Schema
------
sessions           — one row per benchmark session
model_runs         — one row per (model, session) pair
prompt_results     — one row per prompt execution within a model run
resource_samples   — one row per second of monitoring during a model run
efficiency_indices — one row per model run (computed after all prompts)
quality_scores     — one row per (category, model_run) pair
error_log          — one row per error/warning event
recommendations    — one row per recommendation generated at session end

Usage
-----
    from ollama_benchmark.database import Database

    db = Database("benchmark.db")
    session_id = db.create_session(hardware, config_snapshot)
    ...
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import sqlalchemy as sa

# ---------------------------------------------------------------------------
# DTOs imported for type hints used in task 4.2 / 4.3 write/query methods.
# They are not used in this file directly but are imported here so that
# subsequent tasks can extend this module without adding new import sections.
# ---------------------------------------------------------------------------
from ollama_benchmark.models import (  # noqa: F401  (used in tasks 4.2 / 4.3)
    HardwareInfo,
    ResourceSample,
    ResourceSummary,
    InferenceResult,
    ModelResult,
    RobustnessMetrics,
    EfficiencyIndices,
    ErrorEntry,
    SessionSummary,
    SessionDetail,
    PromptResult,
    Recommendation,
    ComparisonReport,
    PullResult,
)

__all__ = ["Database"]


class Database:
    """
    SQLAlchemy Core database interface for ollama-benchmark.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file.  Pass ``":memory:"``
        for a fully in-memory database (useful in tests).

    On construction the engine is created, all tables are defined in a
    shared ``MetaData`` object, and ``metadata.create_all()`` is called so
    that the schema is present on the first connection (idempotent — tables
    that already exist are not recreated).
    """

    def __init__(self, db_path: str) -> None:
        # ------------------------------------------------------------------
        # Engine
        # ------------------------------------------------------------------
        self.engine: sa.Engine = sa.create_engine(
            f"sqlite:///{db_path}",
            # Keep connections alive across threads (needed by ResourceMonitor
            # background thread); SQLite's check_same_thread guard is replaced
            # by the StaticPool / connection-level locking strategy.
            connect_args={"check_same_thread": False},
        )

        # ------------------------------------------------------------------
        # Metadata + table definitions
        # ------------------------------------------------------------------
        self.metadata: sa.MetaData = sa.MetaData()

        self.sessions: sa.Table = sa.Table(
            "sessions",
            self.metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("started_at", sa.Text, nullable=False),
            sa.Column("finished_at", sa.Text, nullable=True),
            sa.Column("status", sa.Text, nullable=False, server_default="running"),
            sa.Column("config_json", sa.Text, nullable=False),
            sa.Column("hw_os", sa.Text, nullable=True),
            sa.Column("hw_python", sa.Text, nullable=True),
            sa.Column("hw_ollama", sa.Text, nullable=True),
            sa.Column("hw_cpu_cores", sa.Integer, nullable=True),
            sa.Column("hw_cpu_mhz", sa.Float, nullable=True),
            sa.Column("hw_ram_mb", sa.Float, nullable=True),
            sa.Column("hw_gpu", sa.Integer, nullable=True),   # boolean 0/1
            sa.Column("hw_vram_mb", sa.Float, nullable=True),
            sa.Column("hw_disk_free_mb", sa.Float, nullable=True),
        )

        self.model_runs: sa.Table = sa.Table(
            "model_runs",
            self.metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "session_id",
                sa.Integer,
                sa.ForeignKey("sessions.id"),
                nullable=False,
            ),
            sa.Column("model_name", sa.Text, nullable=False),
            sa.Column("status", sa.Text, nullable=False),
            sa.Column("download_time_s", sa.Float, nullable=True),
            sa.Column("model_size_gb", sa.Float, nullable=True),
            sa.Column("cold_start_s", sa.Float, nullable=True),
            sa.Column("avg_ttft_ms", sa.Float, nullable=True),
            sa.Column("avg_latency_ms", sa.Float, nullable=True),
            sa.Column("avg_tps", sa.Float, nullable=True),
            sa.Column("avg_inter_token_ms", sa.Float, nullable=True),
            sa.Column("quality_score", sa.Float, nullable=True),
            sa.Column("stability_score", sa.Float, nullable=True),
            sa.Column("overall_rank", sa.Float, nullable=True),
            sa.Column("error_restarts", sa.Integer, server_default="0"),
            sa.Column("total_timeouts", sa.Integer, server_default="0"),
            sa.Column("total_oom", sa.Integer, server_default="0"),
            sa.Column("total_errors", sa.Integer, server_default="0"),
            sa.Column("started_at", sa.Text, nullable=True),
            sa.Column("finished_at", sa.Text, nullable=True),
        )

        self.prompt_results: sa.Table = sa.Table(
            "prompt_results",
            self.metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "model_run_id",
                sa.Integer,
                sa.ForeignKey("model_runs.id"),
                nullable=False,
            ),
            sa.Column("category", sa.Text, nullable=False),
            sa.Column("prompt_text", sa.Text, nullable=False),
            sa.Column("response_text", sa.Text, nullable=True),
            sa.Column("ttft_ms", sa.Float, nullable=True),
            sa.Column("total_ms", sa.Float, nullable=True),
            sa.Column("tokens_generated", sa.Integer, nullable=True),
            sa.Column("tokens_per_second", sa.Float, nullable=True),
            sa.Column("avg_inter_token_ms", sa.Float, nullable=True),
            sa.Column("quality_score", sa.Float, nullable=True),
            sa.Column("timed_out", sa.Integer, server_default="0"),  # boolean 0/1
            sa.Column("error", sa.Text, nullable=True),
        )

        self.resource_samples: sa.Table = sa.Table(
            "resource_samples",
            self.metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "model_run_id",
                sa.Integer,
                sa.ForeignKey("model_runs.id"),
                nullable=False,
            ),
            sa.Column("sampled_at", sa.Text, nullable=False),
            sa.Column("cpu_percent", sa.Float, nullable=True),
            sa.Column("ram_mb", sa.Float, nullable=True),
            sa.Column("gpu_percent", sa.Float, nullable=True),
            sa.Column("vram_mb", sa.Float, nullable=True),
            sa.Column("cpu_temp_c", sa.Float, nullable=True),
            sa.Column("gpu_temp_c", sa.Float, nullable=True),
            sa.Column("power_watts", sa.Float, nullable=True),
            # Composite index added below via Index()
        )

        # Index on resource_samples(model_run_id, sampled_at) for time-series queries.
        sa.Index(
            "idx_resource_samples_run",
            self.resource_samples.c.model_run_id,
            self.resource_samples.c.sampled_at,
        )

        self.efficiency_indices: sa.Table = sa.Table(
            "efficiency_indices",
            self.metadata,
            # PRIMARY KEY is also a FK — one row per model_run
            sa.Column(
                "model_run_id",
                sa.Integer,
                sa.ForeignKey("model_runs.id"),
                primary_key=True,
            ),
            sa.Column("quality_per_ram", sa.Float, nullable=True),
            sa.Column("quality_per_latency", sa.Float, nullable=True),
            sa.Column("quality_per_cpu", sa.Float, nullable=True),
            sa.Column("quality_per_disk", sa.Float, nullable=True),
            sa.Column("tps_per_gb_ram", sa.Float, nullable=True),
            sa.Column("quality_per_energy", sa.Float, nullable=True),
            sa.Column("norm_quality_per_ram", sa.Float, nullable=True),
            sa.Column("norm_quality_per_latency", sa.Float, nullable=True),
            sa.Column("norm_quality_per_cpu", sa.Float, nullable=True),
            sa.Column("norm_quality_per_disk", sa.Float, nullable=True),
            sa.Column("norm_tps_per_gb_ram", sa.Float, nullable=True),
            sa.Column("norm_quality_per_energy", sa.Float, nullable=True),
        )

        self.quality_scores: sa.Table = sa.Table(
            "quality_scores",
            self.metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "model_run_id",
                sa.Integer,
                sa.ForeignKey("model_runs.id"),
                nullable=False,
            ),
            sa.Column("category", sa.Text, nullable=False),
            sa.Column("score", sa.Float, nullable=True),
        )

        self.error_log: sa.Table = sa.Table(
            "error_log",
            self.metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "model_run_id",
                sa.Integer,
                sa.ForeignKey("model_runs.id"),
                nullable=True,
            ),
            sa.Column(
                "session_id",
                sa.Integer,
                sa.ForeignKey("sessions.id"),
                nullable=True,
            ),
            sa.Column("error_type", sa.Text, nullable=False),
            sa.Column("message", sa.Text, nullable=True),
            sa.Column("logged_at", sa.Text, nullable=False),
        )

        self.recommendations: sa.Table = sa.Table(
            "recommendations",
            self.metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "session_id",
                sa.Integer,
                sa.ForeignKey("sessions.id"),
                nullable=False,
            ),
            sa.Column("profile", sa.Text, nullable=False),
            sa.Column("model_name", sa.Text, nullable=False),
            sa.Column("justification", sa.Text, nullable=False),
        )

        # ------------------------------------------------------------------
        # Create all tables (idempotent — existing tables are skipped).
        # ------------------------------------------------------------------
        self.metadata.create_all(self.engine)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_with_retry(self, stmt: sa.sql.ClauseElement, values: dict) -> Optional[sa.engine.CursorResult]:
        """Execute a write statement. Retries once on failure; logs and continues on second failure."""
        _log = logging.getLogger(__name__)
        for attempt in range(2):
            try:
                with self.engine.begin() as conn:
                    result = conn.execute(stmt, values)
                    return result
            except Exception as exc:
                if attempt == 0:
                    # retry once
                    continue
                # second failure: log and return None
                _log.error("DB write failed after retry: %s", exc)
                return None
        return None  # unreachable, satisfies type checker

    def _execute_with_retry_n(
        self,
        stmt: sa.sql.ClauseElement,
        values: dict,
        retries: int = 3,
    ) -> Optional[sa.engine.CursorResult]:
        """Execute a write statement with up to *retries* attempts; logs and returns None on final failure."""
        _log = logging.getLogger(__name__)
        for attempt in range(retries):
            try:
                with self.engine.begin() as conn:
                    result = conn.execute(stmt, values)
                    return result
            except Exception as exc:
                if attempt < retries - 1:
                    continue
                _log.error("DB write failed after %d retries: %s", retries, exc)
                return None
        return None  # unreachable

    # ------------------------------------------------------------------
    # Write interface — implemented in task 4.2
    # ------------------------------------------------------------------

    def create_session(self, hardware: HardwareInfo, config_snapshot: dict) -> int:
        """Insert a new session row and return the new session id."""
        stmt = self.sessions.insert()
        values = {
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": None,
            "status": "running",
            "config_json": json.dumps(config_snapshot),
            "hw_os": hardware.os,
            "hw_python": hardware.python_version,
            "hw_ollama": hardware.ollama_version,
            "hw_cpu_cores": hardware.cpu_cores,
            "hw_cpu_mhz": hardware.cpu_mhz,
            "hw_ram_mb": hardware.ram_mb,
            "hw_gpu": 1 if hardware.has_gpu else 0,
            "hw_vram_mb": hardware.vram_mb,
            "hw_disk_free_mb": hardware.disk_free_mb,
        }
        result = self._execute_with_retry(stmt, values)
        if result is not None:
            return result.inserted_primary_key[0]
        raise RuntimeError("Failed to create session after retry")

    def save_resource_sample(self, model_run_id: int, sample: ResourceSample) -> None:
        """Persist one resource sample. Retries up to 3 times on failure; logs and discards on final failure."""
        cpu_values = [v for v in sample.cpu_per_core if v is not None]
        cpu_percent = sum(cpu_values) / len(cpu_values) if cpu_values else None

        stmt = self.resource_samples.insert()
        values = {
            "model_run_id": model_run_id,
            "sampled_at": sample.timestamp.isoformat(),
            "cpu_percent": cpu_percent,
            "ram_mb": sample.ram_mb,
            "gpu_percent": sample.gpu_percent,
            "vram_mb": sample.vram_mb,
            "cpu_temp_c": sample.cpu_temp_c,
            "gpu_temp_c": sample.gpu_temp_c,
            "power_watts": sample.power_watts,
        }
        self._execute_with_retry_n(stmt, values, retries=3)

    def save_inference_result(self, model_run_id: int, result: InferenceResult, category: str) -> None:
        """Persist one prompt inference result to the prompt_results table."""
        stmt = self.prompt_results.insert()
        values = {
            "model_run_id": model_run_id,
            "category": category,
            "prompt_text": result.prompt_text,
            "response_text": result.response_text,
            "ttft_ms": result.ttft_ms,
            "total_ms": result.total_response_ms,
            "tokens_generated": result.tokens_generated,
            "tokens_per_second": result.tokens_per_second,
            "avg_inter_token_ms": result.avg_inter_token_ms,
            "quality_score": None,
            "timed_out": 1 if result.timed_out else 0,
            "error": result.error,
        }
        self._execute_with_retry(stmt, values)

    def update_prompt_quality_score(
        self, model_run_id: int, prompt_text: str, score: Optional[float]
    ) -> None:
        """Update the quality_score for a persisted prompt result."""
        stmt = (
            self.prompt_results.update()
            .where(self.prompt_results.c.model_run_id == model_run_id)
            .where(self.prompt_results.c.prompt_text == prompt_text)
            .values(quality_score=score)
        )
        self._execute_with_retry(stmt, {})

    def _find_model_run_id(self, session_id: int, model_name: str) -> Optional[int]:
        """Return the latest model_run id for a session/model pair, if any."""
        with self.engine.connect() as conn:
            row = conn.execute(
                sa.select(self.model_runs.c.id)
                .where(self.model_runs.c.session_id == session_id)
                .where(self.model_runs.c.model_name == model_name)
                .order_by(self.model_runs.c.id.desc())
                .limit(1)
            ).fetchone()
        return row[0] if row else None

    def _compute_model_run_values(
        self, session_id: int, result: ModelResult, *, is_new: bool
    ) -> dict:
        """Build column values for inserting or updating a model_runs row."""
        ttft_values = [r.ttft_ms for r in result.inference_results if r.ttft_ms is not None]
        tps_values = [
            r.tokens_per_second for r in result.inference_results if r.tokens_per_second is not None
        ]
        inter_token_values = [
            r.avg_inter_token_ms for r in result.inference_results if r.avg_inter_token_ms is not None
        ]
        latency_values = [
            r.total_response_ms for r in result.inference_results if r.total_response_ms is not None
        ]

        avg_ttft_ms = sum(ttft_values) / len(ttft_values) if ttft_values else None
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else None
        avg_inter_token_ms = (
            sum(inter_token_values) / len(inter_token_values) if inter_token_values else None
        )
        avg_latency_ms = sum(latency_values) / len(latency_values) if latency_values else 0

        quality_values = [v for v in result.quality_scores.values() if v is not None]
        quality_score = sum(quality_values) / len(quality_values) if quality_values else None

        rob = result.robustness
        now = datetime.utcnow().isoformat()

        values = {
            "status": result.status,
            "download_time_s": result.download_time_s,
            "model_size_gb": result.model_size_gb,
            "cold_start_s": result.cold_start_s,
            "avg_ttft_ms": avg_ttft_ms,
            "avg_latency_ms": avg_latency_ms,
            "avg_tps": avg_tps,
            "avg_inter_token_ms": avg_inter_token_ms,
            "quality_score": quality_score,
            "stability_score": rob.stability_score,
            "overall_rank": result.overall_rank,
            "error_restarts": rob.restart_count,
            "total_timeouts": rob.total_timeouts,
            "total_oom": rob.oom_count,
            "total_errors": rob.total_errors,
            "finished_at": now,
        }
        if is_new:
            values.update({
                "session_id": session_id,
                "model_name": result.model_name,
                "started_at": now,
            })
        return values

    def save_model_result(
        self,
        session_id: int,
        result: ModelResult,
        model_run_id: Optional[int] = None,
    ) -> int:
        """Insert or update a model_runs row. Returns the model_run_id."""
        if model_run_id is None:
            model_run_id = self._find_model_run_id(session_id, result.model_name)

        is_new = model_run_id is None
        values = self._compute_model_run_values(session_id, result, is_new=is_new)

        if is_new:
            stmt = self.model_runs.insert()
            db_result = self._execute_with_retry(stmt, values)
            if db_result is None:
                raise RuntimeError(f"Failed to save model result for {result.model_name} after retry")
            model_run_id = db_result.inserted_primary_key[0]
        else:
            stmt = (
                self.model_runs.update()
                .where(self.model_runs.c.id == model_run_id)
                .values(**values)
            )
            if self._execute_with_retry(stmt, {}) is None:
                raise RuntimeError(f"Failed to update model result for {result.model_name} after retry")

        if result.quality_scores:
            self._replace_quality_scores(model_run_id, result.quality_scores)

        if result.efficiency_indices is not None:
            self.save_efficiency_indices(model_run_id, result.efficiency_indices)

        return model_run_id

    def save_quality_scores(self, model_run_id: int, scores: dict[str, Optional[float]]) -> None:
        """Insert quality_scores rows for each category."""
        for category, score in scores.items():
            stmt = self.quality_scores.insert()
            values = {
                "model_run_id": model_run_id,
                "category": category,
                "score": score,
            }
            self._execute_with_retry(stmt, values)

    def _replace_quality_scores(
        self, model_run_id: int, scores: dict[str, Optional[float]]
    ) -> None:
        """Replace all category quality scores for a model run."""
        with self.engine.begin() as conn:
            conn.execute(
                sa.delete(self.quality_scores).where(
                    self.quality_scores.c.model_run_id == model_run_id
                )
            )
        self.save_quality_scores(model_run_id, scores)

    def save_efficiency_indices(self, model_run_id: int, indices: EfficiencyIndices) -> None:
        """Insert or update the efficiency_indices row for this model run."""
        # Use INSERT OR REPLACE (SQLite upsert) so re-running after normalization
        # updates the normalized columns without creating a duplicate row.
        stmt = sa.text(
            """
            INSERT OR REPLACE INTO efficiency_indices (
                model_run_id,
                quality_per_ram, quality_per_latency, quality_per_cpu,
                quality_per_disk, tps_per_gb_ram, quality_per_energy,
                norm_quality_per_ram, norm_quality_per_latency, norm_quality_per_cpu,
                norm_quality_per_disk, norm_tps_per_gb_ram, norm_quality_per_energy
            ) VALUES (
                :model_run_id,
                :quality_per_ram, :quality_per_latency, :quality_per_cpu,
                :quality_per_disk, :tps_per_gb_ram, :quality_per_energy,
                :norm_quality_per_ram, :norm_quality_per_latency, :norm_quality_per_cpu,
                :norm_quality_per_disk, :norm_tps_per_gb_ram, :norm_quality_per_energy
            )
            """
        )
        values = {
            "model_run_id": model_run_id,
            "quality_per_ram": indices.quality_per_ram,
            "quality_per_latency": indices.quality_per_latency,
            "quality_per_cpu": indices.quality_per_cpu,
            "quality_per_disk": indices.quality_per_disk,
            "tps_per_gb_ram": indices.tps_per_gb_ram,
            "quality_per_energy": indices.quality_per_energy,
            "norm_quality_per_ram": indices.norm_quality_per_ram,
            "norm_quality_per_latency": indices.norm_quality_per_latency,
            "norm_quality_per_cpu": indices.norm_quality_per_cpu,
            "norm_quality_per_disk": indices.norm_quality_per_disk,
            "norm_tps_per_gb_ram": indices.norm_tps_per_gb_ram,
            "norm_quality_per_energy": indices.norm_quality_per_energy,
        }
        self._execute_with_retry(stmt, values)

    def save_recommendation(self, session_id: int, rec: Recommendation) -> None:
        """Insert a recommendation row."""
        stmt = self.recommendations.insert()
        values = {
            "session_id": session_id,
            "profile": rec.profile,
            "model_name": rec.model_name,
            "justification": rec.justification,
        }
        self._execute_with_retry(stmt, values)

    def finalize_session(self, session_id: int, status: str = "completed") -> None:
        """Set session status and finished_at timestamp."""
        stmt = (
            self.sessions.update()
            .where(self.sessions.c.id == sa.bindparam("_session_id"))
            .values(
                status=sa.bindparam("_status"),
                finished_at=sa.bindparam("_finished_at"),
            )
        )
        values = {
            "_session_id": session_id,
            "_status": status,
            "_finished_at": datetime.utcnow().isoformat(),
        }
        self._execute_with_retry(stmt, values)

    def get_recommendations(self, session_id: int) -> list[Recommendation]:
        """Return all recommendations generated for a session."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(self.recommendations)
                .where(self.recommendations.c.session_id == session_id)
                .order_by(self.recommendations.c.id)
            ).fetchall()

        return [
            Recommendation(
                profile=row.profile,
                model_name=row.model_name,
                justification=row.justification,
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Query interface — implemented in task 4.3
    # ------------------------------------------------------------------

    def get_sessions(self) -> list[SessionSummary]:
        """Return all sessions ordered by started_at desc."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(self.sessions).order_by(self.sessions.c.started_at.desc())
            ).fetchall()

            summaries: list[SessionSummary] = []
            for row in rows:
                # Count model_runs for this session
                count_result = conn.execute(
                    sa.select(sa.func.count())
                    .select_from(self.model_runs)
                    .where(self.model_runs.c.session_id == row.id)
                ).scalar()
                model_count = count_result or 0

                started_at = datetime.fromisoformat(row.started_at)
                finished_at = (
                    datetime.fromisoformat(row.finished_at)
                    if row.finished_at
                    else None
                )

                summaries.append(
                    SessionSummary(
                        id=row.id,
                        started_at=started_at,
                        finished_at=finished_at,
                        status=row.status,
                        model_count=model_count,
                    )
                )

        return summaries

    def get_session_detail(self, session_id: int) -> SessionDetail:
        """Return full session including hardware, config snapshot, and model results."""
        with self.engine.connect() as conn:
            row = conn.execute(
                sa.select(self.sessions).where(self.sessions.c.id == session_id)
            ).fetchone()

        if row is None:
            raise ValueError(f"Session {session_id} not found")

        # Reconstruct HardwareInfo from hw_* columns
        hardware = HardwareInfo(
            os=row.hw_os or "",
            python_version=row.hw_python or "",
            ollama_version=row.hw_ollama or "",
            cpu_cores=row.hw_cpu_cores or 0,
            cpu_mhz=row.hw_cpu_mhz or 0.0,
            ram_mb=row.hw_ram_mb or 0.0,
            has_gpu=bool(row.hw_gpu),
            vram_mb=row.hw_vram_mb or 0.0,
            disk_free_mb=row.hw_disk_free_mb or 0.0,
        )

        # Parse config snapshot from JSON
        config_snapshot: dict = json.loads(row.config_json) if row.config_json else {}

        # Build summary
        started_at = datetime.fromisoformat(row.started_at)
        finished_at = (
            datetime.fromisoformat(row.finished_at) if row.finished_at else None
        )
        with self.engine.connect() as conn:
            model_count = conn.execute(
                sa.select(sa.func.count())
                .select_from(self.model_runs)
                .where(self.model_runs.c.session_id == session_id)
            ).scalar() or 0

        summary = SessionSummary(
            id=row.id,
            started_at=started_at,
            finished_at=finished_at,
            status=row.status,
            model_count=model_count,
        )

        model_results = self.get_model_results(session_id)

        return SessionDetail(
            summary=summary,
            hardware=hardware,
            config_snapshot=config_snapshot,
            model_results=model_results,
        )

    def get_model_results(self, session_id: int) -> list[ModelResult]:
        """Return all ModelResult DTOs for a session."""
        with self.engine.connect() as conn:
            runs = conn.execute(
                sa.select(self.model_runs).where(
                    self.model_runs.c.session_id == session_id
                )
            ).fetchall()

            results: list[ModelResult] = []
            for run in runs:
                model_run_id = run.id

                # Fetch quality scores: category -> score
                qs_rows = conn.execute(
                    sa.select(self.quality_scores).where(
                        self.quality_scores.c.model_run_id == model_run_id
                    )
                ).fetchall()
                quality_scores_dict: dict[str, float | None] = {
                    r.category: r.score for r in qs_rows
                }

                # Fetch efficiency indices (may not exist)
                ei_row = conn.execute(
                    sa.select(self.efficiency_indices).where(
                        self.efficiency_indices.c.model_run_id == model_run_id
                    )
                ).fetchone()

                efficiency: EfficiencyIndices | None = None
                if ei_row is not None:
                    efficiency = EfficiencyIndices(
                        quality_per_ram=ei_row.quality_per_ram,
                        quality_per_latency=ei_row.quality_per_latency,
                        quality_per_cpu=ei_row.quality_per_cpu,
                        quality_per_disk=ei_row.quality_per_disk,
                        tps_per_gb_ram=ei_row.tps_per_gb_ram,
                        quality_per_energy=ei_row.quality_per_energy,
                        norm_quality_per_ram=ei_row.norm_quality_per_ram,
                        norm_quality_per_latency=ei_row.norm_quality_per_latency,
                        norm_quality_per_cpu=ei_row.norm_quality_per_cpu,
                        norm_quality_per_disk=ei_row.norm_quality_per_disk,
                        norm_tps_per_gb_ram=ei_row.norm_tps_per_gb_ram,
                        norm_quality_per_energy=ei_row.norm_quality_per_energy,
                    )

                # Reconstruct RobustnessMetrics
                total_prompts_row = conn.execute(
                    sa.select(sa.func.count())
                    .select_from(self.prompt_results)
                    .where(self.prompt_results.c.model_run_id == model_run_id)
                ).scalar() or 0
                timed_out_count = conn.execute(
                    sa.select(sa.func.count())
                    .select_from(self.prompt_results)
                    .where(
                        self.prompt_results.c.model_run_id == model_run_id,
                        self.prompt_results.c.timed_out == 1,
                    )
                ).scalar() or 0
                error_count = conn.execute(
                    sa.select(sa.func.count())
                    .select_from(self.prompt_results)
                    .where(
                        self.prompt_results.c.model_run_id == model_run_id,
                        self.prompt_results.c.error.isnot(None),
                    )
                ).scalar() or 0
                incomplete_prompts = timed_out_count + error_count

                stability_score = run.stability_score if run.stability_score is not None else (
                    (total_prompts_row - incomplete_prompts) / total_prompts_row
                    if total_prompts_row > 0
                    else 0.0
                )

                robustness = RobustnessMetrics(
                    total_errors=run.total_errors or 0,
                    total_timeouts=run.total_timeouts or 0,
                    oom_count=run.total_oom or 0,
                    restart_count=run.error_restarts or 0,
                    incomplete_prompts=incomplete_prompts,
                    stability_score=stability_score,
                )

                results.append(
                    ModelResult(
                        model_name=run.model_name,
                        status=run.status,
                        download_time_s=run.download_time_s,
                        model_size_gb=run.model_size_gb,
                        cold_start_s=run.cold_start_s,
                        inference_results=[],  # not stored per-run; use get_prompt_results
                        resource_summary=None,  # use get_resource_timeseries for raw samples
                        quality_scores=quality_scores_dict,
                        efficiency_indices=efficiency,
                        robustness=robustness,
                        error_log=[],
                        model_run_id=model_run_id,
                        overall_rank=run.overall_rank,
                        quality_score=run.quality_score,
                        avg_tps=run.avg_tps,
                        avg_ttft_ms=run.avg_ttft_ms,
                        avg_latency_ms=run.avg_latency_ms,
                        avg_ram_mb=None,  # filled below from resource samples
                    )
                )

        # Attach avg_ram_mb from resource samples (not stored on model_runs)
        for result in results:
            if result.model_run_id is None:
                continue
            samples = self.get_resource_timeseries(result.model_run_id)
            if samples:
                ram_values = [s.ram_mb for s in samples if s.ram_mb is not None]
                if ram_values:
                    result.avg_ram_mb = sum(ram_values) / len(ram_values)

        return results

    def get_resource_timeseries(self, model_run_id: int) -> list[ResourceSample]:
        """Return all resource samples for a model run, ordered by sampled_at."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(self.resource_samples)
                .where(self.resource_samples.c.model_run_id == model_run_id)
                .order_by(self.resource_samples.c.sampled_at)
            ).fetchall()

        samples: list[ResourceSample] = []
        for row in rows:
            sampled_at = datetime.fromisoformat(row.sampled_at)
            # DB stores average cpu_percent as a single float; wrap it in a list
            cpu_per_core: list[float] = (
                [row.cpu_percent] if row.cpu_percent is not None else []
            )
            samples.append(
                ResourceSample(
                    timestamp=sampled_at,
                    cpu_per_core=cpu_per_core,
                    ram_mb=row.ram_mb or 0.0,
                    gpu_percent=row.gpu_percent,
                    vram_mb=row.vram_mb,
                    cpu_temp_c=row.cpu_temp_c,
                    gpu_temp_c=row.gpu_temp_c,
                    power_watts=row.power_watts,
                )
            )

        return samples

    def get_prompt_results(self, model_run_id: int) -> list[PromptResult]:
        """Return all prompt results for a model run."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(self.prompt_results).where(
                    self.prompt_results.c.model_run_id == model_run_id
                )
            ).fetchall()

        return [
            PromptResult(
                id=row.id,
                model_run_id=row.model_run_id,
                category=row.category,
                prompt_text=row.prompt_text,
                response_text=row.response_text,
                ttft_ms=row.ttft_ms,
                total_ms=row.total_ms,
                tokens_generated=row.tokens_generated,
                tokens_per_second=row.tokens_per_second,
                avg_inter_token_ms=row.avg_inter_token_ms,
                quality_score=row.quality_score,
                timed_out=bool(row.timed_out),
                error=row.error,
            )
            for row in rows
        ]

    def compare_sessions(self, session_ids: list[int]) -> ComparisonReport:
        """Return a cross-session comparison report."""
        # Gather all model names across the requested sessions
        all_model_names: list[str] = []
        # metrics: metric_name -> model_name -> value
        metrics: dict[str, dict[str, float | None]] = {
            "avg_latency_ms": {},
            "avg_tps": {},
            "quality_score": {},
            "stability_score": {},
            "overall_rank": {},
        }

        for session_id in session_ids:
            with self.engine.connect() as conn:
                runs = conn.execute(
                    sa.select(self.model_runs).where(
                        self.model_runs.c.session_id == session_id
                    )
                ).fetchall()

            for run in runs:
                model_name = run.model_name
                if model_name not in all_model_names:
                    all_model_names.append(model_name)

                # Use session-qualified key when same model appears in multiple sessions
                # to avoid overwriting — key format: "<model>@<session_id>"
                key = f"{model_name}@{session_id}"
                if key not in all_model_names:
                    all_model_names.append(key)
                # Remove the bare model name added above if it would be ambiguous
                # (present in more than one session); keep simple names for single-session models
                metrics["avg_latency_ms"][key] = run.avg_latency_ms
                metrics["avg_tps"][key] = run.avg_tps
                metrics["quality_score"][key] = run.quality_score
                metrics["stability_score"][key] = run.stability_score
                metrics["overall_rank"][key] = run.overall_rank

        # Rebuild model list: use session-qualified keys as model identifiers
        qualified_models = list(metrics["avg_latency_ms"].keys())

        return ComparisonReport(
            session_ids=list(session_ids),
            models=qualified_models,
            metrics=metrics,
        )
