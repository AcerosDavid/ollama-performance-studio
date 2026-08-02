"""
Score Engine for ollama-performance-studio.

Computes derived efficiency indices, normalizes them across models in a session,
computes overall rankings, and generates automatic recommendations.

Task coverage
-------------
10.1 — ``compute_efficiency_indices``: raw index computation for one model result.
10.2 — ``normalize_indices``: min-max normalization across all model results.
10.4 — ``compute_rankings``, ``compute_overall_rank``, and ``compute_stability_score``.
10.7 — ``generate_recommendations``.

Requirements: 7.1–7.5, 11.1–11.3, 12.2

Usage
-----
    from ollama_benchmark.score_engine import ScoreEngine

    engine = ScoreEngine(db)
    indices = engine.compute_efficiency_indices(model_result)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import sqlalchemy as sa

from ollama_benchmark.database import Database
from ollama_benchmark.models import (
    EfficiencyIndices,
    ModelResult,
    Recommendation,
)

__all__ = ["ScoreEngine"]

_log = logging.getLogger(__name__)

# All raw efficiency index field names and their corresponding norm_ fields.
_INDEX_FIELDS = [
    "quality_per_ram",
    "quality_per_latency",
    "quality_per_cpu",
    "quality_per_disk",
    "tps_per_gb_ram",
    "quality_per_energy",
]


class ScoreEngine:
    """
    Computes efficiency indices, normalizes them, ranks models, and generates
    recommendations for a benchmark session.

    Parameters
    ----------
    db:
        Initialized ``Database`` instance used to persist computed values.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Task 10.1 — Raw efficiency index computation
    # ------------------------------------------------------------------

    def compute_efficiency_indices(self, model_result: ModelResult) -> EfficiencyIndices:
        """
        Compute raw efficiency indices for one model result.

        Indices are ratios of quality to resource consumption.  Any index
        whose denominator is unavailable or zero is returned as ``None``
        (marked as null in the DB per Requirement 7.5).  Normalized
        counterparts (``norm_*`` fields) are left as ``None`` and filled
        later by ``normalize_indices`` (task 10.2).

        Parameters
        ----------
        model_result:
            Fully-populated ``ModelResult`` for a single model, including
            ``inference_results``, ``resource_summary``, ``quality_scores``,
            and ``model_size_gb``.

        Returns
        -------
        EfficiencyIndices
            Raw indices with all ``norm_*`` fields set to ``None``.
        """
        # ----------------------------------------------------------------
        # 1. Extract source metrics
        # ----------------------------------------------------------------

        # quality_score: mean of non-None values across all categories
        quality_values = [
            v for v in model_result.quality_scores.values() if v is not None
        ]
        quality_score: Optional[float] = (
            sum(quality_values) / len(quality_values) if quality_values else None
        )

        # resource metrics from resource_summary if available
        avg_ram_mb: Optional[float] = None
        avg_cpu_percent: Optional[float] = None
        avg_power_watts: Optional[float] = None

        if model_result.resource_summary is not None:
            rs = model_result.resource_summary
            avg_ram_mb = rs.avg_ram_mb
            avg_cpu_percent = rs.avg_cpu_percent
            avg_power_watts = rs.avg_power_watts

        # avg_latency_ms: mean of non-None total_response_ms from inference_results
        latency_values = [
            r.total_response_ms
            for r in model_result.inference_results
            if r.total_response_ms is not None
        ]
        avg_latency_ms: Optional[float] = (
            sum(latency_values) / len(latency_values) if latency_values else None
        )

        # model_size_gb: directly from model_result
        model_size_gb: Optional[float] = model_result.model_size_gb

        # avg_tps: mean of non-None tokens_per_second from inference_results
        tps_values = [
            r.tokens_per_second
            for r in model_result.inference_results
            if r.tokens_per_second is not None
        ]
        avg_tps: Optional[float] = (
            sum(tps_values) / len(tps_values) if tps_values else None
        )

        # ----------------------------------------------------------------
        # 2. Compute each index safely (Req 7.1, 7.2)
        # ----------------------------------------------------------------

        quality_per_ram = _safe_div(quality_score, avg_ram_mb)
        quality_per_latency = _safe_div(quality_score, avg_latency_ms)
        quality_per_cpu = _safe_div(quality_score, avg_cpu_percent)
        quality_per_disk = _safe_div(quality_score, model_size_gb)

        # tps_per_gb_ram = tokens_per_second / (avg_ram_mb / 1024)
        ram_gb: Optional[float] = (
            avg_ram_mb / 1024.0 if avg_ram_mb is not None else None
        )
        tps_per_gb_ram = _safe_div(avg_tps, ram_gb)

        # quality_per_energy only computed when avg_power_watts is available and > 0 (Req 7.2)
        quality_per_energy: Optional[float] = None
        if avg_power_watts is not None and avg_power_watts > 0:
            quality_per_energy = _safe_div(quality_score, avg_power_watts)

        # ----------------------------------------------------------------
        # 3. Log any null indices (Req 7.5)
        # ----------------------------------------------------------------
        _null_indices = {
            "quality_per_ram": quality_per_ram,
            "quality_per_latency": quality_per_latency,
            "quality_per_cpu": quality_per_cpu,
            "quality_per_disk": quality_per_disk,
            "tps_per_gb_ram": tps_per_gb_ram,
        }
        null_names = [k for k, v in _null_indices.items() if v is None]
        if null_names:
            _log.debug(
                "Model '%s': incomplete data — indices set to null: %s",
                model_result.model_name,
                ", ".join(null_names),
            )

        # ----------------------------------------------------------------
        # 4. Return with all norm_* fields as None (filled by task 10.2)
        # ----------------------------------------------------------------
        return EfficiencyIndices(
            quality_per_ram=quality_per_ram,
            quality_per_latency=quality_per_latency,
            quality_per_cpu=quality_per_cpu,
            quality_per_disk=quality_per_disk,
            tps_per_gb_ram=tps_per_gb_ram,
            quality_per_energy=quality_per_energy,
            norm_quality_per_ram=None,
            norm_quality_per_latency=None,
            norm_quality_per_cpu=None,
            norm_quality_per_disk=None,
            norm_tps_per_gb_ram=None,
            norm_quality_per_energy=None,
        )

    # ------------------------------------------------------------------
    # Task 10.2 — Min-max normalization across all models in a session
    # ------------------------------------------------------------------

    def normalize_indices(self, all_model_results: List[ModelResult]) -> None:
        """
        Min-max normalize efficiency indices across all models in the session,
        updating each ``ModelResult``'s ``efficiency_indices.norm_*`` fields
        in-place.  (Req 7.3)

        Rules:
        - score = (value - min) / (max - min)
        - If all raw values for an index are identical → all normalized to 1.0
        - Null raw values remain null; their corresponding norm_* stays None

        Parameters
        ----------
        all_model_results:
            All ``ModelResult`` instances for a completed session.  Each
            must have ``efficiency_indices`` already populated by
            ``compute_efficiency_indices``.
        """
        for field in _INDEX_FIELDS:
            norm_field = f"norm_{field}"

            # Collect (list-index, raw_value) pairs for non-null entries
            indexed_values: list[tuple[int, float]] = [
                (i, getattr(mr.efficiency_indices, field))
                for i, mr in enumerate(all_model_results)
                if mr.efficiency_indices is not None
                and getattr(mr.efficiency_indices, field) is not None
            ]

            if not indexed_values:
                # No data for this index across all models — leave norm_ as None
                continue

            raw_vals = [v for _, v in indexed_values]
            mn = min(raw_vals)
            mx = max(raw_vals)

            for i, raw_val in indexed_values:
                ei: EfficiencyIndices = all_model_results[i].efficiency_indices  # type: ignore[assignment]
                if mn == mx:
                    # All identical — assign 1.0 to all (Req 7.3)
                    norm_val = 1.0
                else:
                    norm_val = (raw_val - mn) / (mx - mn)
                # EfficiencyIndices is a plain dataclass (not frozen), so
                # setattr works directly.
                setattr(ei, norm_field, norm_val)

    # ------------------------------------------------------------------
    # Task 10.4 — Overall rank and stability score
    # ------------------------------------------------------------------

    def compute_stability_score(self, completed: int, total: int) -> float:
        """
        Stability score = completed / total. (Req 12.2)

        Returns 0.0 when *total* <= 0 to avoid division by zero.
        Result is clamped to [0.0, 1.0].

        Parameters
        ----------
        completed:
            Number of prompts that completed successfully.
        total:
            Total number of prompts in the benchmark suite.
        """
        if total <= 0:
            return 0.0
        return min(1.0, max(0.0, completed / total))

    def compute_overall_rank(self, model_result: ModelResult) -> float:
        """
        Unweighted mean of all available normalized component scores. (Req 7.4)

        Components considered (any that are not None):
        - norm_quality_per_ram
        - norm_quality_per_latency
        - norm_quality_per_cpu
        - norm_quality_per_disk
        - norm_tps_per_gb_ram
        - norm_quality_per_energy
        - stability_score (from robustness)

        Returns 0.0 when no component scores are available.
        """
        scores: list[float] = []

        ei = model_result.efficiency_indices
        if ei is not None:
            for field in [f"norm_{f}" for f in _INDEX_FIELDS]:
                val = getattr(ei, field, None)
                if val is not None:
                    scores.append(val)

        # Include stability score if available
        stability = model_result.robustness.stability_score
        if stability is not None:
            scores.append(stability)

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def compute_rankings(self, session_id: int) -> None:
        """
        Compute and persist ``overall_rank`` for all models in a session.

        Steps:
        1. Load all ``ModelResult`` objects for the session.
        2. Compute raw efficiency indices for any model that doesn't have them.
        3. Min-max normalize all indices across the session.
        4. Compute ``overall_rank`` (unweighted mean of normalized scores) and
           persist it alongside the normalized efficiency indices.

        Parameters
        ----------
        session_id:
            Database session identifier whose model runs should be ranked.
        """
        model_results = self._db.get_model_results(session_id)
        if not model_results:
            _log.info("No model results found for session %d — skipping ranking.", session_id)
            return

        # Step 1: ensure every model has raw efficiency indices
        for mr in model_results:
            if mr.efficiency_indices is None:
                mr.efficiency_indices = self.compute_efficiency_indices(mr)

        # Step 2: normalize across models
        self.normalize_indices(model_results)

        # Step 3: persist overall_rank and normalized indices for each model
        with self._db.engine.begin() as conn:
            for mr in model_results:
                rank = self.compute_overall_rank(mr)

                conn.execute(
                    self._db.model_runs.update()
                    .where(self._db.model_runs.c.session_id == session_id)
                    .where(self._db.model_runs.c.model_name == mr.model_name)
                    .values(overall_rank=rank)
                )

        # Persist normalized efficiency indices via the DB write interface
        for mr in model_results:
            if mr.efficiency_indices is not None:
                model_run_id = self._get_model_run_id(session_id, mr.model_name)
                if model_run_id is not None:
                    self._db.save_efficiency_indices(model_run_id, mr.efficiency_indices)

    def _get_model_run_id(self, session_id: int, model_name: str) -> Optional[int]:
        """Return the ``model_runs.id`` for a given session + model name pair."""
        with self._db.engine.connect() as conn:
            row = conn.execute(
                sa.select(self._db.model_runs.c.id)
                .where(self._db.model_runs.c.session_id == session_id)
                .where(self._db.model_runs.c.model_name == model_name)
            ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Task 10.7 — Automatic recommendations
    # ------------------------------------------------------------------

    def generate_recommendations(self, session_id: int) -> List[Recommendation]:
        """
        Generate automatic recommendations for nine hardware/use-case profiles.
        (Req 11.1, 11.2, 11.3)

        Profiles:
        - ``8gb_ram``        — best quality among models using ≤ 8 192 MB RAM
        - ``12gb_ram``       — best quality among models using ≤ 12 288 MB RAM
        - ``cpu_only``       — best quality without GPU usage
        - ``coding``         — best score in the "coding" quality category
        - ``reasoning``      — best score in the "reasoning" quality category
        - ``conversation``   — best score in the "conversation" quality category
        - ``best_quality_perf`` — best quality_per_latency efficiency index
        - ``fastest``        — highest average tokens-per-second
        - ``most_stable``    — highest stability_score

        Returns an empty list and logs a note when fewer than two models
        completed (Req 11.3).  Each recommendation is also persisted to the DB.

        Parameters
        ----------
        session_id:
            Database session identifier from which model results are read.

        Returns
        -------
        List[Recommendation]
            Up to nine ``Recommendation`` objects, one per profile.
        """
        model_results = self._db.get_model_results(session_id)
        completed = [mr for mr in model_results if mr.status == "completed"]

        # Req 11.3: skip if fewer than 2 models completed
        if len(completed) < 2:
            _log.info(
                "Session %d: fewer than 2 models completed (%d) — "
                "skipping recommendations.",
                session_id,
                len(completed),
            )
            return []

        recommendations: List[Recommendation] = []

        # ----------------------------------------------------------------
        # Helper lambdas / utilities
        # ----------------------------------------------------------------

        def _avg_ram(mr: ModelResult) -> Optional[float]:
            return mr.resource_summary.avg_ram_mb if mr.resource_summary else None

        def _quality(mr: ModelResult) -> Optional[float]:
            vals = [v for v in mr.quality_scores.values() if v is not None]
            return sum(vals) / len(vals) if vals else None

        def _avg_tps(mr: ModelResult) -> Optional[float]:
            tps_vals = [
                r.tokens_per_second
                for r in mr.inference_results
                if r.tokens_per_second is not None
            ]
            return sum(tps_vals) / len(tps_vals) if tps_vals else None

        def _best_by(
            candidates: List[ModelResult],
            key_fn,
            profile: str,
            justification_fn,
        ) -> None:
            """Append the best candidate (by key_fn) to recommendations."""
            eligible = [mr for mr in candidates if key_fn(mr) is not None]
            if not eligible:
                return
            best = max(eligible, key=lambda mr: key_fn(mr))  # type: ignore[arg-type]
            recommendations.append(
                Recommendation(
                    profile=profile,
                    model_name=best.model_name,
                    justification=justification_fn(best),
                )
            )

        # ----------------------------------------------------------------
        # Profile 1 — 8 GB RAM
        # ----------------------------------------------------------------
        fits_8gb = [
            mr for mr in completed
            if _avg_ram(mr) is not None and _avg_ram(mr) <= 8_192  # type: ignore[operator]
        ]
        if fits_8gb:
            best = max(fits_8gb, key=lambda mr: _quality(mr) or 0.0)
            q = _quality(best)
            ram = _avg_ram(best)
            recommendations.append(
                Recommendation(
                    profile="8gb_ram",
                    model_name=best.model_name,
                    justification=(
                        f"Best quality ({q:.3f}) among models using ≤8 GB RAM "
                        f"(used {ram:.0f} MB)"
                    ),
                )
            )

        # ----------------------------------------------------------------
        # Profile 2 — 12 GB RAM
        # ----------------------------------------------------------------
        fits_12gb = [
            mr for mr in completed
            if _avg_ram(mr) is not None and _avg_ram(mr) <= 12_288  # type: ignore[operator]
        ]
        if fits_12gb:
            best = max(fits_12gb, key=lambda mr: _quality(mr) or 0.0)
            q = _quality(best)
            recommendations.append(
                Recommendation(
                    profile="12gb_ram",
                    model_name=best.model_name,
                    justification=(
                        f"Best quality ({q:.3f}) among models using ≤12 GB RAM"
                    ),
                )
            )

        # ----------------------------------------------------------------
        # Profile 3 — CPU only (no meaningful GPU usage)
        # ----------------------------------------------------------------
        cpu_only = [
            mr for mr in completed
            if (
                mr.resource_summary is None
                or mr.resource_summary.avg_gpu_percent is None
                or mr.resource_summary.avg_gpu_percent < 1.0
            )
        ]
        if cpu_only:
            best = max(cpu_only, key=lambda mr: _quality(mr) or 0.0)
            q = _quality(best)
            recommendations.append(
                Recommendation(
                    profile="cpu_only",
                    model_name=best.model_name,
                    justification=f"Best quality ({q:.3f}) without GPU",
                )
            )

        # ----------------------------------------------------------------
        # Profile 4 — Coding
        # ----------------------------------------------------------------
        _best_by(
            completed,
            key_fn=lambda mr: mr.quality_scores.get("coding"),
            profile="coding",
            justification_fn=lambda mr: (
                f"Best coding score ({mr.quality_scores.get('coding'):.3f})"
            ),
        )

        # ----------------------------------------------------------------
        # Profile 5 — Reasoning
        # ----------------------------------------------------------------
        _best_by(
            completed,
            key_fn=lambda mr: mr.quality_scores.get("reasoning"),
            profile="reasoning",
            justification_fn=lambda mr: (
                f"Best reasoning score ({mr.quality_scores.get('reasoning'):.3f})"
            ),
        )

        # ----------------------------------------------------------------
        # Profile 6 — Conversation
        # ----------------------------------------------------------------
        _best_by(
            completed,
            key_fn=lambda mr: mr.quality_scores.get("conversation"),
            profile="conversation",
            justification_fn=lambda mr: (
                f"Best conversation score ({mr.quality_scores.get('conversation'):.3f})"
            ),
        )

        # ----------------------------------------------------------------
        # Profile 7 — Best quality/performance ratio (quality_per_latency)
        # ----------------------------------------------------------------
        def _qpl(mr: ModelResult) -> Optional[float]:
            return (
                mr.efficiency_indices.quality_per_latency
                if mr.efficiency_indices is not None
                else None
            )

        _best_by(
            completed,
            key_fn=_qpl,
            profile="best_quality_perf",
            justification_fn=lambda mr: (
                f"Best quality/latency ratio ({_qpl(mr):.5f})"
            ),
        )

        # ----------------------------------------------------------------
        # Profile 8 — Fastest (highest TPS)
        # ----------------------------------------------------------------
        _best_by(
            completed,
            key_fn=_avg_tps,
            profile="fastest",
            justification_fn=lambda mr: (
                f"Highest tokens/second ({_avg_tps(mr):.1f} TPS)"
            ),
        )

        # ----------------------------------------------------------------
        # Profile 9 — Most stable
        # ----------------------------------------------------------------
        _best_by(
            completed,
            key_fn=lambda mr: mr.robustness.stability_score,
            profile="most_stable",
            justification_fn=lambda mr: (
                f"Highest stability score ({mr.robustness.stability_score:.3f})"
            ),
        )

        # ----------------------------------------------------------------
        # Persist all recommendations to the DB
        # ----------------------------------------------------------------
        for rec in recommendations:
            self._db.save_recommendation(session_id, rec)

        _log.info(
            "Session %d: generated %d recommendations.",
            session_id,
            len(recommendations),
        )
        return recommendations


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_div(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    """
    Divide *numerator* by *denominator*, returning ``None`` whenever the
    operation would be undefined (either operand is ``None``, or
    *denominator* is zero).
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator
