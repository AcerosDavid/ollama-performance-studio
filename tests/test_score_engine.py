"""
Tests for ScoreEngine — property-based and unit tests.

Tasks 10.3, 10.5, 10.6, 10.8
Validates: Requirements 7.3, 7.4, 11.3, 12.2
"""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_benchmark.models import (
    EfficiencyIndices,
    InferenceResult,
    ModelResult,
    Recommendation,
    ResourceSummary,
    RobustnessMetrics,
)
from ollama_benchmark.score_engine import ScoreEngine, _safe_div


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_robustness(stability: float = 1.0) -> RobustnessMetrics:
    return RobustnessMetrics(
        total_errors=0,
        total_timeouts=0,
        oom_count=0,
        restart_count=0,
        incomplete_prompts=0,
        stability_score=stability,
    )


def _make_model_result(
    model_name: str = "model",
    status: str = "completed",
    quality_scores: dict | None = None,
    inference_results: list | None = None,
    resource_summary=None,
    model_size_gb: float | None = None,
    efficiency_indices=None,
    stability: float = 1.0,
) -> ModelResult:
    return ModelResult(
        model_name=model_name,
        status=status,
        download_time_s=1.0,
        model_size_gb=model_size_gb,
        cold_start_s=1.0,
        inference_results=inference_results or [],
        resource_summary=resource_summary,
        quality_scores=quality_scores or {},
        efficiency_indices=efficiency_indices,
        robustness=_make_robustness(stability),
    )


def _make_inference_result(
    tps: float | None = 10.0,
    latency_ms: float | None = 100.0,
) -> InferenceResult:
    return InferenceResult(
        prompt_text="test",
        response_text="response",
        ttft_ms=50.0,
        total_response_ms=latency_ms,
        tokens_generated=10,
        tokens_per_second=tps,
        avg_inter_token_ms=10.0,
        timed_out=False,
    )


def _make_resource_summary(
    avg_ram_mb: float = 4096.0,
    avg_cpu_percent: float = 40.0,
    avg_gpu_percent: float | None = None,
    avg_power_watts: float | None = None,
) -> ResourceSummary:
    return ResourceSummary(
        avg_cpu_percent=avg_cpu_percent,
        max_cpu_percent=avg_cpu_percent + 10.0,
        avg_ram_mb=avg_ram_mb,
        max_ram_mb=avg_ram_mb + 512.0,
        avg_gpu_percent=avg_gpu_percent,
        max_gpu_percent=avg_gpu_percent,
        avg_vram_mb=None,
        max_vram_mb=None,
        max_temp_cpu_c=None,
        max_temp_gpu_c=None,
        avg_power_watts=avg_power_watts,
        samples=[],
    )


def _make_efficiency_indices(**kwargs) -> EfficiencyIndices:
    defaults = dict(
        quality_per_ram=None,
        quality_per_latency=None,
        quality_per_cpu=None,
        quality_per_disk=None,
        tps_per_gb_ram=None,
        quality_per_energy=None,
        norm_quality_per_ram=None,
        norm_quality_per_latency=None,
        norm_quality_per_cpu=None,
        norm_quality_per_disk=None,
        norm_tps_per_gb_ram=None,
        norm_quality_per_energy=None,
    )
    defaults.update(kwargs)
    return EfficiencyIndices(**defaults)


def _make_mock_db(session_id: int = 1, model_results: list | None = None) -> MagicMock:
    db = MagicMock()
    db.get_model_results.return_value = model_results or []
    db.save_recommendation = MagicMock()
    db.save_efficiency_indices = MagicMock()

    # Mock engine.begin() context manager
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute = MagicMock()
    db.engine.begin.return_value = mock_conn

    # Mock engine.connect() context manager
    mock_conn2 = MagicMock()
    mock_conn2.__enter__ = MagicMock(return_value=mock_conn2)
    mock_conn2.__exit__ = MagicMock(return_value=False)
    mock_conn2.execute.return_value.fetchone.return_value = (1,)
    db.engine.connect.return_value = mock_conn2
    db.model_runs = MagicMock()

    return db


# ---------------------------------------------------------------------------
# Task 10.3 — Property 5: Min-max normalization
# Validates: Requirements 7.3
# ---------------------------------------------------------------------------

class TestNormalizationUnit:
    """Unit tests for normalize_indices() edge cases."""

    def test_single_model_normalized_to_1(self) -> None:
        """Single model with data should have all norm fields set to 1.0."""
        ei = _make_efficiency_indices(quality_per_ram=0.5)
        mr = _make_model_result(efficiency_indices=ei)

        db = _make_mock_db()
        engine = ScoreEngine(db)
        engine.normalize_indices([mr])

        assert mr.efficiency_indices.norm_quality_per_ram == pytest.approx(1.0)

    def test_identical_values_normalize_to_1(self) -> None:
        """All models with identical values should all get 1.0."""
        mr1 = _make_model_result("a", efficiency_indices=_make_efficiency_indices(quality_per_ram=0.5))
        mr2 = _make_model_result("b", efficiency_indices=_make_efficiency_indices(quality_per_ram=0.5))
        mr3 = _make_model_result("c", efficiency_indices=_make_efficiency_indices(quality_per_ram=0.5))

        db = _make_mock_db()
        engine = ScoreEngine(db)
        engine.normalize_indices([mr1, mr2, mr3])

        for mr in [mr1, mr2, mr3]:
            assert mr.efficiency_indices.norm_quality_per_ram == pytest.approx(1.0)

    def test_min_normalizes_to_0_max_to_1(self) -> None:
        """Min value → 0.0, max value → 1.0."""
        mr1 = _make_model_result("min", efficiency_indices=_make_efficiency_indices(quality_per_ram=1.0))
        mr2 = _make_model_result("max", efficiency_indices=_make_efficiency_indices(quality_per_ram=5.0))

        db = _make_mock_db()
        engine = ScoreEngine(db)
        engine.normalize_indices([mr1, mr2])

        assert mr1.efficiency_indices.norm_quality_per_ram == pytest.approx(0.0)
        assert mr2.efficiency_indices.norm_quality_per_ram == pytest.approx(1.0)

    def test_order_preserved_after_normalization(self) -> None:
        """If original[i] < original[j], then normalized[i] < normalized[j]."""
        values = [1.0, 3.0, 5.0, 2.0, 4.0]
        model_results = [
            _make_model_result(f"m{i}", efficiency_indices=_make_efficiency_indices(quality_per_ram=v))
            for i, v in enumerate(values)
        ]

        db = _make_mock_db()
        engine = ScoreEngine(db)
        engine.normalize_indices(model_results)

        norm_values = [mr.efficiency_indices.norm_quality_per_ram for mr in model_results]

        # Check all in [0, 1]
        for nv in norm_values:
            assert 0.0 <= nv <= 1.0

        # Check order preserved
        for i in range(len(values)):
            for j in range(len(values)):
                if values[i] < values[j]:
                    assert norm_values[i] < norm_values[j]

    def test_null_values_remain_null(self) -> None:
        """Models with None raw values should have None norm_ values."""
        mr1 = _make_model_result("a", efficiency_indices=_make_efficiency_indices(quality_per_ram=None))
        mr2 = _make_model_result("b", efficiency_indices=_make_efficiency_indices(quality_per_ram=1.0))

        db = _make_mock_db()
        engine = ScoreEngine(db)
        engine.normalize_indices([mr1, mr2])

        assert mr1.efficiency_indices.norm_quality_per_ram is None
        assert mr2.efficiency_indices.norm_quality_per_ram == pytest.approx(1.0)

    def test_no_data_leaves_all_norm_none(self) -> None:
        """If all models have None for an index, norm_ stays None for all."""
        mr1 = _make_model_result("a", efficiency_indices=_make_efficiency_indices(quality_per_ram=None))
        mr2 = _make_model_result("b", efficiency_indices=_make_efficiency_indices(quality_per_ram=None))

        db = _make_mock_db()
        engine = ScoreEngine(db)
        engine.normalize_indices([mr1, mr2])

        assert mr1.efficiency_indices.norm_quality_per_ram is None
        assert mr2.efficiency_indices.norm_quality_per_ram is None


@given(
    raw_values=st.lists(
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    )
)
@settings(max_examples=200)
def test_normalization_property_range_and_order(raw_values: list[float]) -> None:
    """
    Property 5: Min-max normalization is correct, preserves order, handles identical values.

    Validates: Requirements 7.3
    """
    model_results = [
        _make_model_result(f"m{i}", efficiency_indices=_make_efficiency_indices(quality_per_ram=v))
        for i, v in enumerate(raw_values)
    ]

    db = _make_mock_db()
    engine = ScoreEngine(db)
    engine.normalize_indices(model_results)

    norm_values = [mr.efficiency_indices.norm_quality_per_ram for mr in model_results]

    # All normalized values must be in [0.0, 1.0]
    for nv in norm_values:
        assert nv is not None
        assert 0.0 <= nv <= 1.0, f"Normalized value {nv} out of range [0, 1]"

    # Order must be preserved: if raw[i] < raw[j], then norm[i] <= norm[j]
    for i in range(len(raw_values)):
        for j in range(len(raw_values)):
            if raw_values[i] < raw_values[j]:
                assert norm_values[i] <= norm_values[j], (
                    f"Order not preserved: raw[{i}]={raw_values[i]} < raw[{j}]={raw_values[j]} "
                    f"but norm[{i}]={norm_values[i]} > norm[{j}]={norm_values[j]}"
                )


@given(
    value=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_identical_values_normalize_to_1_property(value: float) -> None:
    """Property 5b: Identical input values all normalize to 1.0."""
    # All models have the same value
    n = 3
    model_results = [
        _make_model_result(f"m{i}", efficiency_indices=_make_efficiency_indices(quality_per_ram=value))
        for i in range(n)
    ]

    db = _make_mock_db()
    engine = ScoreEngine(db)
    engine.normalize_indices(model_results)

    for mr in model_results:
        nv = mr.efficiency_indices.norm_quality_per_ram
        assert nv == pytest.approx(1.0), f"Expected 1.0 for identical values, got {nv}"


# ---------------------------------------------------------------------------
# Task 10.5 — Property 6: Stability score reflects completion ratio
# Validates: Requirements 12.2
# ---------------------------------------------------------------------------

@given(
    completed=st.integers(min_value=0, max_value=1000),
    total=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=300)
def test_stability_score_property(completed: int, total: int) -> None:
    """
    Property 6: Stability score = completed / total ∈ [0.0, 1.0].

    Validates: Requirements 12.2
    """
    # Only test valid pairs where 0 <= completed <= total
    if completed > total:
        completed = total  # clamp to valid range

    db = _make_mock_db()
    engine = ScoreEngine(db)
    score = engine.compute_stability_score(completed, total)

    expected = completed / total
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(expected)


class TestStabilityScoreUnit:
    """Unit tests for compute_stability_score()."""

    def test_full_completion(self) -> None:
        db = _make_mock_db()
        engine = ScoreEngine(db)
        assert engine.compute_stability_score(10, 10) == pytest.approx(1.0)

    def test_zero_completion(self) -> None:
        db = _make_mock_db()
        engine = ScoreEngine(db)
        assert engine.compute_stability_score(0, 10) == pytest.approx(0.0)

    def test_partial_completion(self) -> None:
        db = _make_mock_db()
        engine = ScoreEngine(db)
        assert engine.compute_stability_score(5, 10) == pytest.approx(0.5)

    def test_zero_total_returns_zero(self) -> None:
        db = _make_mock_db()
        engine = ScoreEngine(db)
        # Should not raise ZeroDivisionError
        assert engine.compute_stability_score(0, 0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Task 10.6 — Property 7: Overall rank is unweighted mean of norm scores
# Validates: Requirements 7.4
# ---------------------------------------------------------------------------

@given(
    norm_scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=200)
def test_overall_rank_property(norm_scores: list[float]) -> None:
    """
    Property 7: Overall rank = sum(norm_scores) / K ∈ [0.0, 1.0].

    Validates: Requirements 7.4
    """
    # Build a model with some normalized indices
    fields = [
        "norm_quality_per_ram",
        "norm_quality_per_latency",
        "norm_quality_per_cpu",
        "norm_quality_per_disk",
        "norm_tps_per_gb_ram",
        "norm_quality_per_energy",
    ]

    # Assign scores to fields, using only the first min(len(fields), K) components
    n_components = min(len(norm_scores), len(fields))
    kwargs = {fields[i]: norm_scores[i] for i in range(n_components)}
    # Include stability in last position if there are extra scores
    if len(norm_scores) > n_components:
        stability = norm_scores[n_components]
    else:
        stability = None

    ei = _make_efficiency_indices(**kwargs)
    mr = _make_model_result(
        efficiency_indices=ei,
        stability=stability if stability is not None else 1.0,
    )
    if stability is None:
        mr.robustness.stability_score = None  # type: ignore

    db = _make_mock_db()
    engine = ScoreEngine(db)
    rank = engine.compute_overall_rank(mr)

    # Must be in [0, 1]
    assert 0.0 <= rank <= 1.0

    # Must equal unweighted mean of available components
    all_components = [norm_scores[i] for i in range(n_components)]
    if stability is not None:
        all_components.append(stability)
    if all_components:
        expected = sum(all_components) / len(all_components)
        assert rank == pytest.approx(expected, abs=1e-9), (
            f"overall_rank {rank} != mean of {all_components} = {expected}"
        )


class TestOverallRankUnit:
    """Unit tests for compute_overall_rank()."""

    def test_no_indices_returns_0(self) -> None:
        mr = _make_model_result(efficiency_indices=None, stability=0.0)
        mr.robustness.stability_score = None  # type: ignore
        db = _make_mock_db()
        engine = ScoreEngine(db)
        assert engine.compute_overall_rank(mr) == pytest.approx(0.0)

    def test_single_norm_score(self) -> None:
        ei = _make_efficiency_indices(norm_quality_per_ram=0.8)
        mr = _make_model_result(efficiency_indices=ei)
        mr.robustness.stability_score = None  # type: ignore
        db = _make_mock_db()
        engine = ScoreEngine(db)
        rank = engine.compute_overall_rank(mr)
        # Only one component (0.8), so mean = 0.8
        assert rank == pytest.approx(0.8)

    def test_all_norm_plus_stability_mean(self) -> None:
        ei = _make_efficiency_indices(
            norm_quality_per_ram=0.5,
            norm_quality_per_latency=0.5,
        )
        mr = _make_model_result(efficiency_indices=ei, stability=1.0)
        db = _make_mock_db()
        engine = ScoreEngine(db)
        rank = engine.compute_overall_rank(mr)
        expected = (0.5 + 0.5 + 1.0) / 3
        assert rank == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Task 10.8 — Unit tests for score engine
# Validates: Requirements 7.3, 7.4, 11.3
# ---------------------------------------------------------------------------

class TestEfficiencyIndexComputation:
    """compute_efficiency_indices() produces correct raw indices."""

    def test_null_when_no_quality_scores(self) -> None:
        mr = _make_model_result(quality_scores={})
        db = _make_mock_db()
        engine = ScoreEngine(db)
        ei = engine.compute_efficiency_indices(mr)
        # Without quality scores, all quality-based indices should be None
        assert ei.quality_per_ram is None
        assert ei.quality_per_latency is None

    def test_null_when_no_resource_summary(self) -> None:
        mr = _make_model_result(
            quality_scores={"reasoning": 0.8},
            resource_summary=None,
        )
        db = _make_mock_db()
        engine = ScoreEngine(db)
        ei = engine.compute_efficiency_indices(mr)
        # No RAM data → quality_per_ram is None
        assert ei.quality_per_ram is None

    def test_quality_per_ram_computed_correctly(self) -> None:
        mr = _make_model_result(
            quality_scores={"reasoning": 0.8},
            resource_summary=_make_resource_summary(avg_ram_mb=4096.0),
        )
        db = _make_mock_db()
        engine = ScoreEngine(db)
        ei = engine.compute_efficiency_indices(mr)
        expected = 0.8 / 4096.0
        assert ei.quality_per_ram == pytest.approx(expected)

    def test_tps_per_gb_ram_computed_correctly(self) -> None:
        mr = _make_model_result(
            inference_results=[_make_inference_result(tps=100.0)],
            resource_summary=_make_resource_summary(avg_ram_mb=4096.0),
        )
        db = _make_mock_db()
        engine = ScoreEngine(db)
        ei = engine.compute_efficiency_indices(mr)
        # tps_per_gb_ram = tps / (ram_mb / 1024) = 100 / 4 = 25
        expected = 100.0 / (4096.0 / 1024.0)
        assert ei.tps_per_gb_ram == pytest.approx(expected)

    def test_norm_fields_all_none_initially(self) -> None:
        mr = _make_model_result(
            quality_scores={"reasoning": 0.8},
            resource_summary=_make_resource_summary(avg_ram_mb=4096.0),
        )
        db = _make_mock_db()
        engine = ScoreEngine(db)
        ei = engine.compute_efficiency_indices(mr)
        # All norm_ fields should be None until normalize_indices() is called
        assert ei.norm_quality_per_ram is None
        assert ei.norm_quality_per_latency is None
        assert ei.norm_quality_per_cpu is None


class TestRecommendations:
    """generate_recommendations() follows Req 11.3: skip if fewer than 2 models completed."""

    def test_fewer_than_2_completed_returns_empty(self) -> None:
        mr = _make_model_result("only-model", status="completed")
        db = _make_mock_db(model_results=[mr])
        engine = ScoreEngine(db)
        result = engine.generate_recommendations(session_id=1)
        assert result == []

    def test_no_completed_returns_empty(self) -> None:
        mr = _make_model_result("failed-model", status="failed")
        db = _make_mock_db(model_results=[mr])
        engine = ScoreEngine(db)
        result = engine.generate_recommendations(session_id=1)
        assert result == []

    def test_two_completed_returns_recommendations(self) -> None:
        mr1 = _make_model_result(
            "model-a",
            status="completed",
            quality_scores={"reasoning": 0.9},
            resource_summary=_make_resource_summary(avg_ram_mb=4096.0),
        )
        mr2 = _make_model_result(
            "model-b",
            status="completed",
            quality_scores={"reasoning": 0.7},
            resource_summary=_make_resource_summary(avg_ram_mb=8000.0),
        )
        db = _make_mock_db(model_results=[mr1, mr2])
        engine = ScoreEngine(db)
        result = engine.generate_recommendations(session_id=1)

        assert isinstance(result, list)
        # At least one recommendation should be generated
        assert len(result) > 0
        for rec in result:
            assert isinstance(rec, Recommendation)
            assert rec.profile
            assert rec.model_name

    def test_recommendations_persisted_to_db(self) -> None:
        mr1 = _make_model_result(
            "model-a", status="completed",
            quality_scores={"reasoning": 0.9},
            resource_summary=_make_resource_summary(avg_ram_mb=4096.0),
        )
        mr2 = _make_model_result(
            "model-b", status="completed",
            quality_scores={"reasoning": 0.7},
            resource_summary=_make_resource_summary(avg_ram_mb=4096.0),
        )
        db = _make_mock_db(model_results=[mr1, mr2])
        engine = ScoreEngine(db)
        engine.generate_recommendations(session_id=1)

        # DB save_recommendation should have been called
        assert db.save_recommendation.called


class TestSafeDiv:
    """_safe_div() helper handles None/zero denominators."""

    def test_normal_division(self) -> None:
        assert _safe_div(10.0, 2.0) == pytest.approx(5.0)

    def test_none_numerator(self) -> None:
        assert _safe_div(None, 2.0) is None

    def test_none_denominator(self) -> None:
        assert _safe_div(10.0, None) is None

    def test_zero_denominator(self) -> None:
        assert _safe_div(10.0, 0.0) is None

    def test_both_none(self) -> None:
        assert _safe_div(None, None) is None
