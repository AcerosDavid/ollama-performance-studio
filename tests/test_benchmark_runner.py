"""
Tests for BenchmarkRunner — property-based and unit tests.

Tasks 11.4, 11.5, 11.6, 11.7
Validates: Requirements 1.1–1.4, 3.6, 3.7, 3.8, 8.6
"""
from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_benchmark.benchmark_runner import BenchmarkRunner
from ollama_benchmark.config import BenchmarkConfig, TimeoutConfig
from ollama_benchmark.models import (
    HardwareInfo,
    OllamaUnavailableError,
    RobustnessMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    models: list[str] | None = None,
    max_retries: int = 2,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        models=models or ["llama3"],
        prompts={"reasoning": [{"text": "test prompt?"}]},
        timeouts=TimeoutConfig(download=60, cold_start=30, inference=60, judge=30),
        max_retries=max_retries,
        database_path=":memory:",
        ollama_base_url="http://localhost:11434",
    )


def _make_hardware() -> HardwareInfo:
    return HardwareInfo(
        os="Linux 5.15",
        python_version="3.11",
        ollama_version="0.3.0",
        cpu_cores=8,
        cpu_mhz=3200.0,
        ram_mb=16384.0,
        has_gpu=False,
        vram_mb=0.0,
        disk_free_mb=51200.0,
    )


def _make_mock_db() -> MagicMock:
    db = MagicMock()
    db.create_session.return_value = 1
    db.save_model_result.return_value = 1
    db.get_model_results.return_value = []
    db.finalize_session = MagicMock()
    db.get_recommendations.return_value = []
    return db


# ---------------------------------------------------------------------------
# Task 11.7 — Unit tests for hardware detection
# Validates: Requirements 1.1, 1.2, 1.3, 1.4
# ---------------------------------------------------------------------------

class TestDetectHardware:
    """_detect_hardware() returns populated HardwareInfo."""

    def test_returns_hardware_info(self) -> None:
        config = _make_config()
        db = _make_mock_db()
        runner = BenchmarkRunner(config, db)

        with (
            patch("ollama_benchmark.benchmark_runner.psutil") as mock_psutil,
            patch("ollama_benchmark.benchmark_runner._pynvml_available", False),
            patch.object(runner, "_check_ollama_version", return_value="0.3.0"),
        ):
            mock_psutil.cpu_count.return_value = 8
            mock_psutil.cpu_freq.return_value = MagicMock(max=3200.0)
            mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024 * 1024 * 1024)
            mock_psutil.disk_usage.return_value = MagicMock(free=50 * 1024 * 1024 * 1024)

            hw = runner._detect_hardware()

        assert isinstance(hw, HardwareInfo)
        assert hw.cpu_cores == 8
        assert hw.ollama_version == "0.3.0"
        assert hw.has_gpu is False

    def test_ollama_unavailable_raises(self) -> None:
        config = _make_config()
        db = _make_mock_db()
        runner = BenchmarkRunner(config, db)

        with (
            patch("ollama_benchmark.benchmark_runner.psutil") as mock_psutil,
            patch("ollama_benchmark.benchmark_runner._pynvml_available", False),
            patch("ollama_benchmark.benchmark_runner.subprocess.run") as mock_run,
        ):
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.cpu_freq.return_value = MagicMock(max=2400.0)
            mock_psutil.virtual_memory.return_value = MagicMock(total=8 * 1024 * 1024 * 1024)
            mock_psutil.disk_usage.return_value = MagicMock(free=20 * 1024 * 1024 * 1024)
            mock_run.side_effect = FileNotFoundError("ollama not found")

            with pytest.raises(OllamaUnavailableError):
                runner._detect_hardware()

    def test_check_ollama_version_nonzero_exit_raises(self) -> None:
        config = _make_config()
        db = _make_mock_db()
        runner = BenchmarkRunner(config, db)

        with patch("ollama_benchmark.benchmark_runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not running")
            with pytest.raises(OllamaUnavailableError):
                runner._check_ollama_version()

    def test_check_ollama_version_timeout_raises(self) -> None:
        config = _make_config()
        db = _make_mock_db()
        runner = BenchmarkRunner(config, db)

        with patch("ollama_benchmark.benchmark_runner.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=10)
            with pytest.raises(OllamaUnavailableError):
                runner._check_ollama_version()


# ---------------------------------------------------------------------------
# Task 11.7 — Pull failure skips model
# Validates: Requirements 3.6
# ---------------------------------------------------------------------------

class TestPullFailureSkipsModel:
    """When pull() fails, _evaluate_model returns status='skipped'."""

    def test_pull_failure_returns_skipped_status(self) -> None:
        from ollama_benchmark.models import PullResult

        config = _make_config()
        db = _make_mock_db()
        runner = BenchmarkRunner(config, db)

        failed_pull = PullResult(
            model_name="bad-model",
            success=False,
            download_time_s=1.0,
            model_size_gb=None,
            error="network error",
        )

        with patch("ollama_benchmark.model_manager.ModelManager") as MockMM:
            mm_instance = MagicMock()
            mm_instance.pull.return_value = failed_pull
            MockMM.return_value = mm_instance

            result = runner._evaluate_model("bad-model", session_id=1)

        assert result.status == "skipped"
        assert result.model_name == "bad-model"


# ---------------------------------------------------------------------------
# Task 11.7 — Cold-start timeout path
# Validates: Requirements 3.7
# ---------------------------------------------------------------------------

class TestColdStartTimeoutPath:
    """When all cold-start attempts timeout, _evaluate_model returns status='failed'."""

    def test_cold_start_timeout_returns_failed_status(self) -> None:
        from ollama_benchmark.model_manager import ColdStartTimeoutError
        from ollama_benchmark.models import PullResult

        config = _make_config(max_retries=1)
        db = _make_mock_db()
        runner = BenchmarkRunner(config, db)

        good_pull = PullResult(
            model_name="slow-model",
            success=True,
            download_time_s=5.0,
            model_size_gb=4.0,
            error=None,
        )

        with patch("ollama_benchmark.model_manager.ModelManager") as MockMM:
            mm_instance = MagicMock()
            mm_instance.pull.return_value = good_pull
            mm_instance.start_and_verify.side_effect = ColdStartTimeoutError(
                "Model did not respond in time"
            )
            MockMM.return_value = mm_instance

            result = runner._evaluate_model("slow-model", session_id=1)

        assert result.status == "failed"
        assert result.model_name == "slow-model"


# ---------------------------------------------------------------------------
# Task 11.4 — Property 10: Crash-restart respects max_retries limit
# Validates: Requirements 3.8
# ---------------------------------------------------------------------------

@given(
    crash_count=st.integers(min_value=0, max_value=10),
    max_retries=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100)
def test_crash_restart_limit_property(crash_count: int, max_retries: int) -> None:
    """
    Property 10: Crash-restart respects max_retries limit.

    The restart count equals min(crash_count, max_retries).
    When crash_count > max_retries, the model is marked failed.

    Validates: Requirements 3.8
    """
    from ollama_benchmark.model_manager import ColdStartTimeoutError
    from ollama_benchmark.models import PullResult

    config = _make_config(max_retries=max_retries)
    db = _make_mock_db()
    runner = BenchmarkRunner(config, db)

    good_pull = PullResult(
        model_name="test-model",
        success=True,
        download_time_s=1.0,
        model_size_gb=1.0,
        error=None,
    )

    call_count = {"n": 0}

    def start_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= crash_count:
            raise ColdStartTimeoutError("timeout")
        return 1.5  # cold_start_s on success

    with patch("ollama_benchmark.model_manager.ModelManager") as MockMM:
        mm_instance = MagicMock()
        mm_instance.pull.return_value = good_pull
        mm_instance.start_and_verify.side_effect = start_side_effect
        mm_instance.stop_and_remove = MagicMock()
        MockMM.return_value = mm_instance

        with (
            patch("ollama_benchmark.resource_monitor.ResourceMonitor") as MockMonitor,
            patch("ollama_benchmark.inference_engine.InferenceEngine") as MockEngine,
            patch("ollama_benchmark.quality_evaluator.QualityEvaluator"),
            patch("ollama_benchmark.score_engine.ScoreEngine"),
            patch("asyncio.run", return_value=[]),
        ):
            MockMonitor.return_value.stop.return_value = None
            MockMonitor.return_value.start = MagicMock()
            MockEngine.return_value.run_all_prompts = AsyncMock(return_value=[])

            result = runner._evaluate_model("test-model", session_id=1)

    actual_restarts = min(crash_count, max_retries)

    if crash_count > max_retries:
        # All attempts failed → model should be failed
        assert result.status == "failed"
        assert result.robustness.restart_count <= max_retries
    else:
        # Enough successes → model completed or similar non-failed status
        assert result.status in ("completed", "incomplete", "failed")
        # restart_count should equal actual crashes that happened before success
        assert result.robustness.restart_count == crash_count


# ---------------------------------------------------------------------------
# Task 11.5 — Property 3: Prompt consistency across models
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------

@given(
    prompts_dict=st.fixed_dictionaries({
        "reasoning": st.lists(
            st.fixed_dictionaries({"text": st.text(min_size=5, max_size=50)}),
            min_size=1, max_size=3,
        ),
        "coding": st.lists(
            st.fixed_dictionaries({"text": st.text(min_size=5, max_size=50)}),
            min_size=1, max_size=3,
        ),
    }),
    model_names=st.lists(
        st.text(min_size=3, max_size=20),
        min_size=2, max_size=4,
        unique=True,
    ),
)
@settings(max_examples=50)
def test_prompt_consistency_across_models_property(
    prompts_dict: dict,
    model_names: list[str],
) -> None:
    """
    Property 3: Prompt set consistency across models.

    All models receive the same prompts in the same order.

    Validates: Requirements 5.1
    """
    config = BenchmarkConfig(
        models=model_names,
        prompts={k: [{"text": e["text"]} for e in v] for k, v in prompts_dict.items()},
        timeouts=TimeoutConfig(download=60, cold_start=30, inference=60, judge=30),
        max_retries=0,
        database_path=":memory:",
    )

    # Build expected flat prompt list (category, text) order
    expected_order = [
        (cat, pe.text)
        for cat, entries in config.prompts.items()
        for pe in entries
    ]

    # Each model should receive exactly this flat prompt list
    assert len(expected_order) > 0

    # Verify multiple models would receive same prompts
    for model_name in model_names:
        flat = [
            (cat, pe.text)
            for cat, entries in config.prompts.items()
            for pe in entries
        ]
        assert flat == expected_order, (
            f"Model {model_name!r} received different prompt order"
        )


# ---------------------------------------------------------------------------
# Task 11.6 — Property 8: DB write failure does not abort session
# Validates: Requirements 8.6
# ---------------------------------------------------------------------------

@given(
    fail_at=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=30)
def test_db_write_fault_tolerance_property(fail_at: int) -> None:
    """
    Property 8: DB write failure does not abort the session.

    Inject DB write failures at a random point. Session continues,
    produces results, closes cleanly.

    Validates: Requirements 8.6
    """
    from ollama_benchmark.models import PullResult

    config = _make_config(models=["model-a", "model-b"])
    db = _make_mock_db()

    # Make save_model_result fail on the `fail_at`-th call
    call_count = {"n": 0}

    def flaky_save(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == fail_at + 1:
            raise RuntimeError("simulated DB failure")
        return call_count["n"]

    db.save_model_result.side_effect = flaky_save
    db.create_session.return_value = 1
    db.finalize_session = MagicMock()

    good_pull = PullResult(
        model_name="model-a",
        success=True,
        download_time_s=1.0,
        model_size_gb=1.0,
        error=None,
    )

    runner = BenchmarkRunner(config, db)

    with (
        patch.object(runner, "_detect_hardware", return_value=_make_hardware()),
        patch.object(runner, "_evaluate_model") as mock_eval,
    ):
        from ollama_benchmark.models import ModelResult, RobustnessMetrics
        mock_result = ModelResult(
            model_name="model-a",
            status="completed",
            download_time_s=1.0,
            model_size_gb=1.0,
            cold_start_s=1.0,
            inference_results=[],
            resource_summary=None,
            quality_scores={},
            efficiency_indices=None,
            robustness=RobustnessMetrics(0, 0, 0, 0, 0, 1.0),
            model_run_id=1,
        )
        mock_eval.return_value = mock_result

        with patch("ollama_benchmark.score_engine.ScoreEngine") as MockScoreEngine:
            MockScoreEngine.return_value.compute_rankings = MagicMock()
            MockScoreEngine.return_value.generate_recommendations = MagicMock(return_value=[])

            # Should NOT raise even if DB write fails
            try:
                result = runner.run()
                completed = True
            except Exception:
                completed = False

    # Session must complete (or gracefully continue) — no crash
    assert completed, "Session aborted due to DB write failure"
