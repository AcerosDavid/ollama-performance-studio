"""
Unit tests for the database layer.

Task 4.4: Validates Requirements 8.5, 8.6
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
from contextlib import contextmanager

import pytest

from ollama_benchmark.database import Database
from ollama_benchmark.models import (
    HardwareInfo,
    InferenceResult,
    ModelResult,
    RobustnessMetrics,
    EfficiencyIndices,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Database:
    """In-memory database instance — fully isolated, no file I/O."""
    return Database(":memory:")


@pytest.fixture
def hardware() -> HardwareInfo:
    """Minimal HardwareInfo for session creation tests."""
    return HardwareInfo(
        os="Linux 5.15",
        python_version="3.11.0",
        ollama_version="0.3.0",
        cpu_cores=8,
        cpu_mhz=3200.0,
        ram_mb=16384.0,
        has_gpu=False,
        vram_mb=0.0,
        disk_free_mb=102400.0,
    )


@pytest.fixture
def config_snapshot() -> dict:
    """Minimal config snapshot dict."""
    return {"models": ["llama3"], "timeout": 120}


def _make_model_result(model_name: str = "llama3", status: str = "completed") -> ModelResult:
    """Build a minimal ModelResult suitable for save_model_result()."""
    robustness = RobustnessMetrics(
        total_errors=0,
        total_timeouts=0,
        oom_count=0,
        restart_count=0,
        incomplete_prompts=0,
        stability_score=1.0,
    )
    return ModelResult(
        model_name=model_name,
        status=status,
        download_time_s=5.0,
        model_size_gb=4.1,
        cold_start_s=1.2,
        inference_results=[],
        resource_summary=None,
        quality_scores={},
        efficiency_indices=None,
        robustness=robustness,
    )


# ---------------------------------------------------------------------------
# 1. Create session
# ---------------------------------------------------------------------------

class TestCreateSession:
    """Validates: Requirements 8.5"""

    def test_returns_integer_id(self, db: Database, hardware: HardwareInfo, config_snapshot: dict) -> None:
        """create_session() must return an integer session ID."""
        session_id = db.create_session(hardware, config_snapshot)
        assert isinstance(session_id, int)

    def test_sequential_calls_return_different_ids(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """Two calls to create_session() must produce distinct IDs."""
        id1 = db.create_session(hardware, config_snapshot)
        id2 = db.create_session(hardware, config_snapshot)
        assert id1 != id2

    def test_id_is_positive(self, db: Database, hardware: HardwareInfo, config_snapshot: dict) -> None:
        """Session IDs must be positive integers (SQLite autoincrement starts at 1)."""
        session_id = db.create_session(hardware, config_snapshot)
        assert session_id >= 1


# ---------------------------------------------------------------------------
# 2. Save model result
# ---------------------------------------------------------------------------

class TestSaveModelResult:
    """Validates: Requirements 8.5, 8.6"""

    def test_returns_model_run_id(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """save_model_result() must return an integer model_run_id."""
        session_id = db.create_session(hardware, config_snapshot)
        result = _make_model_result()
        model_run_id = db.save_model_result(session_id, result)
        assert isinstance(model_run_id, int)
        assert model_run_id >= 1

    def test_get_model_results_returns_saved_row(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """get_model_results() must include the row that was just saved."""
        session_id = db.create_session(hardware, config_snapshot)
        result = _make_model_result(model_name="gemma2", status="completed")
        db.save_model_result(session_id, result)

        rows = db.get_model_results(session_id)
        assert len(rows) == 1
        assert rows[0].model_name == "gemma2"

    def test_get_model_results_correct_status(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """get_model_results() must preserve the status field."""
        session_id = db.create_session(hardware, config_snapshot)
        result = _make_model_result(status="failed")
        db.save_model_result(session_id, result)

        rows = db.get_model_results(session_id)
        assert rows[0].status == "failed"

    def test_second_save_returns_different_run_id(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """Saving two different models in the same session produces distinct run IDs."""
        session_id = db.create_session(hardware, config_snapshot)
        id1 = db.save_model_result(session_id, _make_model_result("modelA"))
        id2 = db.save_model_result(session_id, _make_model_result("modelB"))
        assert id1 != id2


# ---------------------------------------------------------------------------
# 3. Query consistency
# ---------------------------------------------------------------------------

class TestQueryConsistency:
    """Validates: Requirements 8.5, 8.6"""

    def test_session_detail_contains_hardware(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """get_session_detail() must return the hardware that was used to create the session."""
        session_id = db.create_session(hardware, config_snapshot)

        detail = db.get_session_detail(session_id)

        assert detail.hardware.os == hardware.os
        assert detail.hardware.cpu_cores == hardware.cpu_cores
        assert detail.hardware.ram_mb == pytest.approx(hardware.ram_mb)
        assert detail.hardware.has_gpu == hardware.has_gpu

    def test_session_detail_correct_model_count(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """get_session_detail() must report the correct number of model_results."""
        session_id = db.create_session(hardware, config_snapshot)
        db.save_model_result(session_id, _make_model_result("model1"))
        db.save_model_result(session_id, _make_model_result("model2"))
        db.save_model_result(session_id, _make_model_result("model3"))

        detail = db.get_session_detail(session_id)

        assert len(detail.model_results) == 3
        assert detail.summary.model_count == 3

    def test_session_detail_model_names_match(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """Model names in session detail must match what was saved."""
        session_id = db.create_session(hardware, config_snapshot)
        db.save_model_result(session_id, _make_model_result("alpha"))
        db.save_model_result(session_id, _make_model_result("beta"))

        detail = db.get_session_detail(session_id)
        saved_names = {r.model_name for r in detail.model_results}

        assert saved_names == {"alpha", "beta"}

    def test_session_detail_not_found_raises(self, db: Database) -> None:
        """get_session_detail() must raise ValueError for a non-existent session."""
        with pytest.raises(ValueError, match="not found"):
            db.get_session_detail(999)


# ---------------------------------------------------------------------------
# 4. Retry on write failure
# ---------------------------------------------------------------------------

class TestRetryOnWriteFailure:
    """Validates: Requirements 8.6"""

    def test_write_succeeds_after_one_failure(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """_execute_with_retry() must retry once and succeed on the second attempt."""
        real_begin = db.engine.begin

        call_count = 0

        @contextmanager
        def flaky_begin():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("simulated transient write failure")
            # Delegate to the real connection context manager on second attempt
            with real_begin() as conn:
                yield conn

        with patch.object(db.engine, "begin", side_effect=flaky_begin):
            # create_session uses _execute_with_retry; should succeed on retry
            session_id = db.create_session(hardware, config_snapshot)

        assert isinstance(session_id, int)
        assert call_count == 2  # first attempt failed, second succeeded

    def test_write_returns_none_after_two_failures(
        self, db: Database
    ) -> None:
        """_execute_with_retry() must log and return None when both attempts fail."""
        import sqlalchemy as sa

        @contextmanager
        def always_fail():
            raise OSError("persistent failure")
            yield  # make it a generator

        with patch.object(db.engine, "begin", side_effect=always_fail):
            result = db._execute_with_retry(db.sessions.insert(), {"started_at": "x", "status": "x", "config_json": "x"})

        assert result is None


# ---------------------------------------------------------------------------
# 5. In-memory DB isolation
# ---------------------------------------------------------------------------

class TestInMemoryIsolation:
    """Each test using Database(':memory:') is fully isolated from others."""

    def test_fresh_db_has_no_sessions(self) -> None:
        """A brand-new in-memory DB must contain zero sessions."""
        db = Database(":memory:")
        sessions = db.get_sessions()
        assert sessions == []

    def test_two_memory_dbs_are_independent(
        self, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """Two separate Database(':memory:') instances do not share state."""
        db_a = Database(":memory:")
        db_b = Database(":memory:")

        db_a.create_session(hardware, config_snapshot)

        # db_b must still be empty
        assert db_b.get_sessions() == []


# ---------------------------------------------------------------------------
# 6. Retry-with-n consecutive failures
# ---------------------------------------------------------------------------

class TestRetryWithNConsecutiveFailures:
    """Validates: Requirements 8.6 — After 3 consecutive failures, logs and continues (no exception raised)."""

    def test_save_resource_sample_handles_persistent_failures_gracefully(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """save_resource_sample() uses _execute_with_retry_n() with up to 3 retries.
        
        After 3 failures, the method should log the error and continue without raising an exception.
        This validates Requirement 8.6: "DB write retries up to 3 times on failure; 
        logs and discards on final failure."
        """
        from ollama_benchmark.models import ResourceSample
        from datetime import datetime as dt

        session_id = db.create_session(hardware, config_snapshot)
        model_result = _make_model_result("test_model")
        model_run_id = db.save_model_result(session_id, model_result)

        # Create a mock resource sample
        sample = ResourceSample(
            timestamp=dt.utcnow(),
            cpu_per_core=[25.0, 30.0],
            ram_mb=4096.0,
            gpu_percent=None,
            vram_mb=None,
            cpu_temp_c=None,
            gpu_temp_c=None,
            power_watts=None,
        )

        # Track how many times engine.begin() is called (each attempt calls it once)
        real_begin = db.engine.begin
        attempt_count = 0

        @contextmanager
        def failing_begin():
            nonlocal attempt_count
            attempt_count += 1
            raise OSError("simulated persistent DB failure")
            yield  # unreachable but needed for generator

        # Patch engine.begin to always fail — this simulates a persistent DB write error
        with patch.object(db.engine, "begin", side_effect=failing_begin):
            # save_resource_sample() calls _execute_with_retry_n() with retries=3
            # The method should gracefully handle the failures and NOT raise
            try:
                db.save_resource_sample(model_run_id, sample)
                success = True
            except Exception as exc:
                success = False
                raise AssertionError(f"save_resource_sample() raised {type(exc).__name__}: {exc}. "
                                   "Should gracefully handle persistent failures without raising.")

        # Verify that retries were attempted (should be 3 attempts)
        assert attempt_count >= 3, f"Expected at least 3 retry attempts, got {attempt_count}"


# ---------------------------------------------------------------------------
# 7. Properly typed DTOs returned
# ---------------------------------------------------------------------------

class TestQueryDTOTypes:
    """Validates: Requirements 8.3, 8.4 — Query methods return properly typed DTOs."""

    def test_get_sessions_returns_list_of_session_summary(self, db: Database) -> None:
        """get_sessions() must return a list of SessionSummary objects."""
        from ollama_benchmark.models import SessionSummary

        result = db.get_sessions()

        assert isinstance(result, list)
        if len(result) > 0:
            assert all(isinstance(s, SessionSummary) for s in result)

    def test_get_session_detail_returns_session_detail_type(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """get_session_detail() must return a SessionDetail object."""
        from ollama_benchmark.models import SessionDetail, HardwareInfo

        session_id = db.create_session(hardware, config_snapshot)
        result = db.get_session_detail(session_id)

        assert isinstance(result, SessionDetail)
        assert isinstance(result.hardware, HardwareInfo)
        assert isinstance(result.config_snapshot, dict)
        assert isinstance(result.model_results, list)

    def test_get_model_results_returns_list_of_model_result(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """get_model_results() must return a list of ModelResult objects."""
        session_id = db.create_session(hardware, config_snapshot)
        db.save_model_result(session_id, _make_model_result("test"))

        result = db.get_model_results(session_id)

        assert isinstance(result, list)
        assert all(isinstance(r, ModelResult) for r in result)

    def test_get_resource_timeseries_returns_list_of_resource_sample(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """get_resource_timeseries() must return a list of ResourceSample objects."""
        from ollama_benchmark.models import ResourceSample
        from datetime import datetime as dt

        session_id = db.create_session(hardware, config_snapshot)
        model_result = _make_model_result("test_model")
        model_run_id = db.save_model_result(session_id, model_result)

        # Save a sample
        sample = ResourceSample(
            timestamp=dt.utcnow(),
            cpu_per_core=[25.0],
            ram_mb=2048.0,
            gpu_percent=None,
            vram_mb=None,
            cpu_temp_c=None,
            gpu_temp_c=None,
            power_watts=None,
        )
        db.save_resource_sample(model_run_id, sample)

        result = db.get_resource_timeseries(model_run_id)

        assert isinstance(result, list)
        if len(result) > 0:
            assert all(isinstance(s, ResourceSample) for s in result)

    def test_get_prompt_results_returns_list_of_prompt_result(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """get_prompt_results() must return a list of PromptResult objects."""
        from ollama_benchmark.models import PromptResult

        session_id = db.create_session(hardware, config_snapshot)
        model_result = _make_model_result("test_model")
        model_run_id = db.save_model_result(session_id, model_result)

        # Create and save an inference result
        inference = InferenceResult(
            prompt_text="What is 2+2?",
            response_text="The answer is 4.",
            ttft_ms=50.0,
            total_response_ms=150.0,
            tokens_generated=5,
            tokens_per_second=33.33,
            avg_inter_token_ms=30.0,
            timed_out=False,
            error=None,
        )
        db.save_inference_result(model_run_id, inference, category="math")

        result = db.get_prompt_results(model_run_id)

        assert isinstance(result, list)
        if len(result) > 0:
            assert all(isinstance(p, PromptResult) for p in result)


# ---------------------------------------------------------------------------
# 8. Timestamps in session records
# ---------------------------------------------------------------------------

class TestSessionTimestamps:
    """Validates: Requirements 8.2 — Session records include timestamps."""

    def test_create_session_sets_started_at(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """create_session() must set started_at timestamp."""
        session_id = db.create_session(hardware, config_snapshot)
        detail = db.get_session_detail(session_id)

        assert detail.summary.started_at is not None

    def test_finalize_session_sets_finished_at(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """finalize_session() must set finished_at timestamp."""
        session_id = db.create_session(hardware, config_snapshot)
        db.finalize_session(session_id, status="completed")

        detail = db.get_session_detail(session_id)

        assert detail.summary.finished_at is not None
        assert detail.summary.status == "completed"

    def test_session_timestamps_ordered_correctly(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """finished_at should be >= started_at (time moves forward)."""
        import time

        session_id = db.create_session(hardware, config_snapshot)
        time.sleep(0.01)  # Small delay to ensure timestamps differ
        db.finalize_session(session_id)

        detail = db.get_session_detail(session_id)

        assert detail.summary.finished_at >= detail.summary.started_at


# ---------------------------------------------------------------------------
# 9. Complete ModelResult persistence
# ---------------------------------------------------------------------------

class TestCompleteModelResultPersistence:
    """Validates: Requirements 8.3 — save_model_result persists complete ModelResult to DB."""

    def test_save_inference_result_persists_prompt_data(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """save_inference_result() must persist prompt results to the database."""
        session_id = db.create_session(hardware, config_snapshot)
        model_result = _make_model_result("test_model")
        model_run_id = db.save_model_result(session_id, model_result)

        # Save multiple inference results via save_inference_result()
        inference1 = InferenceResult(
            prompt_text="Question 1",
            response_text="Answer 1",
            ttft_ms=100.0,
            total_response_ms=500.0,
            tokens_generated=50,
            tokens_per_second=100.0,
            avg_inter_token_ms=10.0,
            timed_out=False,
            error=None,
        )
        inference2 = InferenceResult(
            prompt_text="Question 2",
            response_text="Answer 2",
            ttft_ms=150.0,
            total_response_ms=600.0,
            tokens_generated=60,
            tokens_per_second=100.0,
            avg_inter_token_ms=10.0,
            timed_out=False,
            error=None,
        )

        db.save_inference_result(model_run_id, inference1, category="math")
        db.save_inference_result(model_run_id, inference2, category="reasoning")

        # Retrieve and verify via get_prompt_results()
        prompt_results = db.get_prompt_results(model_run_id)
        assert len(prompt_results) == 2
        assert prompt_results[0].prompt_text == "Question 1"
        assert prompt_results[1].prompt_text == "Question 2"
        assert prompt_results[0].category == "math"
        assert prompt_results[1].category == "reasoning"

    def test_save_model_result_with_efficiency_indices(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """save_model_result() must persist EfficiencyIndices when present."""
        session_id = db.create_session(hardware, config_snapshot)

        result = _make_model_result("test_model")
        result.efficiency_indices = EfficiencyIndices(
            quality_per_ram=0.5,
            quality_per_latency=0.6,
            quality_per_cpu=0.7,
            quality_per_disk=0.8,
            tps_per_gb_ram=0.9,
            quality_per_energy=None,
            norm_quality_per_ram=0.5,
            norm_quality_per_latency=0.6,
            norm_quality_per_cpu=0.7,
            norm_quality_per_disk=0.8,
            norm_tps_per_gb_ram=0.9,
            norm_quality_per_energy=None,
        )

        model_run_id = db.save_model_result(session_id, result)

        # Retrieve and verify efficiency indices were saved
        model_results = db.get_model_results(session_id)
        assert model_results[0].efficiency_indices is not None
        assert model_results[0].efficiency_indices.quality_per_ram == pytest.approx(0.5)

    def test_save_model_result_with_quality_scores(
        self, db: Database, hardware: HardwareInfo, config_snapshot: dict
    ) -> None:
        """save_model_result() must persist quality_scores for each category."""
        session_id = db.create_session(hardware, config_snapshot)

        result = _make_model_result("test_model")
        result.quality_scores = {
            "reasoning": 0.85,
            "math": 0.92,
            "coding": 0.78,
        }

        db.save_model_result(session_id, result)

        model_results = db.get_model_results(session_id)
        assert len(model_results[0].quality_scores) == 3
        assert model_results[0].quality_scores.get("reasoning") == pytest.approx(0.85)
        assert model_results[0].quality_scores.get("math") == pytest.approx(0.92)
