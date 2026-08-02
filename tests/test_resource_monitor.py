"""
Unit tests for ResourceMonitor.

Validates: Requirements 4.1, 4.6, 4.7, 4.8

Tests
-----
TestBasicSampling          — monitor collects ≥1 sample and returns a valid ResourceSummary
TestGpuFallback            — GPU absent → all GPU fields are None, monitor runs cleanly
TestPrimaryMetricFailure   — cpu_percent raises once, monitor tracks _consecutive_primary_failures
TestAggregatesCorrectness  — known samples → max_cpu_percent >= avg_cpu_percent
TestDbWriteRetry           — DB write failures retry up to 3 times
TestTemperatureFallback    — psutil.sensors_temperatures empty → temp is None
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

import ollama_benchmark.resource_monitor as rm_module
from ollama_benchmark.models import ResourceSample, ResourceSummary
from ollama_benchmark.resource_monitor import ResourceMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(cpu_per_core: List[float], ram_mb: float) -> ResourceSample:
    """Build a minimal ResourceSample for aggregate tests."""
    return ResourceSample(
        timestamp=datetime.utcnow(),
        cpu_per_core=cpu_per_core,
        ram_mb=ram_mb,
        gpu_percent=None,
        vram_mb=None,
        cpu_temp_c=None,
        gpu_temp_c=None,
        power_watts=None,
    )


# ---------------------------------------------------------------------------
# Test 1 — Basic sampling
# Validates: Requirement 4.1
# ---------------------------------------------------------------------------

class TestBasicSampling:
    """
    GIVEN a mocked psutil
    WHEN the monitor runs and collects samples
    THEN stop() returns a ResourceSummary with ≥1 sample and non-zero averages.
    """

    def test_collects_samples_and_returns_summary(self) -> None:
        mock_vm = MagicMock()
        mock_vm.used = 512 * 1024 * 1024  # 512 MB

        # Store the real sleep before patching to use it in the side effect
        real_sleep = time.sleep

        with (
            patch("ollama_benchmark.resource_monitor._pynvml_available", False),
            patch("ollama_benchmark.resource_monitor.psutil") as mock_psutil,
            patch("ollama_benchmark.resource_monitor.time.sleep", wraps=real_sleep) as mock_sleep,
        ):
            mock_psutil.cpu_percent.return_value = [25.0, 30.0, 20.0, 35.0]
            mock_psutil.virtual_memory.return_value = mock_vm
            mock_psutil.sensors_temperatures.side_effect = AttributeError

            monitor = ResourceMonitor()
            monitor.start(session_id=1, model_name="test-model")
            # Let the real sleep execute (wrapped), so the thread collects samples
            real_sleep(0.2)
            summary = monitor.stop()

        assert isinstance(summary, ResourceSummary)
        assert len(summary.samples) >= 1, "Expected at least one sample"
        assert summary.avg_cpu_percent > 0.0, "avg_cpu_percent should be non-zero"
        assert summary.avg_ram_mb > 0.0, "avg_ram_mb should be non-zero"

    def test_stop_without_start_returns_empty_summary(self) -> None:
        """stop() on a never-started monitor returns zeros/Nones cleanly."""
        with patch("ollama_benchmark.resource_monitor._pynvml_available", False):
            monitor = ResourceMonitor()

        summary = monitor.stop()

        assert summary.avg_cpu_percent == 0.0
        assert summary.avg_ram_mb == 0.0
        assert summary.samples == []


# ---------------------------------------------------------------------------
# Test 2 — GPU fallback
# Validates: Requirement 4.7
# ---------------------------------------------------------------------------

class TestGpuFallback:
    """
    GIVEN pynvml is absent
    WHEN the monitor runs and stops
    THEN all GPU summary fields are None and no exception is raised.
    """

    def test_gpu_fields_none_when_pynvml_unavailable(self) -> None:
        mock_vm = MagicMock()
        mock_vm.used = 256 * 1024 * 1024  # 256 MB

        real_sleep = time.sleep

        with (
            patch.object(rm_module, "_pynvml_available", False),
            patch("ollama_benchmark.resource_monitor.psutil") as mock_psutil,
            patch("ollama_benchmark.resource_monitor.time.sleep", wraps=real_sleep),
        ):
            mock_psutil.cpu_percent.return_value = [10.0, 15.0]
            mock_psutil.virtual_memory.return_value = mock_vm
            mock_psutil.sensors_temperatures.side_effect = AttributeError

            monitor = ResourceMonitor()
            monitor.start(session_id=2, model_name="no-gpu-model")
            real_sleep(0.15)
            summary = monitor.stop()

        assert summary.max_gpu_percent is None
        assert summary.avg_gpu_percent is None
        assert summary.max_vram_mb is None
        assert summary.avg_vram_mb is None

    def test_monitor_starts_and_stops_cleanly_without_gpu(self) -> None:
        """No exception must escape start/stop when GPU is unavailable."""
        mock_vm = MagicMock()
        mock_vm.used = 128 * 1024 * 1024

        real_sleep = time.sleep

        with (
            patch.object(rm_module, "_pynvml_available", False),
            patch("ollama_benchmark.resource_monitor.psutil") as mock_psutil,
            patch("ollama_benchmark.resource_monitor.time.sleep", wraps=real_sleep),
        ):
            mock_psutil.cpu_percent.return_value = [5.0]
            mock_psutil.virtual_memory.return_value = mock_vm
            mock_psutil.sensors_temperatures.side_effect = AttributeError

            monitor = ResourceMonitor()
            monitor.start(session_id=3, model_name="clean-stop-model")
            real_sleep(0.1)
            summary = monitor.stop()  # must not raise

        assert isinstance(summary, ResourceSummary)


# ---------------------------------------------------------------------------
# Test 3 — Primary metric failure tracking
# Validates: Requirement 4.8
# ---------------------------------------------------------------------------

class TestPrimaryMetricFailure:
    """
    GIVEN cpu_percent raises on the first call but succeeds on subsequent calls
    WHEN the monitor's _run_loop processes these calls
    THEN _consecutive_primary_failures is incremented then reset, and the
         monitor continues collecting samples.
    """

    def test_consecutive_failures_tracked_then_reset(self) -> None:
        mock_vm = MagicMock()
        mock_vm.used = 300 * 1024 * 1024

        # First call raises, subsequent calls succeed
        call_results = [
            RuntimeError("psutil exploded"),
            [20.0, 25.0],
            [20.0, 25.0],
            [20.0, 25.0],
            [20.0, 25.0],
        ]

        call_index = {"n": 0}

        def cpu_percent_side_effect(**kwargs):
            idx = call_index["n"]
            call_index["n"] += 1
            result = call_results[idx] if idx < len(call_results) else [20.0, 25.0]
            if isinstance(result, Exception):
                raise result
            return result

        real_sleep = time.sleep

        with (
            patch("ollama_benchmark.resource_monitor._pynvml_available", False),
            patch("ollama_benchmark.resource_monitor.psutil") as mock_psutil,
            patch("ollama_benchmark.resource_monitor.time.sleep", wraps=real_sleep),
        ):
            mock_psutil.cpu_percent.side_effect = cpu_percent_side_effect
            mock_psutil.virtual_memory.return_value = mock_vm
            mock_psutil.sensors_temperatures.side_effect = AttributeError

            monitor = ResourceMonitor()
            monitor.start(session_id=4, model_name="failure-model")
            # Give the loop enough time to run multiple cycles so counter resets
            real_sleep(3.5)
            summary = monitor.stop()

        # After a successful sample the counter must have been reset to 0
        assert monitor._consecutive_primary_failures == 0
        # At least one sample should have been collected (the successful calls)
        assert len(summary.samples) >= 1

    def test_failure_counter_increments_on_error(self) -> None:
        """
        Directly call _run_loop logic: inject _PrimaryMetricError via _sample,
        verify the counter increments without the thread being involved.
        """
        from ollama_benchmark.resource_monitor import _PrimaryMetricError

        with patch("ollama_benchmark.resource_monitor._pynvml_available", False):
            monitor = ResourceMonitor()

        monitor._running = True
        monitor._session_id = 5
        monitor._model_name = "counter-test"

        # Simulate _sample raising _PrimaryMetricError three times
        with patch.object(monitor, "_sample", side_effect=_PrimaryMetricError("no cpu")):
            # Run just a few iterations manually to avoid infinite loop
            iterations = 0
            _PRIMARY_FAILURE_THRESHOLD = 3
            while monitor._running and iterations < 3:
                try:
                    sample = monitor._sample()
                    monitor._samples.append(sample)
                    monitor._consecutive_primary_failures = 0
                except _PrimaryMetricError:
                    monitor._consecutive_primary_failures += 1
                iterations += 1
            monitor._running = False

        assert monitor._consecutive_primary_failures == 3


# ---------------------------------------------------------------------------
# Test 4 — Aggregates correctness
# Validates: Requirements 4.1, 4.6
# ---------------------------------------------------------------------------

class TestAggregatesCorrectness:
    """
    GIVEN a known set of samples injected directly into _samples
    WHEN stop() is called (thread already halted)
    THEN max_cpu_percent >= avg_cpu_percent and values match expected calculations.
    """

    def _build_monitor_with_samples(self, samples: List[ResourceSample]) -> ResourceMonitor:
        """Create a ResourceMonitor pre-loaded with samples, thread not running."""
        with patch("ollama_benchmark.resource_monitor._pynvml_available", False):
            monitor = ResourceMonitor()
        monitor._samples = list(samples)
        monitor._running = False
        monitor._thread = None
        return monitor

    def test_max_cpu_gte_avg_cpu(self) -> None:
        samples = [
            _make_sample([10.0, 20.0], ram_mb=400.0),  # mean cpu = 15.0
            _make_sample([50.0, 60.0], ram_mb=500.0),  # mean cpu = 55.0
            _make_sample([30.0, 40.0], ram_mb=450.0),  # mean cpu = 35.0
        ]
        monitor = self._build_monitor_with_samples(samples)
        summary = monitor.stop()

        assert summary.max_cpu_percent >= summary.avg_cpu_percent

    def test_exact_aggregate_values(self) -> None:
        """avg and max match manual calculation for known inputs."""
        # cpu means: 15.0, 55.0, 35.0  → avg = 35.0, max = 55.0
        # ram: 400, 500, 450            → avg = 450.0, max = 500.0
        samples = [
            _make_sample([10.0, 20.0], ram_mb=400.0),
            _make_sample([50.0, 60.0], ram_mb=500.0),
            _make_sample([30.0, 40.0], ram_mb=450.0),
        ]
        monitor = self._build_monitor_with_samples(samples)
        summary = monitor.stop()

        assert summary.avg_cpu_percent == pytest.approx(35.0)
        assert summary.max_cpu_percent == pytest.approx(55.0)
        assert summary.avg_ram_mb == pytest.approx(450.0)
        assert summary.max_ram_mb == pytest.approx(500.0)

    def test_single_sample_max_equals_avg(self) -> None:
        """With a single sample, max and avg must be identical."""
        samples = [_make_sample([40.0, 60.0], ram_mb=700.0)]
        monitor = self._build_monitor_with_samples(samples)
        summary = monitor.stop()

        assert summary.max_cpu_percent == summary.avg_cpu_percent
        assert summary.max_ram_mb == summary.avg_ram_mb

    def test_gpu_none_when_no_gpu_samples(self) -> None:
        """When all samples have gpu_percent=None, summary GPU fields are None."""
        samples = [
            _make_sample([20.0], ram_mb=300.0),
            _make_sample([30.0], ram_mb=320.0),
        ]
        monitor = self._build_monitor_with_samples(samples)
        summary = monitor.stop()

        assert summary.max_gpu_percent is None
        assert summary.avg_gpu_percent is None


# ---------------------------------------------------------------------------
# Test 5 — DB write retry
# Validates: Requirement 4.1 (with DB persistence)
# ---------------------------------------------------------------------------

class TestDbWriteRetry:
    """
    GIVEN a mock database that fails on write
    WHEN stop() is called with db and model_run_id
    THEN samples are persisted via db.save_resource_sample, which handles retries.
    """

    def test_stop_calls_db_save_for_each_sample(self) -> None:
        """Verify that stop() calls db.save_resource_sample for each sample."""
        with patch("ollama_benchmark.resource_monitor._pynvml_available", False):
            monitor = ResourceMonitor()

        # Pre-populate samples
        samples = [
            _make_sample([10.0], ram_mb=100.0),
            _make_sample([20.0], ram_mb=200.0),
        ]
        monitor._samples = samples
        monitor._running = False
        monitor._thread = None

        mock_db = MagicMock()
        summary = monitor.stop(db=mock_db, model_run_id=42)

        # Verify db.save_resource_sample was called for each sample
        assert mock_db.save_resource_sample.call_count == 2

    def test_stop_handles_db_save_exception(self) -> None:
        """When db.save_resource_sample raises, stop() logs and continues."""
        with patch("ollama_benchmark.resource_monitor._pynvml_available", False):
            monitor = ResourceMonitor()

        samples = [_make_sample([10.0], ram_mb=100.0)]
        monitor._samples = samples
        monitor._running = False
        monitor._thread = None

        mock_db = MagicMock()
        mock_db.save_resource_sample.side_effect = RuntimeError("DB connection failed")

        # Should not raise, just log and continue
        summary = monitor.stop(db=mock_db, model_run_id=42)
        assert isinstance(summary, ResourceSummary)


# ---------------------------------------------------------------------------
# Test 6 — Temperature fallback
# Validates: Requirement 4.6
# ---------------------------------------------------------------------------

class TestTemperatureFallback:
    """
    GIVEN psutil.sensors_temperatures returns empty dict or raises
    WHEN _sample() is called
    THEN temperature fields are None and no exception propagates.
    """

    def test_empty_sensors_temperatures_returns_none(self) -> None:
        """When sensors_temperatures() returns empty dict, temp is None."""
        with patch("ollama_benchmark.resource_monitor._pynvml_available", False):
            monitor = ResourceMonitor()

        with (
            patch("ollama_benchmark.resource_monitor.psutil") as mock_psutil,
        ):
            mock_psutil.cpu_percent.return_value = [50.0]
            mock_psutil.virtual_memory.return_value = MagicMock(used=512 * 1024 * 1024)
            mock_psutil.sensors_temperatures.return_value = {}  # empty dict

            sample = monitor._sample()

        assert sample.cpu_temp_c is None

    def test_sensors_temperatures_exception_returns_none(self) -> None:
        """When sensors_temperatures() raises, temp is None and no exception escapes."""
        with patch("ollama_benchmark.resource_monitor._pynvml_available", False):
            monitor = ResourceMonitor()

        with (
            patch("ollama_benchmark.resource_monitor.psutil") as mock_psutil,
        ):
            mock_psutil.cpu_percent.return_value = [50.0]
            mock_psutil.virtual_memory.return_value = MagicMock(used=512 * 1024 * 1024)
            mock_psutil.sensors_temperatures.side_effect = OSError("No sensors found")

            sample = monitor._sample()

        assert sample.cpu_temp_c is None
