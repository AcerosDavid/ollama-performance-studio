"""
Resource Monitor for ollama-performance-studio.

Runs a background daemon thread that samples system resource metrics every
second while a model is under evaluation.  Metrics are collected via:

  - ``psutil``          — CPU per-core %, RAM usage, temperatures (where available)
  - ``pynvml``          — NVIDIA GPU utilisation, VRAM, GPU temperature, power (optional)

Both optional libraries degrade gracefully:
  - ``psutil.sensors_temperatures`` is not available on Windows → temperature
    fields are ``None`` and a single warning is logged at first call.
  - ``pynvml`` may not be installed → all GPU fields are ``None`` and a single
    warning is logged at initialisation time.

Usage
-----
    monitor = ResourceMonitor()
    monitor.start(session_id=1, model_name="llama3")
    ...run inference...
    summary = monitor.stop()   # stop() implemented in task 5.2

Requirements satisfied: 4.1, 4.2, 4.3, 4.6, 4.7
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import List, Optional

import psutil

from ollama_benchmark.models import ResourceSample, ResourceSummary

# ---------------------------------------------------------------------------
# Optional pynvml import — graceful fallback if not installed or unavailable
# ---------------------------------------------------------------------------

try:
    import pynvml  # type: ignore

    _pynvml_available: bool = True
except ImportError:
    pynvml = None  # type: ignore
    _pynvml_available = False

__all__ = ["ResourceMonitor"]

_log = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Background-thread resource sampler.

    Call ``start(session_id, model_name)`` to begin sampling and
    ``stop()`` to halt the thread and collect aggregated results.

    The instance is designed to be reused across multiple model runs:
    each ``start`` / ``stop`` pair is an independent monitoring session.
    """

    def __init__(self) -> None:
        self._samples: List[ResourceSample] = []
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._session_id: Optional[int] = None
        self._model_name: Optional[str] = None

        # Consecutive primary-metric failure counter (req 4.8)
        self._consecutive_primary_failures: int = 0
        _PRIMARY_FAILURE_THRESHOLD = 3  # log after this many consecutive failures

        # Track one-time log flags so we don't spam the log on every sample
        self._temp_unavailable_logged: bool = False
        self._gpu_unavailable_logged: bool = False

        # Counter to reduce logging frequency (log every 10 samples instead of every second)
        self._sample_count: int = 0
        _LOG_INTERVAL = 10  # log resources every 10 samples (10 seconds)

        # Attempt pynvml initialisation once at construction so that the
        # "GPU not available" message is logged a single time (req 4.7).
        self._gpu_handle = None
        self._init_gpu()

    # ------------------------------------------------------------------
    # GPU initialisation (called once at construction)
    # ------------------------------------------------------------------

    def _init_gpu(self) -> None:
        """Attempt to initialise pynvml and cache the device handle for GPU 0."""
        if not _pynvml_available:
            _log.warning(
                "pynvml is not installed — GPU metrics (GPU %%, VRAM, "
                "temperature, power) will not be collected."
            )
            self._gpu_unavailable_logged = True
            return

        try:
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _log.debug("pynvml initialised successfully — GPU 0 handle acquired.")
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "GPU monitoring unavailable — pynvml initialisation failed: %s. "
                "GPU metrics will not be collected.",
                exc,
            )
            self._gpu_handle = None
            self._gpu_unavailable_logged = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, session_id: int, model_name: str) -> None:
        """
        Start background resource sampling.

        Launches a daemon thread that calls ``_sample()`` every second and
        appends results to ``_samples``.  The thread is a daemon so it will
        not prevent interpreter shutdown if ``stop()`` is never called.

        Parameters
        ----------
        session_id:
            Identifier of the current benchmark session (for context in logs).
        model_name:
            Name of the model being benchmarked (for context in logs).
        """
        if self._running:
            _log.warning(
                "ResourceMonitor.start() called while already running "
                "(session_id=%s, model=%s). Ignoring.",
                session_id,
                model_name,
            )
            return

        self._session_id = session_id
        self._model_name = model_name
        self._samples = []
        self._consecutive_primary_failures = 0
        self._running = True

        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"resource-monitor-{model_name}",
            daemon=True,
        )
        self._thread.start()
        _log.debug(
            "ResourceMonitor started (session_id=%s, model=%s).",
            session_id,
            model_name,
        )

    def stop(self, db=None, model_run_id: Optional[int] = None) -> ResourceSummary:
        """
        Stop background sampling, persist samples to DB, and return aggregated summary.

        Sets ``_running = False``, joins the background thread (timeout=5 s), then
        computes max/avg aggregates from the collected samples.  If *db* is provided
        and *model_run_id* is not ``None``, each sample is written to the database via
        ``db.save_resource_sample``; the Database class handles its own 3-retry logic
        internally, so failures are logged and discarded there without propagating here.

        Returns an empty ``ResourceSummary`` (zeros/Nones, empty sample list) when no
        samples were collected.

        Parameters
        ----------
        db:
            Optional ``Database`` instance.  When provided together with
            *model_run_id*, each sample is persisted.
        model_run_id:
            The model-run primary key to associate resource samples with.

        Returns
        -------
        ResourceSummary
            Aggregated metrics plus the full time-series sample list.
        """
        # 1. Signal the loop to stop and join the thread.
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _log.warning(
                    "ResourceMonitor background thread did not finish within the "
                    "join timeout (session=%s, model=%s) — continuing anyway.",
                    self._session_id,
                    self._model_name,
                )
            self._thread = None

        # 2. Persist each sample when a DB handle and run ID are supplied.
        if db is not None and model_run_id is not None:
            for sample in self._samples:
                try:
                    db.save_resource_sample(model_run_id, sample)
                except Exception as exc:  # noqa: BLE001
                    # Database.save_resource_sample already retries 3× and logs;
                    # this outer catch handles any unexpected propagation.
                    _log.error(
                        "Unexpected error persisting resource sample "
                        "(model_run_id=%s): %s — sample discarded.",
                        model_run_id,
                        exc,
                    )

        # 3. Compute aggregates.  Return an empty summary when no samples exist.
        if not self._samples:
            return ResourceSummary(
                max_cpu_percent=0.0,
                avg_cpu_percent=0.0,
                max_ram_mb=0.0,
                avg_ram_mb=0.0,
                max_gpu_percent=None,
                avg_gpu_percent=None,
                max_vram_mb=None,
                avg_vram_mb=None,
                max_temp_cpu_c=None,
                max_temp_gpu_c=None,
                avg_power_watts=None,
                samples=[],
            )

        # Helper: mean of per-core CPU values for a single sample.
        def _mean_cpu(sample: ResourceSample) -> float:
            values = [v for v in sample.cpu_per_core if v is not None]
            return sum(values) / len(values) if values else 0.0

        cpu_means = [_mean_cpu(s) for s in self._samples]
        max_cpu_percent = max(cpu_means)
        avg_cpu_percent = sum(cpu_means) / len(cpu_means)

        ram_values = [s.ram_mb for s in self._samples]
        max_ram_mb = max(ram_values)
        avg_ram_mb = sum(ram_values) / len(ram_values)

        # Optional metrics — only aggregate non-None values.
        gpu_values = [s.gpu_percent for s in self._samples if s.gpu_percent is not None]
        max_gpu_percent: Optional[float] = max(gpu_values) if gpu_values else None
        avg_gpu_percent: Optional[float] = sum(gpu_values) / len(gpu_values) if gpu_values else None

        vram_values = [s.vram_mb for s in self._samples if s.vram_mb is not None]
        max_vram_mb: Optional[float] = max(vram_values) if vram_values else None
        avg_vram_mb: Optional[float] = sum(vram_values) / len(vram_values) if vram_values else None

        cpu_temp_values = [s.cpu_temp_c for s in self._samples if s.cpu_temp_c is not None]
        max_temp_cpu_c: Optional[float] = max(cpu_temp_values) if cpu_temp_values else None

        gpu_temp_values = [s.gpu_temp_c for s in self._samples if s.gpu_temp_c is not None]
        max_temp_gpu_c: Optional[float] = max(gpu_temp_values) if gpu_temp_values else None

        power_values = [s.power_watts for s in self._samples if s.power_watts is not None]
        avg_power_watts: Optional[float] = sum(power_values) / len(power_values) if power_values else None

        return ResourceSummary(
            max_cpu_percent=max_cpu_percent,
            avg_cpu_percent=avg_cpu_percent,
            max_ram_mb=max_ram_mb,
            avg_ram_mb=avg_ram_mb,
            max_gpu_percent=max_gpu_percent,
            avg_gpu_percent=avg_gpu_percent,
            max_vram_mb=max_vram_mb,
            avg_vram_mb=avg_vram_mb,
            max_temp_cpu_c=max_temp_cpu_c,
            max_temp_gpu_c=max_temp_gpu_c,
            avg_power_watts=avg_power_watts,
            samples=self._samples,
        )

    # ------------------------------------------------------------------
    # Internal sampling loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """
        Main loop executed in the background thread.

        Calls ``_sample()`` on each iteration, appends the result to
        ``_samples``, then sleeps for one second.  Any exception raised
        by ``_sample()`` is caught and logged so that a single bad sample
        never terminates the thread.
        """
        _PRIMARY_FAILURE_THRESHOLD = 3

        while self._running:
            try:
                sample = self._sample()
                self._samples.append(sample)
                # Reset consecutive failure counter on success
                self._consecutive_primary_failures = 0
            except _PrimaryMetricError as exc:
                # CPU or RAM sampling failed — track consecutive failures
                self._consecutive_primary_failures += 1
                if self._consecutive_primary_failures >= _PRIMARY_FAILURE_THRESHOLD:
                    _log.error(
                        "Primary metric (CPU/RAM) sampling failed %d consecutive "
                        "times (session=%s, model=%s): %s — continuing.",
                        self._consecutive_primary_failures,
                        self._session_id,
                        self._model_name,
                        exc,
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "Unexpected error during resource sampling "
                    "(session=%s, model=%s): %s — sample discarded.",
                    self._session_id,
                    self._model_name,
                    exc,
                )

            time.sleep(1)

    # ------------------------------------------------------------------
    # Core sampling method
    # ------------------------------------------------------------------

    def _sample(self) -> ResourceSample:
        """
        Collect one snapshot of system resource metrics.

        Returns
        -------
        ResourceSample
            A single timestamped reading of all available metrics.

        Raises
        ------
        _PrimaryMetricError
            If CPU or RAM sampling fails.  The caller (``_run_loop``) tracks
            consecutive failures per requirement 4.8.
        """
        timestamp = datetime.utcnow()

        # ------------------------------------------------------------------
        # Primary metrics — CPU per-core % and RAM (req 4.1)
        # ------------------------------------------------------------------
        try:
            cpu_per_core: List[float] = psutil.cpu_percent(percpu=True)  # type: ignore[assignment]
            ram_mb: float = psutil.virtual_memory().used / (1024.0 * 1024.0)
        except Exception as exc:  # noqa: BLE001
            raise _PrimaryMetricError(
                f"Failed to read CPU/RAM metrics: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Temperature — CPU (req 4.2)
        # ------------------------------------------------------------------
        cpu_temp_c: Optional[float] = self._read_cpu_temperature()

        # ------------------------------------------------------------------
        # GPU metrics via pynvml (req 4.3, 4.6, 4.7)
        # ------------------------------------------------------------------
        gpu_percent: Optional[float] = None
        vram_mb: Optional[float] = None
        gpu_temp_c: Optional[float] = None
        power_watts: Optional[float] = None

        if self._gpu_handle is not None:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_percent = float(utilization.gpu)
            except Exception as exc:  # noqa: BLE001
                _log.debug("GPU utilisation read failed: %s", exc)
                gpu_percent = None

            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_mb = mem_info.used / (1024.0 * 1024.0)
            except Exception as exc:  # noqa: BLE001
                _log.debug("VRAM read failed: %s", exc)
                vram_mb = None

            try:
                gpu_temp_c = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._gpu_handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                _log.debug("GPU temperature read failed: %s", exc)
                gpu_temp_c = None

            try:
                # nvmlDeviceGetPowerUsage returns milliwatts → convert to watts
                power_watts = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0
            except Exception as exc:  # noqa: BLE001
                _log.debug("GPU power read failed: %s", exc)
                power_watts = None

        # Log resource metrics in verbose mode (every 10 samples to avoid spamming)
        self._sample_count += 1
        if self._sample_count % 10 == 0:  # Log every 10 samples (10 seconds)
            cpu_avg = sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0.0
            gpu_str = f"{gpu_percent:.1f}%" if gpu_percent is not None else "N/A"
            vram_str = f"{vram_mb:.1f}MB" if vram_mb is not None else "N/A"
            _log.debug("[RESOURCES] CPU: %.1f%% RAM: %.1fMB GPU: %s VRAM: %s", cpu_avg, ram_mb, gpu_str, vram_str)

        return ResourceSample(
            timestamp=timestamp,
            cpu_per_core=cpu_per_core,
            ram_mb=ram_mb,
            gpu_percent=gpu_percent,
            vram_mb=vram_mb,
            cpu_temp_c=cpu_temp_c,
            gpu_temp_c=gpu_temp_c,
            power_watts=power_watts,
        )

    # ------------------------------------------------------------------
    # Temperature helper
    # ------------------------------------------------------------------

    def _read_cpu_temperature(self) -> Optional[float]:
        """
        Attempt to read the CPU temperature in degrees Celsius.

        Tries ``psutil.sensors_temperatures()`` and looks for common
        hardware sensor keys: ``coretemp`` (Intel), ``k10temp`` (AMD),
        ``cpu-thermal`` (ARM/embedded).

        Returns ``None`` on any failure, logging the unavailability once
        (requirement 4.6).
        """
        try:
            sensors = psutil.sensors_temperatures()  # type: ignore[attr-defined]
        except AttributeError:
            # Windows / platform without temperature support
            if not self._temp_unavailable_logged:
                _log.warning(
                    "psutil.sensors_temperatures() is not available on this "
                    "platform — CPU temperature will not be collected."
                )
                self._temp_unavailable_logged = True
            return None
        except Exception as exc:  # noqa: BLE001
            if not self._temp_unavailable_logged:
                _log.warning(
                    "CPU temperature monitoring unavailable: %s — "
                    "temperature metrics will not be collected.",
                    exc,
                )
                self._temp_unavailable_logged = True
            return None

        if not sensors:
            if not self._temp_unavailable_logged:
                _log.warning(
                    "No temperature sensors found via psutil — "
                    "CPU temperature will not be collected."
                )
                self._temp_unavailable_logged = True
            return None

        # Try known CPU sensor keys in order of preference
        for key in ("coretemp", "k10temp", "cpu-thermal"):
            entries = sensors.get(key)
            if entries:
                # Take the first entry's current reading
                return float(entries[0].current)

        # Fallback: use the first available sensor that looks CPU-related
        for key, entries in sensors.items():
            if entries and "cpu" in key.lower():
                return float(entries[0].current)

        # Last resort: just use the first sensor available
        for entries in sensors.values():
            if entries:
                return float(entries[0].current)

        return None


# ---------------------------------------------------------------------------
# Internal exception — signals primary metric failure to _run_loop
# ---------------------------------------------------------------------------


class _PrimaryMetricError(Exception):
    """Raised when CPU or RAM sampling fails inside ``_sample()``."""
