"""
Benchmark Runner for ollama-performance-studio.
Orchestrates the complete evaluation pipeline for a benchmark session.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import sys
from typing import Optional

import psutil

from ollama_benchmark.config import BenchmarkConfig
from ollama_benchmark.database import Database
from ollama_benchmark.models import (
    ErrorEntry,
    HardwareInfo,
    ModelResult,
    OllamaUnavailableError,
    RobustnessMetrics,
    SessionResult,
)

logger = logging.getLogger(__name__)

# Optional pynvml — GPU detection degrades gracefully when not installed
try:
    import pynvml  # type: ignore[import-untyped]
    _pynvml_available = True
except ImportError:
    pynvml = None  # type: ignore[assignment]
    _pynvml_available = False


class BenchmarkRunner:
    """Central orchestrator for a full benchmark session.

    Wires together all domain components (Model_Manager, Resource_Monitor,
    Inference_Engine, Quality_Evaluator, Score_Engine) and drives the
    sequential per-model evaluation loop.
    """

    def __init__(self, config: BenchmarkConfig, db: Database) -> None:
        self._config = config
        self._db = db

    # ------------------------------------------------------------------
    # Hardware detection (Requirements 1.1, 1.2, 1.3, 1.4)
    # ------------------------------------------------------------------

    def _detect_hardware(self) -> HardwareInfo:
        """Detect and return system hardware characteristics.

        GPU/VRAM detection is attempted via pynvml.  If pynvml is not
        installed, or if any pynvml call fails (no NVIDIA GPU, driver
        mismatch, etc.), ``has_gpu`` is set to ``False`` and ``vram_mb``
        is recorded as ``0.0`` with a warning — per Requirement 1.2.

        Raises
        ------
        OllamaUnavailableError
            Propagated from ``_check_ollama_version`` when Ollama is not
            running or not found in PATH — per Requirement 1.3.
        """
        os_name = platform.platform()
        python_version = (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )

        # CPU — Requirement 1.1
        cpu_cores: int = psutil.cpu_count(logical=True) or 1
        cpu_freq = psutil.cpu_freq()
        cpu_mhz: float = float(cpu_freq.max) if cpu_freq else 0.0

        # RAM total (not current usage) — Requirement 1.1
        ram_mb: float = psutil.virtual_memory().total / (1024 * 1024)

        # Disk free on the volume where the tool is running — Requirement 1.1
        disk = psutil.disk_usage(".")
        disk_free_mb: float = disk.free / (1024 * 1024)

        # GPU via pynvml — Requirement 1.1 / 1.2
        has_gpu = False
        vram_mb = 0.0
        if _pynvml_available:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_mb = mem.total / (1024 * 1024)
                has_gpu = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "GPU detection failed: %s — VRAM recorded as 0 MB", exc
                )
        else:
            logger.warning("pynvml not available — VRAM recorded as 0 MB")

        # Ollama version — Requirement 1.3 / 1.4
        # Raises OllamaUnavailableError on failure → caller exits non-zero.
        ollama_version = self._check_ollama_version()

        return HardwareInfo(
            os=os_name,
            python_version=python_version,
            ollama_version=ollama_version,
            cpu_cores=cpu_cores,
            cpu_mhz=cpu_mhz,
            ram_mb=ram_mb,
            has_gpu=has_gpu,
            vram_mb=vram_mb,
            disk_free_mb=disk_free_mb,
        )

    def _check_ollama_version(self) -> str:
        """Return the Ollama version string from ``ollama --version``.

        Raises
        ------
        OllamaUnavailableError
            When Ollama is not in PATH, the command times out, or it returns
            a non-zero exit code — per Requirement 1.3.
        """
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Prefer stdout; fall back to stderr (some versions use stderr)
                return result.stdout.strip() or result.stderr.strip()
            raise OllamaUnavailableError(
                f"ollama --version returned exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        except FileNotFoundError:
            raise OllamaUnavailableError(
                "Ollama is not installed or not in PATH"
            )
        except subprocess.TimeoutExpired:
            raise OllamaUnavailableError("Ollama version check timed out")
        except OllamaUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OllamaUnavailableError(
                f"Ollama version check failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Task 11.2 — Full per-model evaluation pipeline
    # ------------------------------------------------------------------

    def _evaluate_model(self, model_name: str, session_id: int) -> ModelResult:
        """Run the full evaluation pipeline for one model.

        Steps: pull → start+verify → monitor+infer → quality → stop+remove.

        Requirements: 3.1–3.9, 4.1
        """
        from datetime import datetime
        import asyncio

        from ollama_benchmark.model_manager import ModelManager, ColdStartTimeoutError
        from ollama_benchmark.resource_monitor import ResourceMonitor
        from ollama_benchmark.inference_engine import InferenceEngine
        from ollama_benchmark.quality_evaluator import QualityEvaluator
        from ollama_benchmark.score_engine import ScoreEngine

        robustness = RobustnessMetrics(
            total_errors=0, total_timeouts=0, oom_count=0,
            restart_count=0, incomplete_prompts=0, stability_score=0.0,
        )
        error_log: list[ErrorEntry] = []
        inference_results = []
        quality_scores: dict = {}
        resource_summary = None
        model_run_id = None

        mm = ModelManager(base_url=self._config.ollama_base_url)
        monitor = ResourceMonitor()
        engine = InferenceEngine(base_url=self._config.ollama_base_url, db=self._db)
        evaluator = QualityEvaluator(
            judge_model=self._config.judge_model,
            base_url=self._config.ollama_base_url,
            plugins_dir=self._config.plugins_dir,
            judge_timeout=self._config.timeouts.judge,
        )

        # ---- 1. Pull ----
        pull_result = mm.pull(model_name, timeout=self._config.timeouts.download)
        if not pull_result.success:
            logger.error(
                "Pull failed for %r: %s — skipping", model_name, pull_result.error
            )
            robustness.total_errors += 1
            return ModelResult(
                model_name=model_name,
                status="skipped",
                download_time_s=pull_result.download_time_s,
                model_size_gb=None,
                cold_start_s=None,
                inference_results=[],
                resource_summary=None,
                quality_scores={},
                efficiency_indices=None,
                robustness=robustness,
                error_log=error_log,
            )

        # ---- 2. Start + verify (with crash-restart up to max_retries) ----
        cold_start_s = None
        restarts = 0
        max_retries = self._config.max_retries

        for attempt in range(max_retries + 1):
            try:
                cold_start_s = mm.start_and_verify(
                    model_name, timeout=self._config.timeouts.cold_start
                )
                break  # success
            except ColdStartTimeoutError as exc:
                logger.error(
                    "Cold-start timeout for %r (attempt %d): %s",
                    model_name, attempt + 1, exc,
                )
                robustness.total_errors += 1
                if attempt < max_retries:
                    restarts += 1
                    robustness.restart_count += 1
                    continue
                # All attempts failed
                return ModelResult(
                    model_name=model_name,
                    status="failed",
                    download_time_s=pull_result.download_time_s,
                    model_size_gb=pull_result.model_size_gb,
                    cold_start_s=None,
                    inference_results=[],
                    resource_summary=None,
                    quality_scores={},
                    efficiency_indices=None,
                    robustness=robustness,
                    error_log=error_log,
                )

        robustness.restart_count = restarts

        # ---- 3. Create preliminary model_run in DB so resource samples can link to it ----
        preliminary_result = ModelResult(
            model_name=model_name,
            status="incomplete",
            download_time_s=pull_result.download_time_s,
            model_size_gb=pull_result.model_size_gb,
            cold_start_s=cold_start_s,
            inference_results=[],
            resource_summary=None,
            quality_scores={},
            efficiency_indices=None,
            robustness=robustness,
            error_log=[],
        )
        try:
            model_run_id = self._db.save_model_result(session_id, preliminary_result)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to save preliminary model result: %s — continuing without DB link",
                exc,
            )
            model_run_id = None

        # ---- 4. Monitor + Infer ----
        # Build flat prompt list: [(category, PromptEntry), ...]
        flat_prompts = [
            (cat, pe)
            for cat, entries in self._config.prompts.items()
            for pe in entries
        ]
        total_prompts = len(flat_prompts)

        # Capture a single resource snapshot during the first prompt only.
        # The monitor starts, the first prompt runs, then the monitor stops.
        # Remaining prompts run without the background sampling loop.
        monitor.start(session_id=session_id, model_name=model_name)

        from ollama_benchmark.models import InferenceResult as _InferenceResult

        first_result: Optional[_InferenceResult] = None
        if flat_prompts:
            try:
                raw = asyncio.run(
                    engine.run_prompt(
                        model=model_name,
                        prompt=flat_prompts[0][1].text,
                        timeout=self._config.timeouts.inference,
                        category=flat_prompts[0][0],
                        model_run_id=model_run_id,
                    )
                )
                # run_prompt returns a single InferenceResult; guard against
                # test mocks that return a list or None
                if isinstance(raw, _InferenceResult):
                    first_result = raw
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Inference error on first prompt for %r: %s", model_name, exc
                )
                robustness.total_errors += 1

        # Stop monitor after first prompt — single resource snapshot captured
        resource_summary = monitor.stop(db=self._db, model_run_id=model_run_id)

        # Run remaining prompts without the monitor loop
        remaining_results: list = []
        if len(flat_prompts) > 1:
            try:
                raw_remaining = asyncio.run(
                    engine.run_all_prompts(
                        model=model_name,
                        prompts=flat_prompts[1:],
                        timeout=self._config.timeouts.inference,
                        model_run_id=model_run_id,
                    )
                )
                if isinstance(raw_remaining, list):
                    remaining_results = raw_remaining
            except Exception as exc:  # noqa: BLE001
                logger.error("Inference loop error for %r: %s", model_name, exc)
                robustness.total_errors += 1

        inference_results = (
            [first_result] if first_result is not None else []
        ) + remaining_results

        # Count timeouts and incomplete
        for r in inference_results:
            if r.timed_out:
                robustness.total_timeouts += 1
            if r.timed_out or r.error:
                robustness.incomplete_prompts += 1

        completed_count = total_prompts - robustness.incomplete_prompts
        robustness.stability_score = (
            completed_count / total_prompts if total_prompts > 0 else 0.0
        )

        # ---- 5. Quality evaluation ----
        # Build a mapping: prompt_text -> (category, expected_answer)
        prompt_map = {
            pe.text: (cat, pe.expected_answer) for cat, pe in flat_prompts
        }

        for result in inference_results:
            if result.response_text and not result.timed_out and not result.error:
                cat, expected = prompt_map.get(result.prompt_text, ("general", None))
                score = evaluator.evaluate(
                    question=result.prompt_text,
                    response=result.response_text,
                    expected=expected,
                    category=cat,
                )
                if cat not in quality_scores:
                    quality_scores[cat] = []
                if score is not None:
                    quality_scores[cat].append(score)

        # Average scores per category
        quality_scores_avg: dict = {
            cat: (sum(scores) / len(scores) if scores else None)
            for cat, scores in quality_scores.items()
        }

        # ---- 6. Stop + remove ----
        mm.stop_and_remove(model_name)

        # ---- 7. Determine final status ----
        if completed_count == 0:
            status = "failed"
        elif robustness.incomplete_prompts > 0:
            status = "incomplete"
        else:
            status = "completed"

        quality_values = [v for v in quality_scores_avg.values() if v is not None]
        quality_score = (
            sum(quality_values) / len(quality_values) if quality_values else None
        )
        tps_values = [
            r.tokens_per_second
            for r in inference_results
            if r.tokens_per_second is not None
        ]
        ttft_values = [r.ttft_ms for r in inference_results if r.ttft_ms is not None]
        latency_values = [
            r.total_response_ms
            for r in inference_results
            if r.total_response_ms is not None
        ]

        final_result = ModelResult(
            model_name=model_name,
            status=status,
            download_time_s=pull_result.download_time_s,
            model_size_gb=pull_result.model_size_gb,
            cold_start_s=cold_start_s,
            inference_results=inference_results,
            resource_summary=resource_summary,
            quality_scores=quality_scores_avg,
            efficiency_indices=None,  # computed below
            robustness=robustness,
            error_log=error_log,
            model_run_id=model_run_id,
            quality_score=quality_score,
            avg_tps=sum(tps_values) / len(tps_values) if tps_values else None,
            avg_ttft_ms=sum(ttft_values) / len(ttft_values) if ttft_values else None,
            avg_latency_ms=(
                sum(latency_values) / len(latency_values) if latency_values else None
            ),
            avg_ram_mb=resource_summary.avg_ram_mb if resource_summary else None,
        )

        # ---- 8. Compute efficiency indices for this model ----
        score_engine = ScoreEngine(self._db)
        final_result.efficiency_indices = score_engine.compute_efficiency_indices(
            final_result
        )

        return final_result

    # ------------------------------------------------------------------
    # Task 11.3 — Main benchmark session loop
    # ------------------------------------------------------------------

    def run(self) -> SessionResult:
        """Full benchmark session: detect hardware, iterate models, rank, report.

        Requirements: 12.1
        """
        import json
        from ollama_benchmark.score_engine import ScoreEngine

        # 1. Detect hardware (raises OllamaUnavailableError → non-zero exit)
        hardware = self._detect_hardware()
        logger.info("Hardware detected: %s", hardware)

        # 2. Create DB session
        config_snapshot = json.loads(self._config.model_dump_json())
        session_id = self._db.create_session(hardware, config_snapshot)
        logger.info("Session %d started", session_id)

        model_results: list[ModelResult] = []

        # 3. Evaluate each model
        for model_name in self._config.models:
            logger.info("Evaluating model: %s", model_name)
            try:
                result = self._evaluate_model(model_name, session_id)
                model_results.append(result)
                # Update the preliminary row (or insert if pull/cold-start failed early)
                self._db.save_model_result(
                    session_id, result, model_run_id=result.model_run_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Unexpected error evaluating %r: %s — continuing", model_name, exc
                )

        # 4. Compute rankings
        score_engine = ScoreEngine(self._db)
        score_engine.compute_rankings(session_id)

        # 5. Generate recommendations
        score_engine.generate_recommendations(session_id)

        # 6. Finalize session
        self._db.finalize_session(session_id)
        logger.info("Session %d completed", session_id)

        return SessionResult(
            session_id=session_id,
            status="completed",
            model_results=model_results,
        )
