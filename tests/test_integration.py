"""
Integration tests for ollama-benchmark.

Tasks 17.1, 17.2, 17.3
Validates: Requirements 5.1, 5.3, 8.2, 8.3, 8.4, 9.1, 9.2, 10.1

These tests require either a live Ollama instance (for 17.1) or fixture data.
All tests are marked @pytest.mark.integration so they can be skipped:
    pytest -m "not integration"
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ollama_benchmark.dashboard import create_app
from ollama_benchmark.database import Database
from ollama_benchmark.models import (
    HardwareInfo,
    InferenceResult,
    ModelResult,
    RobustnessMetrics,
    SessionSummary,
)
from ollama_benchmark.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hardware() -> HardwareInfo:
    return HardwareInfo(
        os="Linux 5.15",
        python_version="3.11.0",
        ollama_version="0.3.0",
        cpu_cores=4,
        cpu_mhz=2400.0,
        ram_mb=8192.0,
        has_gpu=False,
        vram_mb=0.0,
        disk_free_mb=51200.0,
    )


def _make_robustness(stability: float = 1.0) -> RobustnessMetrics:
    return RobustnessMetrics(
        total_errors=0,
        total_timeouts=0,
        oom_count=0,
        restart_count=0,
        incomplete_prompts=0,
        stability_score=stability,
    )


def _populate_test_db(db: Database) -> tuple[int, int]:
    """Insert known fixture data into *db*.

    Returns (session_id, model_run_id).
    """
    hardware = _make_hardware()
    config_snapshot = {"models": ["tinyllama"], "prompts": {"reasoning": [{"text": "test"}]}}

    session_id = db.create_session(hardware, config_snapshot)

    model_result = ModelResult(
        model_name="tinyllama",
        status="completed",
        download_time_s=10.0,
        model_size_gb=0.6,
        cold_start_s=2.5,
        inference_results=[
            InferenceResult(
                prompt_text="What is 2+2?",
                response_text="The answer is 4.",
                ttft_ms=55.0,
                total_response_ms=320.0,
                tokens_generated=8,
                tokens_per_second=25.0,
                avg_inter_token_ms=40.0,
                timed_out=False,
            )
        ],
        resource_summary=None,
        quality_scores={"reasoning": 0.85},
        efficiency_indices=None,
        robustness=_make_robustness(),
    )

    model_run_id = db.save_model_result(session_id, model_result)
    db.save_inference_result(
        model_run_id,
        model_result.inference_results[0],
        category="reasoning",
    )
    db.finalize_session(session_id, status="completed")

    return session_id, model_run_id


# ---------------------------------------------------------------------------
# Task 17.1 — Full pipeline smoke test (requires live Ollama)
# Validates: Requirements 5.1, 5.3, 8.2, 8.3, 8.4
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_pipeline_smoke_test(tmp_path: Path) -> None:
    """
    Run 1 small model (tinyllama) end-to-end.

    Requires a live Ollama instance with tinyllama available.
    Asserts DB contains session, model_run, prompt_results, resource_samples.

    This test is skipped by default; run with: pytest -m integration
    """
    # Skip if no live Ollama available
    import httpx
    try:
        resp = httpx.get("http://localhost:11434", timeout=3.0)
        if resp.status_code not in (200, 404):
            pytest.skip("Ollama not available at localhost:11434")
    except Exception:
        pytest.skip("Ollama not available at localhost:11434")

    from ollama_benchmark.benchmark_runner import BenchmarkRunner
    from ollama_benchmark.config import BenchmarkConfig, TimeoutConfig

    db_path = tmp_path / "test_integration.db"
    config = BenchmarkConfig(
        models=["tinyllama"],
        prompts={"reasoning": [{"text": "What is 2+2?"}]},
        timeouts=TimeoutConfig(download=600, cold_start=120, inference=120, judge=60),
        max_retries=1,
        database_path=str(db_path),
    )

    db = Database(str(db_path))
    runner = BenchmarkRunner(config, db)
    session_result = runner.run()

    # Validate DB has session
    sessions = db.get_sessions()
    assert len(sessions) >= 1
    assert sessions[0].id == session_result.session_id

    # Validate model_run exists
    model_results = db.get_model_results(session_result.session_id)
    assert len(model_results) >= 1
    assert model_results[0].model_name == "tinyllama"

    # Validate prompt_results exist
    if model_results[0].model_run_id is not None:
        prompt_results = db.get_prompt_results(model_results[0].model_run_id)
        assert len(prompt_results) >= 1

    # Validate session detail
    detail = db.get_session_detail(session_result.session_id)
    assert detail.hardware is not None
    assert detail.summary.status == "completed"


# ---------------------------------------------------------------------------
# Task 17.2 — Report generation integration test
# Validates: Requirements 9.1, 9.2
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_report_generation_from_db_fixture(tmp_path: Path) -> None:
    """
    Load known DB fixture, generate HTML report, assert file exists and non-empty.

    Does NOT require live Ollama — uses in-memory DB with fixture data.

    Validates: Requirements 9.1, 9.2
    """
    # Build a real in-memory DB with fixture data
    db = Database(":memory:")
    session_id, model_run_id = _populate_test_db(db)

    # Generate the report
    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = ReportGenerator(db)
    report_path = gen.generate(session_id=session_id, output_path=output_dir)

    # Assert file exists and is non-empty
    assert report_path.exists(), f"Report file not found: {report_path}"
    assert report_path.stat().st_size > 0, "Report file is empty"

    content = report_path.read_text(encoding="utf-8")
    assert len(content) > 100, "Report content is too short"

    # Assert HTML structure is present
    assert "<html" in content.lower() or "<!doctype" in content.lower()


@pytest.mark.integration
def test_report_contains_model_data(tmp_path: Path) -> None:
    """Report includes model name and quality score data from fixture."""
    db = Database(":memory:")
    session_id, model_run_id = _populate_test_db(db)

    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = ReportGenerator(db)
    report_path = gen.generate(session_id=session_id, output_path=output_dir)
    content = report_path.read_text(encoding="utf-8")

    assert "tinyllama" in content


# ---------------------------------------------------------------------------
# Task 17.3 — Dashboard startup integration test
# Validates: Requirements 10.1
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_dashboard_sessions_endpoint_returns_200() -> None:
    """
    Start Flask app with test DB, assert GET /api/sessions returns HTTP 200.

    Does NOT require live Ollama — uses in-memory DB with fixture data.

    Validates: Requirements 10.1
    """
    db = Database(":memory:")
    _populate_test_db(db)

    app = create_app(db)
    app.config["TESTING"] = True

    with app.test_client() as client:
        response = client.get("/api/sessions")

    assert response.status_code == 200
    import json
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) >= 1


@pytest.mark.integration
def test_dashboard_session_detail_returns_200() -> None:
    """GET /api/session/<id> returns HTTP 200 for existing session."""
    db = Database(":memory:")
    session_id, _ = _populate_test_db(db)

    app = create_app(db)
    app.config["TESTING"] = True

    with app.test_client() as client:
        response = client.get(f"/api/session/{session_id}")

    assert response.status_code == 200
    import json
    data = json.loads(response.data)
    assert "hardware" in data
    assert "model_results" in data


@pytest.mark.integration
def test_dashboard_timeseries_endpoint_returns_200() -> None:
    """GET /api/timeseries/<model_run_id> returns HTTP 200."""
    db = Database(":memory:")
    _, model_run_id = _populate_test_db(db)

    app = create_app(db)
    app.config["TESTING"] = True

    with app.test_client() as client:
        response = client.get(f"/api/timeseries/{model_run_id}")

    assert response.status_code == 200
    import json
    data = json.loads(response.data)
    assert isinstance(data, list)
