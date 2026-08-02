"""
Unit tests for the dashboard Flask app.

Task 14.4
Validates: Requirements 10.1, 10.2
"""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ollama_benchmark.dashboard import create_app
from ollama_benchmark.models import (
    ComparisonReport,
    HardwareInfo,
    ModelResult,
    ResourceSample,
    RobustnessMetrics,
    SessionDetail,
    SessionSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_summary(session_id: int = 1) -> SessionSummary:
    return SessionSummary(
        id=session_id,
        started_at=datetime(2026, 1, 1, 10, 0, 0),
        finished_at=datetime(2026, 1, 1, 11, 0, 0),
        status="completed",
        model_count=2,
    )


def _make_hardware() -> HardwareInfo:
    return HardwareInfo(
        os="Linux 5.15",
        python_version="3.11.0",
        ollama_version="0.3.0",
        cpu_cores=8,
        cpu_mhz=3200.0,
        ram_mb=16384.0,
        has_gpu=False,
        vram_mb=0.0,
        disk_free_mb=51200.0,
    )


def _make_model_result(model_name: str = "llama3", status: str = "completed") -> ModelResult:
    return ModelResult(
        model_name=model_name,
        status=status,
        download_time_s=5.0,
        model_size_gb=4.1,
        cold_start_s=1.2,
        inference_results=[],
        resource_summary=None,
        quality_scores={"reasoning": 0.85},
        efficiency_indices=None,
        robustness=RobustnessMetrics(0, 0, 0, 0, 0, 1.0),
        model_run_id=1,
    )


def _make_session_detail(session_id: int = 1, model_count: int = 2) -> SessionDetail:
    return SessionDetail(
        summary=_make_session_summary(session_id),
        hardware=_make_hardware(),
        config_snapshot={"models": ["llama3", "gemma2"]},
        model_results=[
            _make_model_result("llama3"),
            _make_model_result("gemma2"),
        ],
    )


def _make_mock_db(
    sessions=None,
    session_detail=None,
    model_results=None,
    timeseries=None,
) -> MagicMock:
    db = MagicMock()
    db.get_sessions.return_value = sessions or [_make_session_summary(1)]
    db.get_session_detail.return_value = session_detail or _make_session_detail()
    db.get_model_results.return_value = model_results or [_make_model_result()]
    db.get_resource_timeseries.return_value = timeseries or []

    # compare_sessions returns a ComparisonReport-like object
    mock_report = MagicMock()
    mock_report.sessions = []
    db.compare_sessions.return_value = mock_report
    return db


@pytest.fixture
def client():
    """Flask test client with a mocked database."""
    db = _make_mock_db()
    app = create_app(db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIndexRoute:
    """GET / — serves the main dashboard HTML."""

    def test_returns_200(self, client) -> None:
        c, db = client
        response = c.get("/")
        assert response.status_code == 200

    def test_returns_html_content_type(self, client) -> None:
        c, db = client
        response = c.get("/")
        assert "text/html" in response.content_type

    def test_html_contains_dashboard_title(self, client) -> None:
        c, db = client
        response = c.get("/")
        assert b"Ollama" in response.data or b"Benchmark" in response.data


class TestApiSessions:
    """GET /api/sessions — returns list of sessions as JSON."""

    def test_returns_200(self, client) -> None:
        c, db = client
        response = c.get("/api/sessions")
        assert response.status_code == 200

    def test_returns_json_array(self, client) -> None:
        c, db = client
        response = c.get("/api/sessions")
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_sessions_contain_id_field(self, client) -> None:
        c, db = client
        response = c.get("/api/sessions")
        data = json.loads(response.data)
        assert len(data) > 0
        assert "id" in data[0]

    def test_calls_db_get_sessions(self, client) -> None:
        c, db = client
        c.get("/api/sessions")
        db.get_sessions.assert_called()

    def test_empty_sessions_returns_empty_array(self) -> None:
        db = MagicMock()
        db.get_sessions.return_value = []
        app = create_app(db)
        app.config["TESTING"] = True
        with app.test_client() as c:
            response = c.get("/api/sessions")
        data = json.loads(response.data)
        assert data == []


class TestApiSessionById:
    """GET /api/session/<id> — returns full session detail."""

    def test_returns_200_for_existing_session(self, client) -> None:
        c, db = client
        response = c.get("/api/session/1")
        assert response.status_code == 200

    def test_returns_json_with_hardware(self, client) -> None:
        c, db = client
        response = c.get("/api/session/1")
        data = json.loads(response.data)
        assert "hardware" in data

    def test_returns_json_with_model_results(self, client) -> None:
        c, db = client
        response = c.get("/api/session/1")
        data = json.loads(response.data)
        assert "model_results" in data

    def test_calls_db_get_session_detail_with_id(self, client) -> None:
        c, db = client
        c.get("/api/session/42")
        db.get_session_detail.assert_called_with(42)

    def test_returns_404_for_missing_session(self) -> None:
        db = _make_mock_db()
        db.get_session_detail.side_effect = ValueError("Session 999 not found")
        app = create_app(db)
        app.config["TESTING"] = True
        with app.test_client() as c:
            response = c.get("/api/session/999")
        assert response.status_code == 404

    def test_404_response_contains_error_key(self) -> None:
        db = _make_mock_db()
        db.get_session_detail.side_effect = ValueError("Session 999 not found")
        app = create_app(db)
        app.config["TESTING"] = True
        with app.test_client() as c:
            response = c.get("/api/session/999")
        data = json.loads(response.data)
        assert "error" in data

    def test_session_model_results_contain_model_name(self, client) -> None:
        c, db = client
        response = c.get("/api/session/1")
        data = json.loads(response.data)
        model_names = [r["model_name"] for r in data.get("model_results", [])]
        assert len(model_names) > 0


class TestApiTimeseries:
    """GET /api/timeseries/<model_run_id> — returns resource time-series."""

    def test_returns_200(self, client) -> None:
        c, db = client
        response = c.get("/api/timeseries/1")
        assert response.status_code == 200

    def test_returns_json_array(self, client) -> None:
        c, db = client
        response = c.get("/api/timeseries/1")
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_calls_db_with_correct_run_id(self, client) -> None:
        c, db = client
        c.get("/api/timeseries/42")
        db.get_resource_timeseries.assert_called_with(42)

    def test_empty_timeseries_returns_empty_array(self, client) -> None:
        c, db = client
        db.get_resource_timeseries.return_value = []
        response = c.get("/api/timeseries/1")
        data = json.loads(response.data)
        assert data == []


class TestApiCompare:
    """GET /api/compare — compares multiple sessions."""

    def test_returns_400_when_no_ids_param(self, client) -> None:
        c, db = client
        response = c.get("/api/compare")
        assert response.status_code == 400

    def test_returns_400_for_malformed_ids(self, client) -> None:
        c, db = client
        response = c.get("/api/compare?ids=abc,def")
        assert response.status_code == 400

    def test_returns_200_for_valid_ids(self, client) -> None:
        c, db = client
        # Return a real ComparisonReport dataclass to allow asdict()
        from ollama_benchmark.models import ComparisonReport
        db.compare_sessions.return_value = ComparisonReport(
            session_ids=[1, 2],
            models=["llama3"],
            metrics={},
        )
        response = c.get("/api/compare?ids=1,2")
        assert response.status_code == 200

    def test_calls_db_compare_sessions(self, client) -> None:
        c, db = client
        from ollama_benchmark.models import ComparisonReport
        db.compare_sessions.return_value = ComparisonReport(
            session_ids=[1, 2],
            models=[],
            metrics={},
        )
        c.get("/api/compare?ids=1,2")
        db.compare_sessions.assert_called_with([1, 2])
