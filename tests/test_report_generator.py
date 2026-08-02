"""
Unit tests for ReportGenerator.

Task 13.3
Validates: Requirements 9.1, 9.2, 9.3
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ollama_benchmark.models import (
    HardwareInfo,
    ModelResult,
    RobustnessMetrics,
    SessionDetail,
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
        cpu_cores=8,
        cpu_mhz=3200.0,
        ram_mb=16384.0,
        has_gpu=False,
        vram_mb=0.0,
        disk_free_mb=51200.0,
    )


def _make_robustness() -> RobustnessMetrics:
    return RobustnessMetrics(
        total_errors=0,
        total_timeouts=0,
        oom_count=0,
        restart_count=0,
        incomplete_prompts=0,
        stability_score=1.0,
    )


def _make_model_result(model_name: str = "llama3") -> ModelResult:
    return ModelResult(
        model_name=model_name,
        status="completed",
        download_time_s=5.0,
        model_size_gb=4.1,
        cold_start_s=1.2,
        inference_results=[],
        resource_summary=None,
        quality_scores={"reasoning": 0.85, "coding": 0.78},
        efficiency_indices=None,
        robustness=_make_robustness(),
        model_run_id=1,
        overall_rank=0.82,
        quality_score=0.815,
        avg_tps=45.3,
        avg_ttft_ms=120.5,
        avg_latency_ms=350.0,
        avg_ram_mb=4096.0,
        resource_timeseries=[],
    )


def _make_session_summary(session_id: int = 1) -> SessionSummary:
    from datetime import datetime
    return SessionSummary(
        id=session_id,
        started_at=datetime(2026, 1, 1, 10, 0, 0),
        finished_at=datetime(2026, 1, 1, 11, 0, 0),
        status="completed",
        model_count=1,
    )


def _make_session_detail(session_id: int = 1) -> SessionDetail:
    return SessionDetail(
        summary=_make_session_summary(session_id),
        hardware=_make_hardware(),
        config_snapshot={"models": ["llama3"]},
        model_results=[_make_model_result()],
    )


def _make_mock_db(
    session_detail=None,
    model_results=None,
    recommendations=None,
) -> MagicMock:
    db = MagicMock()
    db.get_session_detail.return_value = session_detail or _make_session_detail()
    db.get_model_results.return_value = model_results or [_make_model_result()]
    db.get_recommendations.return_value = recommendations or []
    db.get_resource_timeseries.return_value = []
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateBasic:
    """generate() produces a non-empty HTML file in the output directory."""

    def test_generates_html_file(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=1, output_path=tmp_path)

        assert output_file.exists()
        assert output_file.suffix == ".html"
        assert output_file.stat().st_size > 0

    def test_output_filename_contains_session_id(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=42, output_path=tmp_path)

        assert "42" in output_file.name

    def test_html_content_is_non_empty(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=1, output_path=tmp_path)

        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_returns_path_object(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        result = gen.generate(session_id=1, output_path=tmp_path)
        assert isinstance(result, Path)

    def test_creates_output_dir_if_not_exists(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "reports" / "session1"
        db = _make_mock_db()
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=1, output_path=output_dir)

        assert output_dir.exists()
        assert output_file.exists()


class TestHtmlSections:
    """Generated HTML contains key section headers and structure."""

    def test_html_contains_doctype(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=1, output_path=tmp_path)
        content = output_file.read_text(encoding="utf-8").lower()
        assert "<!doctype html>" in content or "<html" in content

    def test_html_contains_model_name(self, tmp_path: Path) -> None:
        mr = _make_model_result("test-llama-3")
        db = _make_mock_db(
            model_results=[mr],
            session_detail=SessionDetail(
                summary=_make_session_summary(),
                hardware=_make_hardware(),
                config_snapshot={},
                model_results=[mr],
            ),
        )
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=1, output_path=tmp_path)
        content = output_file.read_text(encoding="utf-8")
        assert "test-llama-3" in content

    def test_html_contains_session_id(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=99, output_path=tmp_path)
        content = output_file.read_text(encoding="utf-8")
        assert "99" in content

    def test_html_contains_hardware_info(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=1, output_path=tmp_path)
        content = output_file.read_text(encoding="utf-8")
        # Hardware OS should appear somewhere
        assert "Linux" in content or "cpu" in content.lower()

    def test_html_contains_quality_data(self, tmp_path: Path) -> None:
        mr = _make_model_result("llama3")
        db = _make_mock_db(model_results=[mr])
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=1, output_path=tmp_path)
        content = output_file.read_text(encoding="utf-8")
        # Quality categories should appear
        assert "reasoning" in content or "quality" in content.lower()


class TestMultipleModels:
    """generate() handles multiple models and failed/skipped ones correctly."""

    def test_multiple_models_all_appear(self, tmp_path: Path) -> None:
        mr1 = _make_model_result("model-alpha")
        mr2 = _make_model_result("model-beta")
        mr2.status = "failed"
        mr2.quality_score = None

        session_detail = SessionDetail(
            summary=_make_session_summary(),
            hardware=_make_hardware(),
            config_snapshot={},
            model_results=[mr1, mr2],
        )
        db = _make_mock_db(model_results=[mr1, mr2], session_detail=session_detail)
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=1, output_path=tmp_path)
        content = output_file.read_text(encoding="utf-8")

        assert "model-alpha" in content
        assert "model-beta" in content

    def test_failed_model_status_shown(self, tmp_path: Path) -> None:
        mr = _make_model_result("bad-model")
        mr.status = "failed"

        session_detail = SessionDetail(
            summary=_make_session_summary(),
            hardware=_make_hardware(),
            config_snapshot={},
            model_results=[mr],
        )
        db = _make_mock_db(model_results=[mr], session_detail=session_detail)
        gen = ReportGenerator(db)
        output_file = gen.generate(session_id=1, output_path=tmp_path)
        content = output_file.read_text(encoding="utf-8")

        assert "failed" in content


class TestQueryCalls:
    """generate() calls the correct DB query methods."""

    def test_calls_get_session_detail(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        gen.generate(session_id=5, output_path=tmp_path)
        db.get_session_detail.assert_called_with(5)

    def test_calls_get_model_results(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        gen.generate(session_id=5, output_path=tmp_path)
        db.get_model_results.assert_called_with(5)

    def test_calls_get_recommendations(self, tmp_path: Path) -> None:
        db = _make_mock_db()
        gen = ReportGenerator(db)
        gen.generate(session_id=5, output_path=tmp_path)
        db.get_recommendations.assert_called_with(5)
