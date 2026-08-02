"""
Offline unit tests for P0 end-to-end contract.

Covers SessionResult return shape, ModelResult scalar mapping from DB,
report generation with rankings/recommendations, and dashboard create_app.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ollama_benchmark.benchmark_runner import BenchmarkRunner
from ollama_benchmark.config import BenchmarkConfig, TimeoutConfig
from ollama_benchmark.database import Database
from ollama_benchmark.models import (
    EfficiencyIndices,
    HardwareInfo,
    InferenceResult,
    ModelResult,
    Recommendation,
    ResourceSample,
    RobustnessMetrics,
    SessionResult,
)
from ollama_benchmark.report_generator import ReportGenerator


def _hardware() -> HardwareInfo:
    return HardwareInfo(
        os="Windows",
        python_version="3.11.0",
        ollama_version="0.1.0",
        cpu_cores=4,
        cpu_mhz=2400.0,
        ram_mb=16384.0,
        has_gpu=False,
        vram_mb=0.0,
        disk_free_mb=50000.0,
    )


def _robustness(**kwargs) -> RobustnessMetrics:
    defaults = dict(
        total_errors=0,
        total_timeouts=0,
        oom_count=0,
        restart_count=0,
        incomplete_prompts=0,
        stability_score=1.0,
    )
    defaults.update(kwargs)
    return RobustnessMetrics(**defaults)


def _model_result(
    name: str,
    *,
    quality: float,
    tps: float,
    overall: float | None = None,
    ram: float | None = None,
) -> ModelResult:
    return ModelResult(
        model_name=name,
        status="completed",
        download_time_s=10.0,
        model_size_gb=1.5,
        cold_start_s=2.0,
        inference_results=[
            InferenceResult(
                prompt_text="What is 2+2?",
                response_text="4",
                ttft_ms=50.0,
                total_response_ms=200.0,
                tokens_generated=10,
                tokens_per_second=tps,
                avg_inter_token_ms=20.0,
            )
        ],
        resource_summary=None,
        quality_scores={"coding": quality, "reasoning": quality},
        efficiency_indices=EfficiencyIndices(
            quality_per_ram=quality / 1000,
            quality_per_latency=quality / 200,
            quality_per_cpu=quality / 50,
            quality_per_disk=quality / 1.5,
            tps_per_gb_ram=tps / 1.0,
            quality_per_energy=None,
            norm_quality_per_ram=quality,
            norm_quality_per_latency=quality,
            norm_quality_per_cpu=quality * 0.9,
            norm_quality_per_disk=quality * 0.8,
            norm_tps_per_gb_ram=tps / 50,
            norm_quality_per_energy=None,
        ),
        robustness=_robustness(),
        overall_rank=overall if overall is not None else quality,
        quality_score=quality,
        avg_tps=tps,
        avg_ttft_ms=50.0,
        avg_latency_ms=200.0,
        avg_ram_mb=ram,
    )


def _seed_session(db: Database) -> int:
    session_id = db.create_session(_hardware(), {"models": ["alpha", "beta"]})

    alpha = _model_result("alpha", quality=0.9, tps=30.0, overall=0.85)
    beta = _model_result("beta", quality=0.7, tps=50.0, overall=0.75)

    alpha_id = db.save_model_result(session_id, alpha)
    beta_id = db.save_model_result(session_id, beta)

    # Persist overall_rank explicitly (save_model_result includes it)
    # Add resource samples so avg_ram_mb is computable
    now = datetime.utcnow()
    for mid, ram in ((alpha_id, 1200.0), (beta_id, 800.0)):
        for i in range(3):
            db.save_resource_sample(
                mid,
                ResourceSample(
                    timestamp=now.replace(second=i),
                    cpu_per_core=[40.0, 45.0],
                    ram_mb=ram + i * 10,
                    gpu_percent=None,
                    vram_mb=None,
                    cpu_temp_c=None,
                    gpu_temp_c=None,
                    power_watts=None,
                ),
            )

    db.save_recommendation(
        session_id,
        Recommendation(
            profile="best_overall",
            model_name="alpha",
            justification="Highest overall rank",
        ),
    )
    db.finalize_session(session_id)
    return session_id


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(str(tmp_path / "p0.db"))


def test_get_model_results_includes_scalars(db: Database) -> None:
    session_id = _seed_session(db)
    results = db.get_model_results(session_id)

    assert len(results) == 2
    by_name = {m.model_name: m for m in results}

    alpha = by_name["alpha"]
    assert alpha.model_run_id is not None
    assert alpha.quality_score == pytest.approx(0.9)
    assert alpha.avg_tps == pytest.approx(30.0)
    assert alpha.overall_rank == pytest.approx(0.85)
    assert alpha.avg_ttft_ms == pytest.approx(50.0)
    assert alpha.avg_latency_ms == pytest.approx(200.0)
    assert alpha.avg_ram_mb is not None
    assert alpha.avg_ram_mb == pytest.approx(1210.0)  # (1200+1210+1220)/3

    beta = by_name["beta"]
    assert beta.avg_tps == pytest.approx(50.0)
    assert beta.avg_ram_mb == pytest.approx(810.0)


def test_get_recommendations(db: Database) -> None:
    session_id = _seed_session(db)
    recs = db.get_recommendations(session_id)
    assert len(recs) == 1
    assert recs[0].model_name == "alpha"
    assert recs[0].profile == "best_overall"


def test_report_generator_rankings_and_recs(db: Database, tmp_path: Path) -> None:
    session_id = _seed_session(db)
    report_path = ReportGenerator(db).generate(session_id, tmp_path / "out")

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    assert "alpha" in content
    assert "beta" in content
    assert "best_overall" in content
    assert "Highest overall rank" in content
    assert "Automatic Recommendations" in content

    # Plotly should be embedded (real bundle starts with comment / IIFE)
    assert "Plotly" in content or "plotly" in content.lower()

    # Rankings: beta should rank first on TPS, alpha first on quality
    gen = ReportGenerator(db)
    models = db.get_model_results(session_id)
    tps = gen._compute_tps_ranking(models)
    quality = gen._compute_quality_ranking(models)
    overall = gen._compute_overall_ranking(models)
    ram = gen._compute_ram_ranking(models)

    assert tps[0].model_name == "beta"
    assert quality[0].model_name == "alpha"
    assert overall[0].model_name == "alpha"
    assert ram[0].model_name == "beta"


def test_report_rankings_do_not_sort_by_dict() -> None:
    """Regression: sorting by quality_scores dict must not happen."""
    models = [
        _model_result("low", quality=0.2, tps=10.0, overall=0.2),
        _model_result("high", quality=0.9, tps=5.0, overall=0.9),
    ]
    gen = ReportGenerator(MagicMock())
    ranked = gen._compute_quality_ranking(models)
    assert ranked[0].model_name == "high"
    assert ranked[1].model_name == "low"


def test_runner_run_returns_session_result(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        models=["tiny"],
        prompts={"test": [{"text": "hi", "expected_answer": None}]},
        timeouts=TimeoutConfig(),
        database_path=str(tmp_path / "runner.db"),
    )
    db = Database(config.database_path)
    runner = BenchmarkRunner(config, db)

    fake_result = _model_result("tiny", quality=0.5, tps=12.0, overall=0.5)

    with (
        patch.object(runner, "_detect_hardware", return_value=_hardware()),
        patch.object(runner, "_evaluate_model", return_value=fake_result),
        patch("ollama_benchmark.score_engine.ScoreEngine") as MockScore,
    ):
        mock_engine = MockScore.return_value
        mock_engine.compute_rankings = MagicMock()
        mock_engine.generate_recommendations = MagicMock()

        session_result = runner.run()

    assert isinstance(session_result, SessionResult)
    assert session_result.session_id is not None
    assert session_result.status == "completed"
    assert len(session_result.model_results) == 1
    assert session_result.model_results[0].model_name == "tiny"


def test_dashboard_create_app_with_database(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "dash.db"))
    session_id = db.create_session(_hardware(), {"models": []})
    db.finalize_session(session_id)

    from ollama_benchmark.dashboard import create_app

    app = create_app(db)
    with app.test_client() as client:
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) >= 1

        response = client.get("/")
        assert response.status_code == 200
        assert b"Ollama Benchmark Dashboard" in response.data
