"""
Report generator for ollama-performance-studio.

Produces self-contained HTML reports with embedded Plotly.js charts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ollama_benchmark.database import Database
from ollama_benchmark.models import ModelResult

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_PLOTLY_PATH = _STATIC_DIR / "plotly.min.js"


def _mean_norm_efficiency(model: ModelResult) -> float | None:
    """Average of available normalized efficiency indices, or None."""
    ei = model.efficiency_indices
    if ei is None:
        return None
    values = [
        v
        for v in (
            ei.norm_quality_per_ram,
            ei.norm_quality_per_latency,
            ei.norm_quality_per_cpu,
            ei.norm_quality_per_disk,
            ei.norm_tps_per_gb_ram,
            ei.norm_quality_per_energy,
        )
        if v is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


class ReportGenerator:
    """Generates self-contained HTML benchmark reports."""

    def __init__(self, db: Database, template_dir: Path | None = None) -> None:
        """Initialize the report generator.

        Parameters
        ----------
        db:
            Database instance for querying benchmark results.
        template_dir:
            Directory containing Jinja2 templates. Defaults to the package's
            templates directory.
        """
        self._db = db

        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self._env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def _load_plotly_js(self) -> str:
        """Load vendored Plotly.js for inline embedding."""
        if not _PLOTLY_PATH.exists():
            logger.warning("Plotly.js not found at %s — charts will be unavailable", _PLOTLY_PATH)
            return "console.warn('Plotly.js not bundled');"
        content = _PLOTLY_PATH.read_text(encoding="utf-8")
        if "placeholder" in content[:200].lower():
            logger.warning("Plotly.js file appears to be a placeholder")
        return content

    def generate(self, session_id: int, output_path: Path | str) -> Path:
        """Generate an HTML report for a benchmark session.

        Parameters
        ----------
        session_id:
            Database session ID to generate a report for.
        output_path:
            Directory where the HTML file will be saved. The file will be
            named ``report_<session_id>.html``.

        Returns
        -------
        Path
            Full path to the generated HTML file.
        """
        output_path = Path(output_path)
        logger.info("Generating report for session %d …", session_id)

        session_detail = self._db.get_session_detail(session_id)
        model_results = self._db.get_model_results(session_id)
        recommendations = self._db.get_recommendations(session_id)

        categories = self._extract_categories(model_results)

        for model in model_results:
            if model.model_run_id is not None and model.status in (
                "completed",
                "incomplete",
            ):
                model.resource_timeseries = self._db.get_resource_timeseries(
                    model.model_run_id
                )
            else:
                model.resource_timeseries = []

        overall_ranking = self._compute_overall_ranking(model_results)
        tps_ranking = self._compute_tps_ranking(model_results)
        quality_ranking = self._compute_quality_ranking(model_results)
        ram_ranking = self._compute_ram_ranking(model_results)
        efficiency_ranking = self._compute_efficiency_ranking(model_results)

        efficiency_chart_data = self._build_efficiency_chart_data(model_results)
        resource_chart_data = self._build_resource_chart_data(model_results)

        context = {
            "session_id": session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": session_detail.hardware,
            "model_results": model_results,
            "categories": categories,
            "overall_ranking": overall_ranking,
            "tps_ranking": tps_ranking,
            "quality_ranking": quality_ranking,
            "ram_ranking": ram_ranking,
            "efficiency_ranking": efficiency_ranking,
            "recommendations": recommendations,
            "plotly_js": self._load_plotly_js(),
            "efficiency_chart_json": json.dumps(efficiency_chart_data),
            "resource_chart_json": json.dumps(resource_chart_data),
        }

        template = self._env.get_template("report.html.j2")
        html_content = template.render(**context)

        output_file = output_path / f"report_{session_id}.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content, encoding="utf-8")

        logger.info("Report generated: %s", output_file)
        return output_file

    def _extract_categories(self, model_results: list[ModelResult]) -> list[str]:
        """Extract all unique quality score categories from model results."""
        categories: set[str] = set()
        for model in model_results:
            if model.quality_scores:
                categories.update(model.quality_scores.keys())
        return sorted(categories)

    def _compute_overall_ranking(self, model_results: list[ModelResult]) -> list[ModelResult]:
        """Rank models by overall_rank (descending)."""
        ranked = [m for m in model_results if m.overall_rank is not None]
        return sorted(ranked, key=lambda m: m.overall_rank or 0.0, reverse=True)

    def _compute_tps_ranking(self, model_results: list[ModelResult]) -> list[ModelResult]:
        """Rank models by tokens per second (descending)."""
        ranked = [m for m in model_results if m.avg_tps is not None]
        return sorted(ranked, key=lambda m: m.avg_tps or 0.0, reverse=True)

    def _compute_quality_ranking(self, model_results: list[ModelResult]) -> list[ModelResult]:
        """Rank models by quality score (descending)."""
        ranked = [m for m in model_results if m.quality_score is not None]
        return sorted(ranked, key=lambda m: m.quality_score or 0.0, reverse=True)

    def _compute_ram_ranking(self, model_results: list[ModelResult]) -> list[ModelResult]:
        """Rank models by average RAM usage (ascending)."""
        ranked = [m for m in model_results if m.avg_ram_mb is not None]
        return sorted(ranked, key=lambda m: m.avg_ram_mb or float("inf"))

    def _compute_efficiency_ranking(self, model_results: list[ModelResult]) -> list[ModelResult]:
        """Rank models by mean normalized efficiency (descending)."""
        scored: list[tuple[float, ModelResult]] = []
        for m in model_results:
            mean_eff = _mean_norm_efficiency(m)
            if mean_eff is not None:
                scored.append((mean_eff, m))
            elif m.overall_rank is not None:
                scored.append((m.overall_rank, m))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [m for _, m in scored]

    def _build_efficiency_chart_data(self, model_results: list[ModelResult]) -> dict:
        """Build Plotly data for the efficiency bar chart."""
        names: list[str] = []
        qpr: list[float | None] = []
        qpl: list[float | None] = []
        qpc: list[float | None] = []
        for m in model_results:
            ei = m.efficiency_indices
            if ei is None:
                continue
            names.append(m.model_name)
            qpr.append(ei.norm_quality_per_ram)
            qpl.append(ei.norm_quality_per_latency)
            qpc.append(ei.norm_quality_per_cpu)
        return {
            "names": names,
            "norm_quality_per_ram": qpr,
            "norm_quality_per_latency": qpl,
            "norm_quality_per_cpu": qpc,
        }

    def _build_resource_chart_data(self, model_results: list[ModelResult]) -> list[dict]:
        """Build Plotly time-series payloads per model."""
        charts: list[dict] = []
        for idx, model in enumerate(model_results, start=1):
            ts = model.resource_timeseries or []
            if not ts:
                continue
            charts.append(
                {
                    "element_id": f"resource-chart-{idx}",
                    "model_name": model.model_name,
                    "timestamps": [s.timestamp.isoformat() for s in ts],
                    "cpu": [
                        (
                            sum(s.cpu_per_core) / len(s.cpu_per_core)
                            if s.cpu_per_core
                            else None
                        )
                        for s in ts
                    ],
                    "ram_mb": [s.ram_mb for s in ts],
                    "vram_mb": [s.vram_mb for s in ts],
                }
            )
        return charts
