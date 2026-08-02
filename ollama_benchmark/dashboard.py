"""
Interactive web dashboard for ollama-performance-studio.

Flask app serving session data from the SQLite database.
Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 13.1, 14.4
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime

from flask import Flask, jsonify, render_template_string, request

from ollama_benchmark.database import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _default_serializer(obj: object) -> object:
    """Fallback JSON serializer for datetimes and any remaining non-serializable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _to_json(obj: object) -> str:
    """Serialize *obj* (dataclass or plain dict) to a JSON string."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        data = dataclasses.asdict(obj)
    else:
        data = obj  # type: ignore[assignment]
    return json.dumps(data, default=_default_serializer)


# ---------------------------------------------------------------------------
# Dashboard HTML — task 14.2 provides the full frontend; this is the shell
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama Benchmark Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #2c3e50; margin-bottom: 20px; }
        h2 { color: #34495e; margin-top: 30px; margin-bottom: 15px; }
        .controls {
            background: white; padding: 20px; border-radius: 8px;
            margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .control-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        select, input {
            width: 100%; padding: 8px; border: 1px solid #ddd;
            border-radius: 4px; font-size: 14px;
        }
        button {
            background: #3498db; color: white; border: none;
            padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 14px;
        }
        button:hover { background: #2980b9; }
        .charts-container {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px; margin-bottom: 20px;
        }
        .chart-card {
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chart-card h3 { margin-bottom: 15px; color: #34495e; }
        .table-container {
            background: white; padding: 20px; border-radius: 8px;
            margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow-x: auto;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #3498db; color: white; }
        tr:hover { background: #f5f5f5; }
        canvas { max-height: 400px; }
        .recommendations {
            background: #e8f4f8; padding: 20px; border-radius: 8px; margin-top: 20px;
        }
        .recommendation-item {
            background: white; padding: 15px; margin: 10px 0;
            border-radius: 4px; border-left: 4px solid #27ae60;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>&#x1F4CA; Ollama Benchmark Dashboard</h1>

        <div class="controls">
            <div class="control-group">
                <label for="session-select">Select Session:</label>
                <select id="session-select">
                    <option value="">-- Select a session --</option>
                </select>
            </div>
            <div class="control-group">
                <label for="category-filter">Filter by Category:</label>
                <select id="category-filter">
                    <option value="">All Categories</option>
                </select>
            </div>
            <div class="control-group">
                <label for="min-quality">Minimum Quality Score:</label>
                <input type="range" id="min-quality" min="0" max="1" step="0.1" value="0">
                <span id="quality-value">0.0</span>
            </div>
            <button onclick="loadSessionData()">Load Session</button>
            <button onclick="compareModels()">Compare Selected Models</button>
        </div>

        <div class="charts-container">
            <div class="chart-card">
                <h3>Tokens per Second Comparison</h3>
                <canvas id="tps-chart"></canvas>
            </div>
            <div class="chart-card">
                <h3>Quality vs RAM</h3>
                <canvas id="quality-ram-chart"></canvas>
            </div>
            <div class="chart-card">
                <h3>Quality vs TPS</h3>
                <canvas id="quality-tps-chart"></canvas>
            </div>
            <div class="chart-card">
                <h3>Resource Consumption Over Time</h3>
                <canvas id="resource-chart"></canvas>
            </div>
        </div>

        <div class="table-container">
            <h2>Model Results</h2>
            <table id="results-table">
                <thead>
                    <tr>
                        <th><input type="checkbox" id="select-all"></th>
                        <th>Model</th>
                        <th>Status</th>
                        <th>Quality</th>
                        <th>TPS</th>
                        <th>Avg RAM (MB)</th>
                        <th>Stability</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>

        <div class="recommendations" id="recommendations">
            <h2>Recommendations</h2>
            <div id="recommendations-list"></div>
        </div>
    </div>

    <script>
        let currentSessionId = null;
        let currentData = null;

        // Load sessions on page load
        fetch('/api/sessions')
            .then(r => r.json())
            .then(sessions => {
                const select = document.getElementById('session-select');
                sessions.forEach(s => {
                    const option = document.createElement('option');
                    option.value = s.id;
                    option.textContent = 'Session ' + s.id + ' - ' + s.started_at;
                    select.appendChild(option);
                });
            });

        document.getElementById('min-quality').addEventListener('input', function() {
            document.getElementById('quality-value').textContent = parseFloat(this.value).toFixed(1);
        });

        function loadSessionData() {
            const sessionId = document.getElementById('session-select').value;
            if (!sessionId) return;
            currentSessionId = sessionId;
            fetch('/api/session/' + sessionId)
                .then(r => r.json())
                .then(data => {
                    currentData = data;
                    renderTable(data.model_results || []);
                    populateCategories(data.model_results || []);
                });
        }

        function renderTable(models) {
            const tbody = document.querySelector('#results-table tbody');
            tbody.innerHTML = '';
            models.forEach(function(model) {
                const row = document.createElement('tr');
                const qs = model.quality_scores || {};
                const qualityVals = Object.values(qs).filter(v => v !== null);
                const avgQuality = qualityVals.length
                    ? (qualityVals.reduce((a, b) => a + b, 0) / qualityVals.length).toFixed(3)
                    : 'N/A';
                row.innerHTML =
                    '<td><input type="checkbox" class="model-select" data-model="' + model.model_name + '"></td>' +
                    '<td><strong>' + model.model_name + '</strong></td>' +
                    '<td>' + model.status + '</td>' +
                    '<td>' + avgQuality + '</td>' +
                    '<td>N/A</td>' +
                    '<td>N/A</td>' +
                    '<td>' + (model.robustness ? model.robustness.stability_score.toFixed(3) : 'N/A') + '</td>';
                tbody.appendChild(row);
            });
        }

        function populateCategories(models) {
            const categories = new Set();
            models.forEach(function(model) {
                if (model.quality_scores) {
                    Object.keys(model.quality_scores).forEach(function(cat) { categories.add(cat); });
                }
            });
            const select = document.getElementById('category-filter');
            select.innerHTML = '<option value="">All Categories</option>';
            categories.forEach(function(cat) {
                const option = document.createElement('option');
                option.value = cat;
                option.textContent = cat;
                select.appendChild(option);
            });
        }

        function compareModels() {
            const selected = Array.from(document.querySelectorAll('.model-select:checked'))
                .map(function(cb) { return cb.dataset.model; });
            if (selected.length < 2) {
                alert('Please select at least 2 models to compare');
                return;
            }
            console.log('Comparing models:', selected);
        }
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app(db: Database) -> Flask:
    """Create and configure the Flask dashboard application.

    Parameters
    ----------
    db:
        An initialised :class:`~ollama_benchmark.database.Database` instance.
        All routes read from this shared instance via the query interface only
        (Requirements 10.1, 14.4).

    Returns
    -------
    Flask
        Configured Flask application ready to be served.
    """
    app = Flask(__name__)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index() -> str:
        """Serve the main dashboard single-page application."""
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/sessions")
    def get_sessions() -> tuple:
        """Return all benchmark sessions as a JSON array.

        Response shape: ``[SessionSummary, ...]``
        """
        sessions = db.get_sessions()
        payload = [dataclasses.asdict(s) for s in sessions]
        return (
            json.dumps(payload, default=_default_serializer),
            200,
            {"Content-Type": "application/json"},
        )

    @app.route("/api/session/<int:session_id>")
    def get_session(session_id: int) -> tuple:
        """Return full session detail including hardware, config, and model results.

        Response shape: ``SessionDetail`` (nested dataclass as dict).
        Returns 404 with ``{"error": "..."}`` when the session is not found.
        """
        try:
            detail = db.get_session_detail(session_id)
            payload = dataclasses.asdict(detail)
            return (
                json.dumps(payload, default=_default_serializer),
                200,
                {"Content-Type": "application/json"},
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/api/compare")
    def compare_sessions() -> tuple:
        """Compare multiple sessions.

        Query parameter: ``?ids=1,2,3`` (comma-separated session IDs, at least 1).
        Returns 400 when the ``ids`` parameter is missing or malformed.
        """
        ids_param = request.args.get("ids", "").strip()
        if not ids_param:
            return jsonify({"error": "ids query parameter is required (e.g. ?ids=1,2,3)"}), 400

        try:
            session_ids = [int(i.strip()) for i in ids_param.split(",") if i.strip()]
        except ValueError:
            return jsonify({"error": "ids must be comma-separated integers"}), 400

        if not session_ids:
            return jsonify({"error": "ids query parameter must contain at least one id"}), 400

        try:
            report = db.compare_sessions(session_ids)
            payload = dataclasses.asdict(report)
            return (
                json.dumps(payload, default=_default_serializer),
                200,
                {"Content-Type": "application/json"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Error comparing sessions %s: %s", session_ids, exc)
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/timeseries/<int:model_run_id>")
    def get_timeseries(model_run_id: int) -> tuple:
        """Return resource time-series samples for a model run.

        Response shape: ``[ResourceSample, ...]``
        """
        samples = db.get_resource_timeseries(model_run_id)
        payload = [dataclasses.asdict(s) for s in samples]
        return (
            json.dumps(payload, default=_default_serializer),
            200,
            {"Content-Type": "application/json"},
        )

    return app


# ---------------------------------------------------------------------------
# Entry point (used by CLI — task 14.3)
# ---------------------------------------------------------------------------

def start_dashboard(db_path: str, port: int = 8080, host: str = "0.0.0.0") -> None:
    """Start the dashboard Flask server.

    Instantiates a :class:`~ollama_benchmark.database.Database` from *db_path*,
    creates the Flask app, and starts serving on *host*:*port*.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file.
    port:
        TCP port to listen on. Defaults to 8080.
    host:
        Host/interface to bind to. Defaults to ``"0.0.0.0"`` (all interfaces).
    """
    db = Database(db_path)
    app = create_app(db)
    logger.info("Starting Ollama Benchmark Dashboard on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False)
