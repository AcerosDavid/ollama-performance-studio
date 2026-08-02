"""
Unit tests for config loading.

Covers: load_config, BenchmarkConfig, ConfigNotFoundError, ConfigValidationError

Validates: Requirements 2.2, 2.3, 2.4
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from ollama_benchmark.config import BenchmarkConfig, load_config
from ollama_benchmark.models import ConfigNotFoundError, ConfigValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid YAML that satisfies all required BenchmarkConfig fields.
_VALID_CONFIG: dict = {
    "models": ["qwen:0.5b", "qwen:1.8b"],
    "prompts": {
        "reasoning": [{"text": "Is the sky blue?", "expected_answer": "Yes"}],
        "coding": [{"text": "Write a hello-world in Python."}],
    },
    "timeouts": {
        "download": 3600,
        "cold_start": 120,
        "inference": 300,
        "judge": 120,
    },
}


def _write_yaml(path: Path, data: dict) -> Path:
    """Write *data* as YAML to *path* and return *path*."""
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Valid YAML — returns a BenchmarkConfig with correct field values
# ---------------------------------------------------------------------------


class TestValidConfig:
    """load_config returns a fully-populated BenchmarkConfig from valid YAML."""

    def test_returns_benchmark_config_instance(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path / "config.yaml", _VALID_CONFIG)
        result = load_config(cfg_file)
        assert isinstance(result, BenchmarkConfig)

    def test_models_field_matches_yaml(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path / "config.yaml", _VALID_CONFIG)
        result = load_config(cfg_file)
        assert result.models == ["qwen:0.5b", "qwen:1.8b"]

    def test_prompts_categories_present(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path / "config.yaml", _VALID_CONFIG)
        result = load_config(cfg_file)
        assert "reasoning" in result.prompts
        assert "coding" in result.prompts

    def test_timeout_values_match_yaml(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path / "config.yaml", _VALID_CONFIG)
        result = load_config(cfg_file)
        assert result.timeouts.download == 3600
        assert result.timeouts.cold_start == 120
        assert result.timeouts.inference == 300
        assert result.timeouts.judge == 120

    def test_defaults_applied_when_omitted(self, tmp_path: Path) -> None:
        """Fields with defaults should be present even when absent from YAML."""
        cfg_file = _write_yaml(tmp_path / "config.yaml", _VALID_CONFIG)
        result = load_config(cfg_file)
        assert result.database_path == "benchmark.db"
        assert result.max_retries == 3
        assert result.ollama_base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# 2. Missing file — raises ConfigNotFoundError with the path in the message
# ---------------------------------------------------------------------------


class TestMissingFile:
    """load_config raises ConfigNotFoundError when the file does not exist."""

    def test_raises_config_not_found_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigNotFoundError):
            load_config(missing)

    def test_error_message_contains_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigNotFoundError, match="nonexistent.yaml"):
            load_config(missing)

    def test_is_subclass_of_file_not_found_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "also_missing.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(missing)


# ---------------------------------------------------------------------------
# 3. Invalid field type — raises ConfigValidationError mentioning the field
# ---------------------------------------------------------------------------


class TestInvalidFieldType:
    """load_config raises ConfigValidationError when a field has the wrong type."""

    def test_models_as_string_raises_validation_error(self, tmp_path: Path) -> None:
        bad = dict(_VALID_CONFIG, models="not_a_list")
        cfg_file = _write_yaml(tmp_path / "config.yaml", bad)
        with pytest.raises(ConfigValidationError):
            load_config(cfg_file)

    def test_models_as_string_error_mentions_field(self, tmp_path: Path) -> None:
        bad = dict(_VALID_CONFIG, models="not_a_list")
        cfg_file = _write_yaml(tmp_path / "config.yaml", bad)
        with pytest.raises(ConfigValidationError, match="models"):
            load_config(cfg_file)

    def test_prompts_as_list_raises_validation_error(self, tmp_path: Path) -> None:
        bad = dict(_VALID_CONFIG, prompts=["a", "b"])
        cfg_file = _write_yaml(tmp_path / "config.yaml", bad)
        with pytest.raises(ConfigValidationError):
            load_config(cfg_file)

    def test_timeouts_as_string_raises_validation_error(self, tmp_path: Path) -> None:
        bad = dict(_VALID_CONFIG, timeouts="fast")
        cfg_file = _write_yaml(tmp_path / "config.yaml", bad)
        with pytest.raises(ConfigValidationError):
            load_config(cfg_file)


# ---------------------------------------------------------------------------
# 4. Empty model list — raises ConfigValidationError
# ---------------------------------------------------------------------------


class TestEmptyModelList:
    """load_config raises ConfigValidationError when models is an empty list."""

    def test_empty_models_raises_validation_error(self, tmp_path: Path) -> None:
        bad = dict(_VALID_CONFIG, models=[])
        cfg_file = _write_yaml(tmp_path / "config.yaml", bad)
        with pytest.raises(ConfigValidationError):
            load_config(cfg_file)

    def test_empty_models_error_mentions_field(self, tmp_path: Path) -> None:
        bad = dict(_VALID_CONFIG, models=[])
        cfg_file = _write_yaml(tmp_path / "config.yaml", bad)
        with pytest.raises(ConfigValidationError, match="models"):
            load_config(cfg_file)


# ---------------------------------------------------------------------------
# 5. Timeout out of range — raises ConfigValidationError
# ---------------------------------------------------------------------------


class TestTimeoutOutOfRange:
    """Timeout values outside [1, 3600] raise ConfigValidationError."""

    def _cfg_with_timeout(self, tmp_path: Path, field: str, value: int) -> Path:
        timeouts = dict(_VALID_CONFIG["timeouts"], **{field: value})
        data = dict(_VALID_CONFIG, timeouts=timeouts)
        return _write_yaml(tmp_path / "config.yaml", data)

    def test_download_timeout_zero_raises(self, tmp_path: Path) -> None:
        cfg_file = self._cfg_with_timeout(tmp_path, "download", 0)
        with pytest.raises(ConfigValidationError):
            load_config(cfg_file)

    def test_inference_timeout_negative_raises(self, tmp_path: Path) -> None:
        cfg_file = self._cfg_with_timeout(tmp_path, "inference", -1)
        with pytest.raises(ConfigValidationError):
            load_config(cfg_file)

    def test_judge_timeout_exceeds_max_raises(self, tmp_path: Path) -> None:
        cfg_file = self._cfg_with_timeout(tmp_path, "judge", 3601)
        with pytest.raises(ConfigValidationError):
            load_config(cfg_file)

    def test_cold_start_timeout_at_boundary_1_passes(self, tmp_path: Path) -> None:
        """Boundary value 1 is valid."""
        cfg_file = self._cfg_with_timeout(tmp_path, "cold_start", 1)
        result = load_config(cfg_file)
        assert result.timeouts.cold_start == 1

    def test_inference_timeout_at_boundary_3600_passes(self, tmp_path: Path) -> None:
        """Boundary value 3600 is valid."""
        cfg_file = self._cfg_with_timeout(tmp_path, "inference", 3600)
        result = load_config(cfg_file)
        assert result.timeouts.inference == 3600


# ---------------------------------------------------------------------------
# 6. Path fallback — no path given → falls back to config.yaml in CWD
# ---------------------------------------------------------------------------


class TestPathFallback:
    """When no path is provided, load_config uses config.yaml in the CWD."""

    def test_falls_back_to_cwd_config_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Place a valid config.yaml in tmp_path and make it the CWD.
        _write_yaml(tmp_path / "config.yaml", _VALID_CONFIG)
        monkeypatch.chdir(tmp_path)

        result = load_config()  # no path argument
        assert isinstance(result, BenchmarkConfig)
        assert result.models == ["qwen:0.5b", "qwen:1.8b"]

    def test_fallback_missing_cwd_config_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CWD has no config.yaml → ConfigNotFoundError.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigNotFoundError):
            load_config()

    def test_fallback_error_mentions_config_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigNotFoundError, match="config.yaml"):
            load_config()
