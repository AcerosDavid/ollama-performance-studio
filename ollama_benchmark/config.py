"""
Configuration models for ollama-performance-studio.

Defines Pydantic v2 models for benchmark configuration:
- PromptEntry: a single prompt with optional expected answer
- TimeoutConfig: per-operation timeout settings
- ResourceThresholds: optional resource alert thresholds
- BenchmarkConfig: root configuration model

Also exports custom exceptions used by the config loading layer:
- ConfigNotFoundError
- ConfigValidationError
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ConfigNotFoundError(FileNotFoundError):
    """Raised when the config file cannot be found at the given path."""


class ConfigValidationError(ValueError):
    """Raised when the config file contains invalid or out-of-range values."""

    def __init__(self, message: str, errors: list[tuple[str, str]] | None = None) -> None:
        super().__init__(message)
        self.errors: list[tuple[str, str]] = errors or []


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PromptEntry(BaseModel):
    """A single benchmark prompt, optionally paired with an expected answer."""

    model_config = ConfigDict(frozen=True)

    text: str
    expected_answer: Optional[str] = None


class TimeoutConfig(BaseModel):
    """Per-operation timeout values in seconds (each must be in [1, 3600])."""

    model_config = ConfigDict(frozen=True)

    download: int = 3600
    cold_start: int = 120
    inference: int = 15
    judge: int = 30

    @field_validator("download", "cold_start", "inference", "judge", mode="before")
    @classmethod
    def _validate_timeout(cls, value: object, info: object) -> object:
        """Validate that each timeout integer is in [1, 3600]."""
        if isinstance(value, int) and not (1 <= value <= 3600):
            field_name = info.field_name if hasattr(info, "field_name") else "timeout"
            raise ValueError(
                f"Field '{field_name}' has value {value!r} which is out of range [1, 3600]"
            )
        return value


class ResourceThresholds(BaseModel):
    """Optional resource alert thresholds. Percentage fields must be in [0, 100];
    MB fields must be positive when provided."""

    model_config = ConfigDict(frozen=True)

    cpu_percent: Optional[float] = None   # 0–100
    gpu_percent: Optional[float] = None   # 0–100
    ram_mb: Optional[float] = None        # positive
    vram_mb: Optional[float] = None       # positive

    @field_validator("cpu_percent", "gpu_percent", mode="before")
    @classmethod
    def _validate_percent(cls, value: object, info: object) -> object:
        """Validate that percentage fields are in [0, 100] when provided."""
        if value is None:
            return value
        if isinstance(value, (int, float)) and not (0 <= float(value) <= 100):
            field_name = info.field_name if hasattr(info, "field_name") else "percent"
            raise ValueError(
                f"Field '{field_name}' has value {value!r} which is out of range [0, 100]"
            )
        return value

    @field_validator("ram_mb", "vram_mb", mode="before")
    @classmethod
    def _validate_positive_mb(cls, value: object, info: object) -> object:
        """Validate that MB fields are positive when provided."""
        if value is None:
            return value
        if isinstance(value, (int, float)) and float(value) <= 0:
            field_name = info.field_name if hasattr(info, "field_name") else "mb"
            raise ValueError(
                f"Field '{field_name}' has value {value!r} which must be a positive number"
            )
        return value


class BenchmarkConfig(BaseModel):
    """Root configuration model for a benchmark session."""

    model_config = ConfigDict(frozen=True)

    models: List[str]
    prompts: Dict[str, List[PromptEntry]]
    timeouts: TimeoutConfig
    judge_model: Optional[str] = None
    resource_thresholds: Optional[ResourceThresholds] = None
    database_path: str = "benchmark.db"
    plugins_dir: Optional[str] = None
    max_retries: int = 3
    ollama_base_url: str = "http://localhost:11434"
    parallel_downloads: int = 3

    @field_validator("models")
    @classmethod
    def _validate_models_non_empty(cls, value: List[str]) -> List[str]:
        """Validate that the models list has at least one element."""
        if len(value) == 0:
            raise ValueError(
                "Field 'models' has value [] which must contain at least 1 model"
            )
        return value

    @field_validator("parallel_downloads")
    @classmethod
    def _validate_parallel_downloads(cls, value: int) -> int:
        if not (1 <= value <= 10):
            raise ValueError(
                f"Field 'parallel_downloads' has value {value!r} which is out of range [1, 10]"
            )
        return value


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


import sys  # noqa: E402  (late import kept close to its only use)

import yaml  # noqa: E402


def load_config(path: Optional["Path"] = None) -> BenchmarkConfig:
    """Load and validate a YAML benchmark configuration file.

    Parameters
    ----------
    path:
        Path to the YAML config file.  When *None* the function falls back to
        ``config.yaml`` in the current working directory.

    Returns
    -------
    BenchmarkConfig
        A fully-validated configuration instance.

    Raises
    ------
    ConfigNotFoundError
        If the file does not exist at the resolved path.  The path is also
        printed to *stderr* so CLI users can see what was missing.
    ConfigValidationError
        If the file exists but contains invalid or out-of-range field values.
        The exception message lists every offending field name together with
        the bad value that was supplied.
    """
    from pathlib import Path as _Path  # local alias to avoid shadowing

    resolved: _Path = _Path(path) if path is not None else _Path("config.yaml")

    if not resolved.exists():
        print(f"Config file not found: {resolved}", file=sys.stderr)
        raise ConfigNotFoundError(f"Config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    import pydantic  # noqa: PLC0415

    try:
        return BenchmarkConfig.model_validate(data)
    except pydantic.ValidationError as exc:
        lines: list[str] = []
        error_pairs: list[tuple[str, str]] = []
        for error in exc.errors():
            field_loc = ".".join(str(part) for part in error["loc"]) if error["loc"] else "<root>"
            bad_value = error.get("input", "<unknown>")
            bad_str = repr(bad_value)
            lines.append(f"  Field '{field_loc}' has invalid value {bad_str}: {error['msg']}")
            error_pairs.append((field_loc, bad_str))
        message = "Configuration validation failed:\n" + "\n".join(lines)
        raise ConfigValidationError(message, errors=error_pairs) from exc
