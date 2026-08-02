"""
Data transfer objects (DTOs) and shared types for ollama-benchmark.

All dataclasses used to shuttle data between components are defined here,
along with the ModelStatus literal type, the PullResult helper, and the
shared exception hierarchy.

Exceptions
----------
ConfigNotFoundError   — imported from config.py and re-exported here
ConfigValidationError — imported from config.py and re-exported here
OllamaUnavailableError — raised when Ollama is not running / not reachable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional

# Re-export config exceptions so callers can do:
#   from ollama_benchmark.models import ConfigNotFoundError, ConfigValidationError
from ollama_benchmark.config import ConfigNotFoundError, ConfigValidationError

__all__ = [
    # Exceptions
    "ConfigNotFoundError",
    "ConfigValidationError",
    "OllamaUnavailableError",
    # Status literal
    "ModelStatus",
    # DTOs
    "HardwareInfo",
    "ResourceSample",
    "ResourceSummary",
    "InferenceResult",
    "ModelResult",
    "RobustnessMetrics",
    "EfficiencyIndices",
    "ErrorEntry",
    "SessionSummary",
    "SessionDetail",
    "PromptResult",
    "Recommendation",
    "ComparisonReport",
    "PullResult",
    "SessionResult",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is not running or not reachable."""


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Valid status values for a model run.
ModelStatus = Literal["completed", "failed", "skipped", "incomplete"]


# ---------------------------------------------------------------------------
# Hardware & resource DTOs
# ---------------------------------------------------------------------------


@dataclass
class HardwareInfo:
    """System hardware characteristics captured at session start."""

    os: str
    python_version: str
    ollama_version: str
    cpu_cores: int
    cpu_mhz: float
    ram_mb: float
    has_gpu: bool
    vram_mb: float
    disk_free_mb: float


@dataclass
class ResourceSample:
    """One-second snapshot of resource utilisation."""

    timestamp: datetime
    cpu_per_core: List[float]          # percent per core
    ram_mb: float
    gpu_percent: Optional[float]
    vram_mb: Optional[float]
    cpu_temp_c: Optional[float]
    gpu_temp_c: Optional[float]
    power_watts: Optional[float]


@dataclass
class ResourceSummary:
    """Aggregated resource statistics for a model run, plus the raw time series."""

    max_cpu_percent: float
    avg_cpu_percent: float
    max_ram_mb: float
    avg_ram_mb: float
    max_gpu_percent: Optional[float]
    avg_gpu_percent: Optional[float]
    max_vram_mb: Optional[float]
    avg_vram_mb: Optional[float]
    max_temp_cpu_c: Optional[float]
    max_temp_gpu_c: Optional[float]
    avg_power_watts: Optional[float]
    samples: List[ResourceSample] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inference DTOs
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Metrics and output for a single prompt/response cycle."""

    prompt_text: str
    response_text: Optional[str]
    ttft_ms: Optional[float]           # time to first streaming chunk
    total_response_ms: Optional[float]
    tokens_generated: Optional[int]    # eval_count from Ollama
    tokens_per_second: Optional[float]
    avg_inter_token_ms: Optional[float]
    timed_out: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Quality & efficiency DTOs
# ---------------------------------------------------------------------------


@dataclass
class RobustnessMetrics:
    """Stability and error counts for a single model run."""

    total_errors: int
    total_timeouts: int
    oom_count: int
    restart_count: int
    incomplete_prompts: int
    stability_score: float             # completed / total


@dataclass
class EfficiencyIndices:
    """Composite efficiency indices relating quality to resource consumption."""

    quality_per_ram: Optional[float]
    quality_per_latency: Optional[float]
    quality_per_cpu: Optional[float]
    quality_per_disk: Optional[float]
    tps_per_gb_ram: Optional[float]
    quality_per_energy: Optional[float]
    # Normalized versions populated by ScoreEngine after all models are scored
    norm_quality_per_ram: Optional[float] = None
    norm_quality_per_latency: Optional[float] = None
    norm_quality_per_cpu: Optional[float] = None
    norm_quality_per_disk: Optional[float] = None
    norm_tps_per_gb_ram: Optional[float] = None
    norm_quality_per_energy: Optional[float] = None


@dataclass
class ErrorEntry:
    """A single error log entry associated with a model run."""

    error_type: str
    message: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Model run DTO
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Complete result for one model within a benchmark session."""

    model_name: str
    status: ModelStatus
    download_time_s: Optional[float]
    model_size_gb: Optional[float]
    cold_start_s: Optional[float]
    inference_results: List[InferenceResult]
    resource_summary: Optional[ResourceSummary]
    quality_scores: Dict[str, Optional[float]]   # category -> score
    efficiency_indices: Optional[EfficiencyIndices]
    robustness: RobustnessMetrics
    error_log: List[ErrorEntry] = field(default_factory=list)
    # Populated when loaded from DB or after persistence
    model_run_id: Optional[int] = None
    overall_rank: Optional[float] = None
    quality_score: Optional[float] = None
    avg_tps: Optional[float] = None
    avg_ttft_ms: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    avg_ram_mb: Optional[float] = None
    resource_timeseries: List[ResourceSample] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session / database query DTOs
# ---------------------------------------------------------------------------


@dataclass
class SessionResult:
    """Aggregated outcome of a full benchmark session."""

    session_id: int
    status: str  # completed | interrupted
    model_results: List[ModelResult] = field(default_factory=list)


@dataclass
class SessionSummary:
    """Lightweight session row returned by ``Database.get_sessions``."""

    id: int
    started_at: datetime
    finished_at: Optional[datetime]
    status: str          # running | completed | interrupted
    model_count: int


@dataclass
class SessionDetail:
    """Full session data including hardware, config snapshot, and model results."""

    summary: SessionSummary
    hardware: HardwareInfo
    config_snapshot: dict
    model_results: List[ModelResult]


@dataclass
class PromptResult:
    """Per-prompt result row as returned by ``Database.get_prompt_results``."""

    id: int
    model_run_id: int
    category: str
    prompt_text: str
    response_text: Optional[str]
    ttft_ms: Optional[float]
    total_ms: Optional[float]
    tokens_generated: Optional[int]
    tokens_per_second: Optional[float]
    avg_inter_token_ms: Optional[float]
    quality_score: Optional[float]
    timed_out: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# Recommendation & comparison DTOs
# ---------------------------------------------------------------------------


@dataclass
class Recommendation:
    """A single automatic recommendation for a hardware/use-case profile."""

    profile: str          # e.g. "8gb_ram", "coding", "fastest"
    model_name: str
    justification: str


@dataclass
class ComparisonReport:
    """Cross-session comparison data returned by ``Database.compare_sessions``."""

    session_ids: List[int]
    models: List[str]
    metrics: Dict[str, Dict[str, Optional[float]]]  # metric_name -> model_name -> value


# ---------------------------------------------------------------------------
# Model lifecycle helper
# ---------------------------------------------------------------------------


@dataclass
class PullResult:
    """Result of an ``ollama pull`` operation."""

    model_name: str
    success: bool
    download_time_s: Optional[float]
    model_size_gb: Optional[float]
    error: Optional[str] = None
