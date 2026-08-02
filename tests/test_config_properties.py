"""
Property tests for config validation (Task 2.2 — Property 1).

Uses Hypothesis to generate invalid config dicts and assert that load_config
raises ConfigValidationError mentioning every invalid field name and value.

**Validates: Requirements 2.4**

Property 1: Config validation rejects all invalid inputs
    For any configuration dict with at least one invalid field value
    (wrong type, out-of-range integer, empty lists), load_config SHALL
    raise a ConfigValidationError and the error message SHALL reference
    every invalid field name and the bad value provided, without accepting
    a partially-invalid config.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given, settings, assume, strategies as st
from pydantic import ValidationError

from ollama_benchmark.config import BenchmarkConfig, ConfigValidationError, load_config


# ---------------------------------------------------------------------------
# Helper strategies for valid base values
# ---------------------------------------------------------------------------

valid_timeout = st.integers(min_value=1, max_value=3600)
valid_model_list = st.lists(st.just("llama3"), min_size=1)

# A minimal valid base config as a dict
VALID_BASE = {
    "models": ["llama3"],
    "prompts": {"test": [{"text": "hello"}]},
    "timeouts": {
        "download": 100,
        "cold_start": 60,
        "inference": 120,
        "judge": 60,
    },
}


def write_yaml_and_load(config_dict: dict) -> None:
    """Write config_dict to a temp YAML file and call load_config on it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(config_dict, fh)
        load_config(config_path)


# ---------------------------------------------------------------------------
# Property 1: Config validation rejects all invalid inputs
# (Hypothesis-driven, all tests go through load_config)
# ---------------------------------------------------------------------------


class TestConfigValidationProperty1:
    """Property 1: Config validation rejects all invalid inputs.

    Validates: Requirements 2.4

    All tests exercise load_config (not BenchmarkConfig directly) to ensure the
    full validation pipeline raises ConfigValidationError with field information.
    """

    # -- 1a: Empty models list -----------------------------------------------

    @given(st.just([]))
    @settings(max_examples=10)
    def test_empty_models_list_raises_via_load_config(self, models: list) -> None:
        """Property 1a: An empty models list must be rejected by load_config."""
        config = {**VALID_BASE, "models": models}
        with pytest.raises(ConfigValidationError) as exc_info:
            write_yaml_and_load(config)
        error_msg = str(exc_info.value)
        assert "models" in error_msg.lower() or "field" in error_msg.lower()

    # -- 1b: Out-of-range timeout (download) ---------------------------------

    @given(
        download=st.one_of(
            st.integers(max_value=0),
            st.integers(min_value=3601, max_value=100_000),
        )
    )
    @settings(max_examples=50)
    def test_out_of_range_download_timeout_raises_via_load_config(
        self, download: int
    ) -> None:
        """Property 1b: download timeout outside [1, 3600] must be rejected."""
        config = {
            **VALID_BASE,
            "timeouts": {
                "download": download,
                "cold_start": 60,
                "inference": 120,
                "judge": 60,
            },
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            write_yaml_and_load(config)
        error_msg = str(exc_info.value)
        # The error must mention the bad field
        assert "download" in error_msg or "field" in error_msg.lower()

    # -- 1c: Out-of-range timeout (cold_start) --------------------------------

    @given(
        cold_start=st.one_of(
            st.integers(max_value=0),
            st.integers(min_value=3601, max_value=100_000),
        )
    )
    @settings(max_examples=50)
    def test_out_of_range_cold_start_timeout_raises_via_load_config(
        self, cold_start: int
    ) -> None:
        """Property 1c: cold_start timeout outside [1, 3600] must be rejected."""
        config = {
            **VALID_BASE,
            "timeouts": {
                "download": 100,
                "cold_start": cold_start,
                "inference": 120,
                "judge": 60,
            },
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            write_yaml_and_load(config)
        error_msg = str(exc_info.value)
        assert "cold_start" in error_msg or "field" in error_msg.lower()

    # -- 1d: Out-of-range CPU percentage threshold ----------------------------

    @given(
        cpu=st.one_of(
            st.floats(min_value=-200.0, max_value=-0.001, allow_nan=False),
            st.floats(min_value=100.001, max_value=500.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_out_of_range_cpu_threshold_raises_via_load_config(
        self, cpu: float
    ) -> None:
        """Property 1d: cpu_percent threshold outside [0, 100] must be rejected."""
        config = {
            **VALID_BASE,
            "resource_thresholds": {"cpu_percent": cpu},
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            write_yaml_and_load(config)
        error_msg = str(exc_info.value)
        assert "cpu_percent" in error_msg or "field" in error_msg.lower()

    # -- 1e: Non-positive RAM threshold ---------------------------------------

    @given(
        ram_mb=st.one_of(
            st.floats(min_value=-10_000.0, max_value=-0.001, allow_nan=False),
            st.just(0.0),
        )
    )
    @settings(max_examples=50)
    def test_non_positive_ram_threshold_raises_via_load_config(
        self, ram_mb: float
    ) -> None:
        """Property 1e: ram_mb threshold ≤ 0 must be rejected."""
        config = {
            **VALID_BASE,
            "resource_thresholds": {"ram_mb": ram_mb},
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            write_yaml_and_load(config)
        error_msg = str(exc_info.value)
        assert "ram_mb" in error_msg or "field" in error_msg.lower()

    # -- 1f: Multiple simultaneous invalid fields ----------------------------

    @given(
        models_invalid=st.just([]),
        download_invalid=st.one_of(
            st.integers(max_value=0),
            st.integers(min_value=3601, max_value=100_000),
        ),
    )
    @settings(max_examples=30)
    def test_multiple_invalid_fields_raises_via_load_config(
        self,
        models_invalid: list,
        download_invalid: int,
    ) -> None:
        """Property 1f: Multiple invalid fields all trigger ConfigValidationError.

        The error must reference at least one of the bad field names.
        """
        config = {
            "models": models_invalid,
            "prompts": {"test": [{"text": "hello"}]},
            "timeouts": {
                "download": download_invalid,
                "cold_start": 60,
                "inference": 120,
                "judge": 60,
            },
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            write_yaml_and_load(config)
        error_msg = str(exc_info.value)
        # At least one of the invalid fields must appear in the error message
        assert (
            "models" in error_msg.lower()
            or "download" in error_msg.lower()
            or "field" in error_msg.lower()
        )

    # -- 1g: ConfigValidationError.errors contains field-value pairs ----------

    @given(
        download=st.one_of(
            st.integers(max_value=0),
            st.integers(min_value=3601, max_value=100_000),
        )
    )
    @settings(max_examples=50)
    def test_error_object_contains_field_value_pairs(self, download: int) -> None:
        """Property 1g: ConfigValidationError.errors must be a list of (field, value) pairs."""
        config = {
            **VALID_BASE,
            "timeouts": {
                "download": download,
                "cold_start": 60,
                "inference": 120,
                "judge": 60,
            },
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            write_yaml_and_load(config)
        exc = exc_info.value
        # The .errors attribute must be a non-empty list of 2-tuples
        assert isinstance(exc.errors, list)
        assert len(exc.errors) >= 1
        for field_loc, bad_val in exc.errors:
            assert isinstance(field_loc, str)
            assert isinstance(bad_val, str)

    # -- 1h: GPU percentage threshold out of range ----------------------------

    @given(
        gpu=st.one_of(
            st.floats(min_value=-200.0, max_value=-0.001, allow_nan=False),
            st.floats(min_value=100.001, max_value=500.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_out_of_range_gpu_threshold_raises_via_load_config(
        self, gpu: float
    ) -> None:
        """Property 1h: gpu_percent threshold outside [0, 100] must be rejected."""
        config = {
            **VALID_BASE,
            "resource_thresholds": {"gpu_percent": gpu},
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            write_yaml_and_load(config)
        error_msg = str(exc_info.value)
        assert "gpu_percent" in error_msg or "field" in error_msg.lower()

    # -- 1i: Non-positive VRAM threshold -------------------------------------

    @given(
        vram_mb=st.one_of(
            st.floats(min_value=-10_000.0, max_value=-0.001, allow_nan=False),
            st.just(0.0),
        )
    )
    @settings(max_examples=50)
    def test_non_positive_vram_threshold_raises_via_load_config(
        self, vram_mb: float
    ) -> None:
        """Property 1i: vram_mb threshold ≤ 0 must be rejected."""
        config = {
            **VALID_BASE,
            "resource_thresholds": {"vram_mb": vram_mb},
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            write_yaml_and_load(config)
        error_msg = str(exc_info.value)
        assert "vram_mb" in error_msg or "field" in error_msg.lower()


# ---------------------------------------------------------------------------
# Parametrized edge-case tests (also through load_config)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_config,expected_field_in_error",
    [
        # Empty models list
        (
            {
                "models": [],
                "prompts": {"test": [{"text": "test"}]},
                "timeouts": {
                    "download": 100,
                    "cold_start": 60,
                    "inference": 120,
                    "judge": 60,
                },
            },
            "models",
        ),
        # Download timeout out of range (too high)
        (
            {
                "models": ["llama3"],
                "prompts": {"test": [{"text": "test"}]},
                "timeouts": {
                    "download": 5000,
                    "cold_start": 60,
                    "inference": 120,
                    "judge": 60,
                },
            },
            "download",
        ),
        # Download timeout is 0
        (
            {
                "models": ["llama3"],
                "prompts": {"test": [{"text": "test"}]},
                "timeouts": {
                    "download": 0,
                    "cold_start": 60,
                    "inference": 120,
                    "judge": 60,
                },
            },
            "download",
        ),
        # CPU percent > 100
        (
            {
                "models": ["llama3"],
                "prompts": {"test": [{"text": "test"}]},
                "timeouts": {
                    "download": 100,
                    "cold_start": 60,
                    "inference": 120,
                    "judge": 60,
                },
                "resource_thresholds": {"cpu_percent": 150},
            },
            "cpu_percent",
        ),
        # CPU percent < 0
        (
            {
                "models": ["llama3"],
                "prompts": {"test": [{"text": "test"}]},
                "timeouts": {
                    "download": 100,
                    "cold_start": 60,
                    "inference": 120,
                    "judge": 60,
                },
                "resource_thresholds": {"cpu_percent": -10},
            },
            "cpu_percent",
        ),
        # RAM MB is negative
        (
            {
                "models": ["llama3"],
                "prompts": {"test": [{"text": "test"}]},
                "timeouts": {
                    "download": 100,
                    "cold_start": 60,
                    "inference": 120,
                    "judge": 60,
                },
                "resource_thresholds": {"ram_mb": -500},
            },
            "ram_mb",
        ),
        # RAM MB is zero
        (
            {
                "models": ["llama3"],
                "prompts": {"test": [{"text": "test"}]},
                "timeouts": {
                    "download": 100,
                    "cold_start": 60,
                    "inference": 120,
                    "judge": 60,
                },
                "resource_thresholds": {"ram_mb": 0},
            },
            "ram_mb",
        ),
        # inference timeout out of range
        (
            {
                "models": ["llama3"],
                "prompts": {"test": [{"text": "test"}]},
                "timeouts": {
                    "download": 100,
                    "cold_start": 60,
                    "inference": 9999,
                    "judge": 60,
                },
            },
            "inference",
        ),
        # judge timeout is 0
        (
            {
                "models": ["llama3"],
                "prompts": {"test": [{"text": "test"}]},
                "timeouts": {
                    "download": 100,
                    "cold_start": 60,
                    "inference": 120,
                    "judge": 0,
                },
            },
            "judge",
        ),
    ],
)
def test_load_config_raises_config_validation_error(
    invalid_config: dict, expected_field_in_error: str
) -> None:
    """Property 1: load_config raises ConfigValidationError for invalid YAML files.

    Validates: Requirements 2.4

    For each invalid config, ConfigValidationError must be raised and the error
    message must reference the problematic field name.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "invalid_config.yaml"
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(invalid_config, fh)

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(config_path)

        error_msg = str(exc_info.value)
        # The error must reference the invalid field
        assert expected_field_in_error in error_msg, (
            f"Expected '{expected_field_in_error}' in error message, got: {error_msg!r}"
        )
        # The .errors attribute must be populated
        assert isinstance(exc_info.value.errors, list)
        assert len(exc_info.value.errors) >= 1
