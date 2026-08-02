"""
Tests for InferenceEngine — property-based and unit tests.

Tasks 6.3, 6.4, 6.5
Validates: Requirements 5.2, 5.3, 5.4, 5.5, 5.6
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_benchmark.config import PromptEntry
from ollama_benchmark.inference_engine import InferenceEngine
from ollama_benchmark.models import InferenceResult


# ---------------------------------------------------------------------------
# Helpers / fake HTTP stream
# ---------------------------------------------------------------------------

def _make_stream_lines(*chunks: dict) -> list[str]:
    """Encode a sequence of Ollama-style JSON chunks as newline strings."""
    return [json.dumps(c) for c in chunks]


def _make_async_line_iter(lines: list[str]) -> AsyncMock:
    """Return an async iterator that yields the given lines one by one."""

    async def _aiter_lines() -> AsyncIterator[str]:
        for line in lines:
            yield line

    mock = MagicMock()
    mock.aiter_lines = _aiter_lines
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.raise_for_status = MagicMock()
    return mock


def _make_client_context(response_mock: MagicMock) -> MagicMock:
    """Wrap a response mock in the two-level context manager that httpx uses:
    ``async with client.stream(...) as response``."""
    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=response_mock)
    stream_ctx.__aexit__ = AsyncMock(return_value=False)

    client_mock = MagicMock()
    client_mock.stream = MagicMock(return_value=stream_ctx)
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=False)
    return client_mock


# ---------------------------------------------------------------------------
# Task 6.3 — Property 2: inference metrics derive correctly from Ollama fields
# **Validates: Requirements 5.2, 5.3**
# ---------------------------------------------------------------------------


@given(
    eval_count=st.integers(min_value=1, max_value=10_000),
    eval_duration=st.integers(min_value=1, max_value=10 ** 12),
)
@settings(max_examples=200)
def test_tokens_per_second_formula(eval_count: int, eval_duration: int) -> None:
    """
    **Validates: Requirements 5.2, 5.3**

    Property: tokens_per_second and avg_inter_token_ms are derived
    from eval_count / (eval_duration / 1e9) and
    eval_duration / eval_count / 1e6 respectively, within 0.001 % tolerance.

    We verify this through the engine by constructing a fake Ollama stream
    whose final chunk carries the given eval_count and eval_duration values.
    """
    import asyncio

    chunks = [
        {"response": "hello", "done": False},
        {
            "response": "",
            "done": True,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
        },
    ]
    lines = _make_stream_lines(*chunks)
    response_mock = _make_async_line_iter(lines)
    client_mock = _make_client_context(response_mock)

    with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
        engine = InferenceEngine()
        result: InferenceResult = asyncio.get_event_loop().run_until_complete(
            engine.run_prompt("test-model", "test prompt", timeout=30)
        )

    assert result.tokens_per_second is not None
    assert result.avg_inter_token_ms is not None

    expected_tps = eval_count / (eval_duration / 1e9)
    expected_inter = eval_duration / eval_count / 1e6

    # Assert within 0.001 % relative tolerance
    assert abs(result.tokens_per_second - expected_tps) / expected_tps < 0.00001, (
        f"tps mismatch: got {result.tokens_per_second}, expected {expected_tps}"
    )
    assert abs(result.avg_inter_token_ms - expected_inter) / expected_inter < 0.00001, (
        f"inter_token_ms mismatch: got {result.avg_inter_token_ms}, expected {expected_inter}"
    )


# ---------------------------------------------------------------------------
# Task 6.4 — Property 4: average latency correctness
# **Validates: Requirements 5.5, 5.6**
# ---------------------------------------------------------------------------


@given(
    latencies=st.lists(
        st.one_of(
            st.none(),
            st.floats(min_value=0, max_value=60_000, allow_nan=False),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=200)
def test_avg_latency(latencies: list[Optional[float]]) -> None:
    """
    **Validates: Requirements 5.5, 5.6**

    Feature: ollama-llm-benchmark, Property 4: Average latency correctness

    Property: run_all_prompts sets last_avg_latency_ms to the mean of
    non-None total_response_ms values, or 0.0 when all are None.

    For any list of prompt total_response_ms values (where some may be
    None for timeouts), the persisted avg_latency_ms SHALL equal the
    arithmetic mean of the non-None values when at least one prompt
    completed, and SHALL equal 0 when all prompts timed out.
    """
    import asyncio

    # Build mock run_prompt results that return pre-set latencies
    results = [
        InferenceResult(
            prompt_text="p",
            response_text=None if lat is None else "r",
            ttft_ms=None,
            total_response_ms=lat,
            tokens_generated=None,
            tokens_per_second=None,
            avg_inter_token_ms=None,
            timed_out=(lat is None),
        )
        for lat in latencies
    ]

    engine = InferenceEngine()

    # Patch run_prompt to return results in order
    call_count = 0

    async def _fake_run_prompt(*args, **kwargs) -> InferenceResult:
        nonlocal call_count
        r = results[call_count]
        call_count += 1
        return r

    engine.run_prompt = _fake_run_prompt  # type: ignore[method-assign]

    prompts = [("cat", PromptEntry(text=f"prompt {i}")) for i in range(len(latencies))]

    asyncio.get_event_loop().run_until_complete(
        engine.run_all_prompts("test-model", prompts, timeout=30)
    )

    non_none = [lat for lat in latencies if lat is not None]
    expected = sum(non_none) / len(non_none) if non_none else 0.0

    assert abs(engine.last_avg_latency_ms - expected) < 1e-9 or (
        expected == 0.0 and engine.last_avg_latency_ms == 0.0
    ), (
        f"avg_latency_ms mismatch: got {engine.last_avg_latency_ms}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Task 6.5 — Unit tests
# **Validates: Requirements 5.2, 5.4**
# ---------------------------------------------------------------------------


class TestTTFTMeasurement:
    """TTFT is measured on the first non-empty response chunk.
    Validates: Requirements 5.2, 5.4
    """

    async def test_ttft_positive_and_less_than_total(self) -> None:
        """WHEN the stream sends two chunks THEN ttft_ms > 0 and < total_response_ms."""
        chunks = [
            {"response": "first token", "done": False},
            {
                "response": " second token",
                "done": True,
                "eval_count": 2,
                "eval_duration": 500_000_000,  # 0.5 s in nanoseconds
            },
        ]
        lines = _make_stream_lines(*chunks)
        response_mock = _make_async_line_iter(lines)
        client_mock = _make_client_context(response_mock)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        assert result.ttft_ms is not None, "ttft_ms should be set"
        assert result.total_response_ms is not None, "total_response_ms should be set"
        assert result.ttft_ms > 0, f"Expected ttft_ms > 0, got {result.ttft_ms}"
        assert result.ttft_ms < result.total_response_ms, (
            f"Expected ttft_ms < total_response_ms, "
            f"got ttft={result.ttft_ms} total={result.total_response_ms}"
        )

    async def test_ttft_not_set_when_no_response_content(self) -> None:
        """WHEN all chunks have empty response fields THEN ttft_ms is None."""
        chunks = [
            {"response": "", "done": False},
            {"response": "", "done": True, "eval_count": 0, "eval_duration": 100_000_000},
        ]
        lines = _make_stream_lines(*chunks)
        response_mock = _make_async_line_iter(lines)
        client_mock = _make_client_context(response_mock)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        assert result.ttft_ms is None


class TestTimeoutPath:
    """httpx.TimeoutException produces timed_out=True with all metrics None.
    Validates: Requirements 5.4
    """

    async def test_timeout_sets_timed_out_flag(self) -> None:
        """WHEN httpx raises TimeoutException THEN result.timed_out is True."""
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(
            side_effect=httpx.TimeoutException("read timed out")
        )
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client_mock = MagicMock()
        client_mock.stream = MagicMock(return_value=stream_ctx)
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=5)

        assert result.timed_out is True
        assert result.ttft_ms is None
        assert result.total_response_ms is None
        assert result.tokens_generated is None
        assert result.tokens_per_second is None
        assert result.avg_inter_token_ms is None

    async def test_timeout_sets_error_message(self) -> None:
        """WHEN timeout occurs THEN result.error contains the exception message."""
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(
            side_effect=httpx.TimeoutException("connection timed out")
        )
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client_mock = MagicMock()
        client_mock.stream = MagicMock(return_value=stream_ctx)
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=5)

        assert result.error is not None
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()


class TestErrorPath:
    """Generic exceptions produce error field set and timed_out=False.
    Validates: Requirements 5.4
    """

    async def test_generic_exception_sets_error(self) -> None:
        """WHEN a generic Exception is raised THEN result.error is set."""
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(
            side_effect=RuntimeError("unexpected connection failure")
        )
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client_mock = MagicMock()
        client_mock.stream = MagicMock(return_value=stream_ctx)
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        assert result.error is not None
        assert "unexpected connection failure" in result.error
        assert result.timed_out is False

    async def test_generic_exception_all_metrics_none(self) -> None:
        """WHEN a generic Exception is raised THEN all metric fields are None."""
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(side_effect=OSError("network unreachable"))
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client_mock = MagicMock()
        client_mock.stream = MagicMock(return_value=stream_ctx)
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        assert result.ttft_ms is None
        assert result.total_response_ms is None
        assert result.tokens_generated is None
        assert result.tokens_per_second is None
        assert result.avg_inter_token_ms is None


class TestRunAllPrompts:
    """Tests for run_all_prompts sequential execution and result ordering.
    Validates: Requirements 5.1, 5.5, 5.6
    """

    async def test_run_all_prompts_maintains_order(self) -> None:
        """WHEN running multiple prompts THEN results are returned in config-defined order."""
        # Create prebuilt results with distinct response texts
        results_list = [
            InferenceResult(
                prompt_text="prompt 1",
                response_text="answer 1",
                ttft_ms=10.0,
                total_response_ms=100.0,
                tokens_generated=5,
                tokens_per_second=50.0,
                avg_inter_token_ms=10.0,
                timed_out=False,
            ),
            InferenceResult(
                prompt_text="prompt 2",
                response_text="answer 2",
                ttft_ms=10.0,
                total_response_ms=100.0,
                tokens_generated=5,
                tokens_per_second=50.0,
                avg_inter_token_ms=10.0,
                timed_out=False,
            ),
            InferenceResult(
                prompt_text="prompt 3",
                response_text="answer 3",
                ttft_ms=10.0,
                total_response_ms=100.0,
                tokens_generated=5,
                tokens_per_second=50.0,
                avg_inter_token_ms=10.0,
                timed_out=False,
            ),
        ]

        call_count = 0

        async def _fake_run_prompt(*args, **kwargs):
            nonlocal call_count
            r = results_list[call_count]
            call_count += 1
            return r

        engine = InferenceEngine()
        engine.run_prompt = _fake_run_prompt  # type: ignore[method-assign]

        prompts = [
            ("category1", PromptEntry(text="prompt 1")),
            ("category2", PromptEntry(text="prompt 2")),
            ("category3", PromptEntry(text="prompt 3")),
        ]

        results = await engine.run_all_prompts("test-model", prompts, timeout=30)

        assert len(results) == 3
        assert results[0].response_text == "answer 1"
        assert results[1].response_text == "answer 2"
        assert results[2].response_text == "answer 3"

    async def test_run_all_prompts_persists_results_to_db(self) -> None:
        """WHEN run_prompt is called with model_run_id and db THEN results are persisted."""
        chunks = [
            {"response": "test response", "done": True, "eval_count": 10, "eval_duration": 1_000_000_000},
        ]
        lines = _make_stream_lines(*chunks)
        response_mock = _make_async_line_iter(lines)
        client_mock = _make_client_context(response_mock)

        # Mock database
        mock_db = MagicMock()

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine(db=mock_db)
            result = await engine.run_prompt(
                "test-model",
                "hello",
                timeout=30,
                category="reasoning",
                model_run_id=42,
            )

        # Verify db.save_inference_result was called with the result
        mock_db.save_inference_result.assert_called_once_with(42, result, "reasoning")

    async def test_run_all_prompts_db_error_does_not_interrupt(self) -> None:
        """WHEN db.save_inference_result fails THEN inference continues without raising."""
        chunks = [
            {"response": "test response", "done": True, "eval_count": 5, "eval_duration": 500_000_000},
        ]
        lines = _make_stream_lines(*chunks)
        response_mock = _make_async_line_iter(lines)
        client_mock = _make_client_context(response_mock)

        # Mock database to raise an exception
        mock_db = MagicMock()
        mock_db.save_inference_result.side_effect = RuntimeError("DB connection failed")

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine(db=mock_db)
            # Should not raise despite DB error
            result = await engine.run_prompt(
                "test-model",
                "hello",
                timeout=30,
                category="reasoning",
                model_run_id=42,
            )

        assert result.response_text == "test response"
        mock_db.save_inference_result.assert_called_once()

    async def test_run_all_prompts_computes_avg_latency_from_multiple_prompts(self) -> None:
        """WHEN run_all_prompts completes THEN last_avg_latency_ms is computed correctly."""
        import asyncio

        # Create 3 mock results with specific latencies
        latencies = [100.0, 200.0, 150.0]
        results = [
            InferenceResult(
                prompt_text=f"prompt {i}",
                response_text=f"response {i}",
                ttft_ms=10.0,
                total_response_ms=lat,
                tokens_generated=5,
                tokens_per_second=50.0,
                avg_inter_token_ms=10.0,
                timed_out=False,
            )
            for i, lat in enumerate(latencies)
        ]

        engine = InferenceEngine()

        call_count = 0

        async def _fake_run_prompt(*args, **kwargs) -> InferenceResult:
            nonlocal call_count
            r = results[call_count]
            call_count += 1
            return r

        engine.run_prompt = _fake_run_prompt  # type: ignore[method-assign]

        prompts = [("cat", PromptEntry(text=f"prompt {i}")) for i in range(len(latencies))]

        await engine.run_all_prompts("test-model", prompts, timeout=30)

        expected_avg = (100.0 + 200.0 + 150.0) / 3
        assert abs(engine.last_avg_latency_ms - expected_avg) < 1e-9

    async def test_run_all_prompts_skips_timeout_results_in_avg_latency(self) -> None:
        """WHEN some prompts timeout THEN avg_latency_ms excludes their None values."""
        import asyncio

        # Create results: one timeout, two successful
        results = [
            InferenceResult(
                prompt_text="prompt 1",
                response_text=None,
                ttft_ms=None,
                total_response_ms=None,
                tokens_generated=None,
                tokens_per_second=None,
                avg_inter_token_ms=None,
                timed_out=True,
            ),
            InferenceResult(
                prompt_text="prompt 2",
                response_text="response 2",
                ttft_ms=10.0,
                total_response_ms=100.0,
                tokens_generated=5,
                tokens_per_second=50.0,
                avg_inter_token_ms=10.0,
                timed_out=False,
            ),
            InferenceResult(
                prompt_text="prompt 3",
                response_text="response 3",
                ttft_ms=15.0,
                total_response_ms=200.0,
                tokens_generated=10,
                tokens_per_second=50.0,
                avg_inter_token_ms=10.0,
                timed_out=False,
            ),
        ]

        engine = InferenceEngine()

        call_count = 0

        async def _fake_run_prompt(*args, **kwargs) -> InferenceResult:
            nonlocal call_count
            r = results[call_count]
            call_count += 1
            return r

        engine.run_prompt = _fake_run_prompt  # type: ignore[method-assign]

        prompts = [("cat", PromptEntry(text=f"prompt {i}")) for i in range(len(results))]

        await engine.run_all_prompts("test-model", prompts, timeout=30)

        # Average should be (100 + 200) / 2 = 150, excluding the timeout
        expected_avg = (100.0 + 200.0) / 2
        assert abs(engine.last_avg_latency_ms - expected_avg) < 1e-9


class TestMetricExtraction:
    """Tests for correct extraction of metrics from Ollama response fields.
    Validates: Requirements 5.2, 5.3
    """

    async def test_eval_count_zero_tokens_generated_set(self) -> None:
        """WHEN eval_count is 0 THEN tokens_generated is 0 but tokens_per_second is None."""
        chunks = [
            {"response": "", "done": True, "eval_count": 0, "eval_duration": 100_000_000},
        ]
        lines = _make_stream_lines(*chunks)
        response_mock = _make_async_line_iter(lines)
        client_mock = _make_client_context(response_mock)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        assert result.tokens_generated == 0
        # tokens_per_second should be None since eval_count is 0 (division by zero guard)
        assert result.tokens_per_second is None

    async def test_eval_duration_missing_metrics_partial(self) -> None:
        """WHEN eval_duration is missing THEN tokens_generated is set but rates are None."""
        chunks = [
            {"response": "hello", "done": True, "eval_count": 5},
        ]
        lines = _make_stream_lines(*chunks)
        response_mock = _make_async_line_iter(lines)
        client_mock = _make_client_context(response_mock)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        assert result.tokens_generated == 5
        assert result.tokens_per_second is None
        assert result.avg_inter_token_ms is None

    async def test_response_text_concatenates_all_chunks(self) -> None:
        """WHEN multiple chunks are streamed THEN response_text is concatenation."""
        chunks = [
            {"response": "Hello", "done": False},
            {"response": " ", "done": False},
            {"response": "world", "done": True, "eval_count": 3, "eval_duration": 300_000_000},
        ]
        lines = _make_stream_lines(*chunks)
        response_mock = _make_async_line_iter(lines)
        client_mock = _make_client_context(response_mock)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        assert result.response_text == "Hello world"

    async def test_empty_response_when_no_chunks(self) -> None:
        """WHEN stream has only empty chunks THEN response_text is None."""
        chunks = [
            {"response": "", "done": False},
            {"response": "", "done": True, "eval_count": 0, "eval_duration": 50_000_000},
        ]
        lines = _make_stream_lines(*chunks)
        response_mock = _make_async_line_iter(lines)
        client_mock = _make_client_context(response_mock)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        assert result.response_text is None

    async def test_response_extraction_on_error_path(self) -> None:
        """WHEN error occurs THEN response_text is None and error is captured."""
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(
            side_effect=ValueError("Invalid JSON in response")
        )
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client_mock = MagicMock()
        client_mock.stream = MagicMock(return_value=stream_ctx)
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        assert result.response_text is None
        assert result.error is not None
        assert "Invalid JSON" in result.error

    async def test_malformed_json_line_skipped(self) -> None:
        """WHEN stream contains a non-JSON line THEN it is skipped and processing continues."""
        # Simulate a stream with a malformed line in the middle
        async def _aiter_lines():
            yield json.dumps({"response": "first", "done": False})
            yield "this is not json at all!!!"
            yield json.dumps({"response": " second", "done": True, "eval_count": 2, "eval_duration": 200_000_000})

        response_mock = MagicMock()
        response_mock.aiter_lines = _aiter_lines
        response_mock.__aenter__ = AsyncMock(return_value=response_mock)
        response_mock.__aexit__ = AsyncMock(return_value=False)
        response_mock.raise_for_status = MagicMock()

        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=response_mock)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client_mock = MagicMock()
        client_mock.stream = MagicMock(return_value=stream_ctx)
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("ollama_benchmark.inference_engine.httpx.AsyncClient", return_value=client_mock):
            engine = InferenceEngine()
            result = await engine.run_prompt("test-model", "hello", timeout=30)

        # Should still concatenate the valid chunks
        assert result.response_text == "first second"
        assert result.tokens_generated == 2
