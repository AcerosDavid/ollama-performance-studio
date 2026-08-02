"""
Tests for QualityEvaluator — property-based and unit tests.

Tasks 9.5, 9.6
Validates: Requirements 6.2, 6.3, 6.5, 14.5
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ollama_benchmark.quality_evaluator import (
    QualityEvaluator,
    _token_overlap_f1,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluator(
    judge_model: str | None = None,
    base_url: str = "http://localhost:11434",
    plugins_dir: str | None = None,
    judge_timeout: int = 10,
) -> QualityEvaluator:
    """Build a QualityEvaluator with mocked sentence-transformers."""
    ev = QualityEvaluator(
        judge_model=judge_model,
        base_url=base_url,
        plugins_dir=plugins_dir,
        judge_timeout=judge_timeout,
    )
    # Disable lazy embedding load by default
    ev._st_model = None
    return ev


def _make_evaluator_with_embedder(fixed_score: float = 0.75) -> QualityEvaluator:
    """Build an evaluator whose sentence-transformer always returns a fixed score."""
    ev = _make_evaluator()

    # Patch the cosine similarity method directly
    ev._cosine_similarity = MagicMock(return_value=fixed_score)  # type: ignore
    return ev


# ---------------------------------------------------------------------------
# Task 9.5 — Property 9: Quality score always in [0.0, 1.0]
# Validates: Requirements 6.2, 6.3
# ---------------------------------------------------------------------------

@given(
    question=st.text(min_size=0, max_size=200),
    response=st.text(min_size=0, max_size=200),
    expected=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
    category=st.sampled_from(["reasoning", "coding", "math", "general", "conversation"]),
)
@settings(max_examples=100, deadline=None)  # deadline=None: first run may load sentence-transformer
def test_quality_score_range_property(
    question: str, response: str, expected: Optional[str], category: str
) -> None:
    """
    Property 9: Quality score is always in valid range [0.0, 1.0].

    Validates: Requirements 6.2, 6.3
    """
    ev = _make_evaluator()  # no judge, no embedder → token-overlap or None

    score = ev.evaluate(
        question=question,
        response=response,
        expected=expected,
        category=category,
    )

    if score is not None:
        assert 0.0 <= score <= 1.0, (
            f"Score {score} out of range for question={question!r}, "
            f"response={response!r}, expected={expected!r}, category={category}"
        )


@given(
    reference=st.text(min_size=1, max_size=200),
    hypothesis=st.text(min_size=1, max_size=200),
)
@settings(max_examples=200)
def test_token_overlap_f1_range_property(reference: str, hypothesis: str) -> None:
    """
    Property 9b: Token-overlap F1 is always in [0.0, 1.0].

    Validates: Requirements 6.2
    """
    score = _token_overlap_f1(reference, hypothesis)
    assert 0.0 <= score <= 1.0


@given(
    fixed_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    question=st.text(min_size=1, max_size=50),
    response=st.text(min_size=1, max_size=50),
    expected=st.text(min_size=1, max_size=50),
)
@settings(max_examples=100)
def test_expected_answer_score_range_property(
    fixed_score: float,
    question: str,
    response: str,
    expected: str,
) -> None:
    """
    Property 9c: Expected-answer cosine scores in [0.0, 1.0] regardless of fixed embedder output.

    Validates: Requirements 6.2
    """
    ev = _make_evaluator_with_embedder(fixed_score=fixed_score)

    score = ev.evaluate(
        question=question,
        response=response,
        expected=expected,
        category="reasoning",  # semantic category
    )

    assert score is not None
    assert 0.0 <= score <= 1.0


@given(
    judge_score=st.floats(min_value=-2.0, max_value=3.0, allow_nan=False),
)
@settings(max_examples=50)
def test_judge_score_clamped_to_range(judge_score: float) -> None:
    """
    Property 9d: Even if judge returns out-of-range float, result is clamped to [0, 1].

    Validates: Requirements 6.3
    """
    # Use _parse_judge_score which applies clamping
    raw_json = json.dumps({"score": judge_score})
    result = QualityEvaluator._parse_judge_score(raw_json)

    if result is not None:
        assert 0.0 <= result <= 1.0


@given(
    scores=st.dictionaries(
        keys=st.sampled_from(["reasoning", "coding", "math", "conversation"]),
        values=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=100)
def test_global_score_range_property(scores: dict) -> None:
    """
    Property 9e: compute_global_score always returns [0.0, 1.0].

    Validates: Requirements 6.4
    """
    ev = _make_evaluator()
    global_score = ev.compute_global_score(scores)
    assert 0.0 <= global_score <= 1.0


# ---------------------------------------------------------------------------
# Task 9.6 — Unit tests for quality evaluator
# Validates: Requirements 6.2, 6.3, 6.5, 14.5
# ---------------------------------------------------------------------------

class TestTokenOverlapF1:
    """Token-overlap F1 for code/math categories."""

    def test_identical_text_scores_1(self) -> None:
        assert _token_overlap_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap_scores_0(self) -> None:
        assert _token_overlap_f1("hello world", "foo bar") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        score = _token_overlap_f1("hello world", "hello foo")
        assert 0.0 < score < 1.0

    def test_empty_ref_and_empty_hyp_scores_1(self) -> None:
        assert _token_overlap_f1("", "") == pytest.approx(1.0)

    def test_empty_ref_nonempty_hyp_scores_0(self) -> None:
        assert _token_overlap_f1("", "hello world") == pytest.approx(0.0)

    def test_nonempty_ref_empty_hyp_scores_0(self) -> None:
        assert _token_overlap_f1("hello world", "") == pytest.approx(0.0)


class TestScoringCategories:
    """Token vs semantic category routing."""

    def test_coding_category_uses_token_overlap(self) -> None:
        """For 'coding' category, token overlap F1 is used."""
        ev = _make_evaluator()

        # Mock cosine similarity to detect if it's called
        ev._cosine_similarity = MagicMock(return_value=0.9)  # type: ignore

        score = ev.evaluate(
            question="Write hello world",
            response="print('hello world')",
            expected="print('hello world')",
            category="coding",
        )

        # For identical code, token overlap should be 1.0
        ev._cosine_similarity.assert_not_called()
        assert score == pytest.approx(1.0)

    def test_math_category_uses_token_overlap(self) -> None:
        ev = _make_evaluator()
        ev._cosine_similarity = MagicMock(return_value=0.9)  # type: ignore

        score = ev.evaluate(
            question="What is 2+2?",
            response="4",
            expected="4",
            category="math",
        )

        ev._cosine_similarity.assert_not_called()
        assert score == pytest.approx(1.0)

    def test_reasoning_category_uses_cosine_similarity(self) -> None:
        """For 'reasoning' category, cosine similarity is used."""
        ev = _make_evaluator()
        ev._cosine_similarity = MagicMock(return_value=0.85)  # type: ignore

        score = ev.evaluate(
            question="Why is the sky blue?",
            response="Rayleigh scattering",
            expected="Due to Rayleigh scattering of light",
            category="reasoning",
        )

        ev._cosine_similarity.assert_called_once()
        assert score == pytest.approx(0.85)


class TestJudgeModelScoring:
    """Judge-model scoring paths."""

    def test_judge_timeout_returns_none(self) -> None:
        """On judge timeout, score is None, execution continues."""
        import httpx

        ev = _make_evaluator(judge_model="llama3", judge_timeout=5)

        with patch("ollama_benchmark.quality_evaluator.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client

            score = ev.evaluate(
                question="Is the sky blue?",
                response="Yes, due to Rayleigh scattering.",
            )

        assert score is None

    def test_judge_success_extracts_score(self) -> None:
        """On judge success, score is extracted from JSON response."""
        ev = _make_evaluator(judge_model="llama3")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"score": 0.85}'}
        mock_resp.raise_for_status = MagicMock()

        with patch("ollama_benchmark.quality_evaluator.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            score = ev.evaluate(
                question="Is the sky blue?",
                response="Yes, due to Rayleigh scattering.",
            )

        assert score == pytest.approx(0.85)

    def test_judge_score_clamped_above_1(self) -> None:
        """Judge scores > 1.0 are clamped to 1.0."""
        ev = _make_evaluator(judge_model="llama3")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"score": 1.5}'}
        mock_resp.raise_for_status = MagicMock()

        with patch("ollama_benchmark.quality_evaluator.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            score = ev.evaluate(
                question="test",
                response="test",
            )

        assert score == pytest.approx(1.0)

    def test_judge_score_clamped_below_0(self) -> None:
        """Judge scores < 0.0 are clamped to 0.0."""
        ev = _make_evaluator(judge_model="llama3")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"score": -0.5}'}
        mock_resp.raise_for_status = MagicMock()

        with patch("ollama_benchmark.quality_evaluator.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            score = ev.evaluate(
                question="test",
                response="test",
            )

        assert score == pytest.approx(0.0)

    def test_malformed_judge_response_returns_none(self) -> None:
        """Unparseable judge response results in None score."""
        ev = _make_evaluator(judge_model="llama3")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "I cannot rate this."}
        mock_resp.raise_for_status = MagicMock()

        with patch("ollama_benchmark.quality_evaluator.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            score = ev.evaluate(
                question="test",
                response="test",
            )

        assert score is None

    def test_parse_judge_score_from_json(self) -> None:
        """_parse_judge_score handles clean JSON."""
        result = QualityEvaluator._parse_judge_score('{"score": 0.72}')
        assert result == pytest.approx(0.72)

    def test_parse_judge_score_from_embedded_json(self) -> None:
        """_parse_judge_score handles JSON embedded in text."""
        raw = 'Based on my analysis, here is the score: {"score": 0.65} I hope this helps.'
        result = QualityEvaluator._parse_judge_score(raw)
        assert result == pytest.approx(0.65)

    def test_parse_judge_score_returns_none_on_failure(self) -> None:
        """_parse_judge_score returns None on completely unparseable input."""
        result = QualityEvaluator._parse_judge_score("no score here at all")
        assert result is None


class TestPluginLoading:
    """Plugin loading (Req 14.2, 14.3, 14.5)."""

    def test_nonexistent_plugins_dir_loads_empty(self) -> None:
        """When plugins_dir doesn't exist, no plugins are loaded."""
        ev = QualityEvaluator(plugins_dir="/nonexistent/path/that/does/not/exist")
        assert ev._plugins == []

    def test_valid_plugin_is_loaded(self, tmp_path: Path) -> None:
        """A valid plugin with evaluate() callable is loaded successfully."""
        plugin_code = '''
def evaluate(question: str, response: str) -> float:
    return 0.5
'''
        (tmp_path / "my_plugin.py").write_text(plugin_code, encoding="utf-8")

        ev = QualityEvaluator(plugins_dir=str(tmp_path))
        assert len(ev._plugins) == 1

    def test_plugin_without_evaluate_is_skipped(self, tmp_path: Path) -> None:
        """Plugin without evaluate() callable is logged and skipped."""
        plugin_code = '''
class MyPlugin:
    pass
'''
        (tmp_path / "bad_plugin.py").write_text(plugin_code, encoding="utf-8")

        ev = QualityEvaluator(plugins_dir=str(tmp_path))
        assert len(ev._plugins) == 0

    def test_failing_plugin_import_is_skipped(self, tmp_path: Path) -> None:
        """Plugin that raises on import is skipped, not propagated."""
        plugin_code = '''
raise ImportError("this plugin fails to import")
'''
        (tmp_path / "failing_plugin.py").write_text(plugin_code, encoding="utf-8")

        # Should not raise
        ev = QualityEvaluator(plugins_dir=str(tmp_path))
        assert len(ev._plugins) == 0

    def test_plugin_score_blended_with_builtin(self, tmp_path: Path) -> None:
        """Plugin score is blended (average) with the built-in score."""
        plugin_code = '''
def evaluate(question: str, response: str) -> float:
    return 1.0
'''
        (tmp_path / "perfect_plugin.py").write_text(plugin_code, encoding="utf-8")

        ev = QualityEvaluator(plugins_dir=str(tmp_path))
        ev._cosine_similarity = MagicMock(return_value=0.0)  # type: ignore

        score = ev.evaluate(
            question="test",
            response="test",
            expected="test",
            category="reasoning",
        )

        # builtin=0.0, plugin=1.0 → blended=0.5
        assert score == pytest.approx(0.5)

    def test_failing_plugin_evaluate_is_skipped(self, tmp_path: Path) -> None:
        """When plugin.evaluate() raises at runtime, it is skipped (Req 14.5)."""
        plugin_code = '''
def evaluate(question: str, response: str) -> float:
    raise ValueError("runtime error in plugin")
'''
        (tmp_path / "runtime_fail.py").write_text(plugin_code, encoding="utf-8")

        ev = QualityEvaluator(plugins_dir=str(tmp_path))
        # Plugin was loaded but its evaluate() fails at call time
        result = ev._run_plugins("q", "r")
        assert result is None  # failed plugin contributes nothing


class TestGlobalScore:
    """compute_global_score() edge cases."""

    def test_empty_categories_returns_0(self) -> None:
        ev = _make_evaluator()
        assert ev.compute_global_score({}) == pytest.approx(0.0)

    def test_all_none_scores_returns_0(self) -> None:
        ev = _make_evaluator()
        assert ev.compute_global_score({"a": None, "b": None}) == pytest.approx(0.0)

    def test_uniform_scores_returns_same(self) -> None:
        ev = _make_evaluator()
        result = ev.compute_global_score({"a": 0.8, "b": 0.8, "c": 0.8})
        assert result == pytest.approx(0.8)

    def test_mixed_scores_average(self) -> None:
        ev = _make_evaluator()
        result = ev.compute_global_score({"a": 1.0, "b": 0.0})
        assert result == pytest.approx(0.5)

    def test_none_categories_excluded(self) -> None:
        ev = _make_evaluator()
        result = ev.compute_global_score({"a": 1.0, "b": None})
        assert result == pytest.approx(1.0)


class TestNoScoreWhenNoStrategy:
    """evaluate() returns None when no scoring strategy is applicable."""

    def test_no_expected_no_judge_no_plugins_returns_none(self) -> None:
        ev = _make_evaluator(judge_model=None, plugins_dir=None)
        score = ev.evaluate(question="test", response="test")
        assert score is None
