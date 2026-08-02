"""
Quality Evaluator for ollama-performance-studio.

Scores model responses using one of three strategies:

1. **Expected-answer scoring** (task 9.1):
   - Semantic categories: cosine similarity with ``sentence-transformers``
     model ``all-MiniLM-L6-v2``.
   - Code / math categories: token-overlap F1 score.
   - Returns float in [0.0, 1.0].

2. **Judge-model scoring** (task 9.2):
   - Posts question + response to a configured judge LLM via Ollama
     ``/api/generate``.
   - Expects JSON ``{"score": <float>}`` in [0.0, 1.0].
   - On judge timeout: records null score, continues.

3. **Plugin loading** (task 9.3):
   - Loads all ``.py`` files from ``config.plugins_dir`` at startup via
     ``importlib``.
   - Each plugin must expose a callable ``evaluate(question, response) -> float``.
   - Invalid / failing plugins are logged and skipped; built-in evaluators
     are used as fallback.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 14.2, 14.3, 14.5
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin protocol (Req 14.2, 14.3)
# ---------------------------------------------------------------------------


@runtime_checkable
class EvaluatorPlugin(Protocol):
    """Protocol that external evaluator plugins must implement (Req 14.2, 14.3)."""

    def evaluate(self, question: str, response: str) -> float:
        """Return a quality score in [0.0, 1.0]."""
        ...


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Lowercased word-level tokenization for token-overlap scoring."""
    return re.findall(r"\b\w+\b", text.lower())


def _token_overlap_f1(reference: str, hypothesis: str) -> float:
    """Token-level F1 between *reference* and *hypothesis* (Req 6.2 — code/math)."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        # If either side is empty, score is 0 unless both are empty (perfect match).
        return 1.0 if (not ref_tokens and not hyp_tokens) else 0.0

    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    common = ref_set & hyp_set

    if not common:
        return 0.0

    precision = len(common) / len(hyp_set)
    recall = len(common) / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall)
    return float(min(1.0, max(0.0, f1)))


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------


class QualityEvaluator:
    """Scores model responses using expected-answer, judge-model, or plugin strategy.

    Parameters
    ----------
    judge_model:
        Name of an Ollama model to use as judge (e.g. ``"llama3"``).
        When *None*, judge scoring is disabled.
    base_url:
        Base URL for the Ollama REST API.
    plugins_dir:
        Directory path to scan for ``.py`` evaluator plugin files at startup.
    judge_timeout:
        Seconds to wait for a judge response before recording a null score.
    """

    # ------------------------------------------------------------------
    # Category classification (Req 6.1, 6.2)
    # ------------------------------------------------------------------

    #: Categories evaluated via cosine similarity (sentence-transformers).
    SEMANTIC_CATEGORIES: frozenset[str] = frozenset(
        {
            "reasoning",
            "comprehension",
            "translation",
            "summarization",
            "technical_explanation",
            "instruction_following",
            "conversation",
            "context_usage",
            # Spanish aliases (Req 6.1 lists them by Spanish names in requirements)
            "razonamiento",
            "comprension_lectora",
            "traduccion",
            "resumenes",
            "explicacion_tecnica",
            "seguimiento_instrucciones",
            "conversacion",
            "uso_contexto",
        }
    )

    #: Categories evaluated via token-overlap F1 (code / math).
    TOKEN_CATEGORIES: frozenset[str] = frozenset(
        {
            "coding",
            "math",
            # Common aliases
            "code",
            "programacion",
            "mathematics",
            "matematicas",
        }
    )

    # Sentinel: embedder not yet loaded; deferred to first use.
    _EMBEDDER_NOT_LOADED = object()

    # Class-level cache so the model is loaded once per process, not once
    # per QualityEvaluator instance (avoids reloading between benchmark models).
    _ST_MODEL_CACHE = _EMBEDDER_NOT_LOADED

    def __init__(
        self,
        judge_model: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        plugins_dir: Optional[str] = None,
        judge_timeout: int = 120,
    ) -> None:
        self._judge_model = judge_model
        self._base_url = base_url.rstrip("/")
        self._judge_timeout = judge_timeout

        # Lazy-loaded sentence-transformer model (avoid import cost at module load).
        # Uses the class-level cache so the heavy model is only loaded once
        # across all evaluator instances in a session.
        self._st_model = self._EMBEDDER_NOT_LOADED

        # Task 9.3: Load plugins at startup.
        self._plugins: list = self._load_plugins(plugins_dir)

    # ------------------------------------------------------------------
    # Task 9.1: Expected-answer scoring — lazy embedder
    # ------------------------------------------------------------------

    def _get_st_model(self):
        """Lazily load the sentence-transformers model on first use (Req 6.2).

        Result is cached at class level so subsequent QualityEvaluator instances
        (one per benchmark model) reuse the already-loaded model without hitting
        HuggingFace again.
        """
        if QualityEvaluator._ST_MODEL_CACHE is self._EMBEDDER_NOT_LOADED:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                QualityEvaluator._ST_MODEL_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
                logger.debug("Loaded sentence-transformer model all-MiniLM-L6-v2")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load sentence-transformers: %s — semantic scoring disabled",
                    exc,
                )
                QualityEvaluator._ST_MODEL_CACHE = None
        # Mirror class cache to instance for convenience
        self._st_model = QualityEvaluator._ST_MODEL_CACHE
        return self._st_model

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity in [0.0, 1.0] using sentence-transformers.

        Falls back to token-overlap F1 when the model is unavailable.
        """
        model = self._get_st_model()
        if model is None:
            logger.debug(
                "sentence-transformers unavailable; falling back to token-overlap"
            )
            return _token_overlap_f1(text1, text2)

        try:
            # normalize_embeddings=True → dot product equals cosine similarity.
            embeddings = model.encode([text1, text2], normalize_embeddings=True)
            score = float(embeddings[0] @ embeddings[1])
            return min(1.0, max(0.0, score))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding computation failed: %s — returning 0.0", exc)
            return 0.0

    def _token_overlap_f1(self, response: str, expected: str) -> float:
        """Instance-method wrapper for token-overlap F1 (Req 6.2 — code/math)."""
        return _token_overlap_f1(expected, response)

    def _score_expected_answer(
        self, response: str, expected: str, category: str
    ) -> float:
        """Return a similarity score in [0.0, 1.0] against *expected*.

        Uses token-overlap F1 for TOKEN_CATEGORIES;
        cosine similarity via sentence-transformers for all others.

        Requirements: 6.1, 6.2
        """
        if category.lower() in self.TOKEN_CATEGORIES:
            return self._token_overlap_f1(response, expected)
        return self._cosine_similarity(response, expected)

    # ------------------------------------------------------------------
    # Task 9.2: Judge-model scoring
    # ------------------------------------------------------------------

    def _score_with_judge(self, question: str, response: str) -> Optional[float]:
        """Query the judge LLM and extract a float score in [0.0, 1.0].

        Returns *None* on timeout, HTTP error, or malformed judge output.

        Requirements: 2.5, 6.3, 6.5
        """
        if not self._judge_model:
            return None

        prompt = (
            "You are an impartial evaluator. Given the following question and response, "
            "rate the quality of the response on a scale from 0.0 to 1.0.\n"
            "Return ONLY valid JSON in the format: {\"score\": <float>}\n\n"
            f"Question: {question}\n\n"
            f"Response: {response}\n\n"
            "Your rating (JSON only):"
        )

        try:
            with httpx.Client(timeout=float(self._judge_timeout)) as client:
                resp = client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._judge_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                raw_text: str = resp.json().get("response", "")
        except httpx.TimeoutException:
            logger.warning(
                "Judge model %r timed out after %ds — recording null score",
                self._judge_model,
                self._judge_timeout,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Judge model request failed: %s — recording null score", exc)
            return None

        return self._parse_judge_score(raw_text)

    @staticmethod
    def _parse_judge_score(raw_text: str) -> Optional[float]:
        """Extract the numeric score from the judge's JSON response.

        Accepts ``{"score": 0.85}`` or any JSON object where "score" key is
        present, possibly surrounded by other text.  Returns *None* if parsing
        fails.
        """
        # Try direct JSON parse first.
        try:
            parsed = json.loads(raw_text.strip())
            score = float(parsed["score"])
            return min(1.0, max(0.0, score))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        # Find a JSON-like object containing "score".
        match = re.search(r'\{[^{}]*"score"\s*:\s*([0-9]*\.?[0-9]+)[^{}]*\}', raw_text)
        if match:
            try:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            except ValueError:
                pass

        logger.warning(
            "Could not parse judge score from response: %r — recording null score",
            raw_text[:200],
        )
        return None

    # ------------------------------------------------------------------
    # Task 9.3: Plugin loading
    # ------------------------------------------------------------------

    def _load_plugins(self, plugins_dir: Optional[str]) -> list:
        """Load evaluator plugins from *plugins_dir* at startup.

        Each ``.py`` file in *plugins_dir* is imported via ``importlib``.
        Valid plugins must expose a callable ``evaluate(question, response) -> float``.
        Invalid / failing plugins are logged and skipped.

        Returns a list of loaded plugin modules.

        Requirements: 14.2, 14.3, 14.5
        """
        plugins: list = []
        if not plugins_dir:
            return plugins

        import importlib.util
        import pathlib

        plugins_path = pathlib.Path(plugins_dir)
        if not plugins_path.exists():
            logger.warning(
                "plugins_dir %r does not exist — skipping plugin load", plugins_dir
            )
            return plugins

        for py_file in sorted(plugins_path.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                spec.loader.exec_module(module)  # type: ignore[union-attr]

                # Validate: module must expose a callable evaluate().
                if not callable(getattr(module, "evaluate", None)):
                    logger.warning(
                        "Plugin %r has no callable evaluate() — skipping", str(py_file)
                    )
                    continue

                plugins.append(module)
                logger.info("Loaded evaluator plugin: %r", str(py_file))

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to load plugin %r: %s — skipping", str(py_file), exc
                )

        return plugins

    def _run_plugins(self, question: str, response: str) -> Optional[float]:
        """Run all loaded plugins and return their average score, or *None*.

        Failed plugin calls are logged and skipped (Req 14.5).
        """
        if not self._plugins:
            return None

        scores: List[float] = []
        for plugin in self._plugins:
            try:
                score = plugin.evaluate(question, response)
                scores.append(min(1.0, max(0.0, float(score))))
            except Exception as exc:  # noqa: BLE001
                plugin_name = getattr(plugin, "__name__", repr(plugin))
                logger.error(
                    "Plugin %r raised during evaluate(): %s — skipping",
                    plugin_name,
                    exc,
                )

        if not scores:
            return None

        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Public evaluate interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        response: str,
        expected: Optional[str] = None,
        category: str = "general",
    ) -> Optional[float]:
        """Score a model *response* and return a float in [0.0, 1.0] or *None*.

        Scoring priority:
        1. If *expected* is provided → expected-answer scoring (9.1).
        2. Else if *judge_model* is configured → judge-model scoring (9.2).
        3. Plugin scores (9.3) are blended in when available.
        4. If no strategy applies → *None*.

        Requirements: 6.1, 6.2, 6.3, 6.5, 14.2, 14.3, 14.5
        """
        builtin_score: Optional[float] = None

        # --- Built-in scoring ---
        if expected is not None:
            builtin_score = self._score_expected_answer(response, expected, category)
        elif self._judge_model:
            builtin_score = self._score_with_judge(question, response)

        # --- Plugin scoring (task 9.3) ---
        plugin_score: Optional[float] = self._run_plugins(question, response)

        # --- Blend results ---
        if builtin_score is not None and plugin_score is not None:
            return min(1.0, max(0.0, (builtin_score + plugin_score) / 2.0))

        if builtin_score is not None:
            return builtin_score

        if plugin_score is not None:
            return plugin_score

        return None

    # ------------------------------------------------------------------
    # Task 9.4: Global score computation
    # ------------------------------------------------------------------

    def compute_global_score(
        self, category_scores: Dict[str, Optional[float]]
    ) -> float:
        """Weighted average of per-category scores (Req 6.4).

        Categories with a *None* score are excluded from the average.
        Returns ``0.0`` when no category produced a valid score.

        All categories are weighted equally (uniform weighting).
        """
        valid = [v for v in category_scores.values() if v is not None]
        if not valid:
            return 0.0
        return min(1.0, max(0.0, sum(valid) / len(valid)))
