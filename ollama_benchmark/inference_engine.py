"""
Inference Engine for ollama-performance-studio.

Streams prompts to a running Ollama model via its REST API and collects
fine-grained latency and throughput metrics.

Metrics collected per prompt
-----------------------------
- TTFT (time to first token): wall-clock ms from request send to first
  streaming chunk with a non-empty ``response`` field.
- total_response_ms: wall-clock ms from request send to stream completion.
- tokens_generated: ``eval_count`` from the final Ollama response object.
- tokens_per_second: ``eval_count / (eval_duration / 1e9)``
  (eval_duration is reported in nanoseconds by Ollama).
- avg_inter_token_ms: ``eval_duration / eval_count / 1e6``
  (nanoseconds → milliseconds per token).

Timeout handling
----------------
On ``httpx.TimeoutException`` the method returns an ``InferenceResult``
with ``timed_out=True`` and all metric fields set to ``None``.  If a
``Database`` instance is wired in, the timeout record is persisted via
``db.save_inference_result``.

Usage
-----
    engine = InferenceEngine(base_url="http://localhost:11434", db=db)
    result = asyncio.run(
        engine.run_prompt("llama3", "What is 2+2?", timeout=30)
    )
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import httpx

from ollama_benchmark.config import PromptEntry
from ollama_benchmark.models import InferenceResult

__all__ = ["InferenceEngine"]

_log = logging.getLogger(__name__)


class InferenceEngine:
    """Async inference client for the Ollama REST API.

    Parameters
    ----------
    base_url:
        Root URL of the Ollama server (no trailing slash).
        Defaults to ``"http://localhost:11434"``.
    db:
        Optional ``Database`` instance.  When provided, each
        ``InferenceResult`` (including timeouts and errors) is persisted
        automatically via ``db.save_inference_result``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        db=None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._db = db
        self._last_avg_latency_ms: float = 0.0
        # Optional callback(model, prompt_idx, total, category) called after each prompt
        self.on_prompt_complete = None

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def run_prompt(
        self,
        model: str,
        prompt: str,
        timeout: int,
        category: str = "unknown",
        model_run_id: Optional[int] = None,
    ) -> InferenceResult:
        """Stream ``/api/generate`` and return an ``InferenceResult``.

        Parameters
        ----------
        model:
            Ollama model name (e.g. ``"llama3"``).
        prompt:
            The prompt text to send.
        timeout:
            Request timeout in seconds.  Applied as the ``httpx`` read
            timeout so that stalled streams are detected promptly.
        category:
            Prompt category label used when persisting to the DB.
        model_run_id:
            Optional model run ID used when persisting to the DB.  When
            ``None`` and ``self._db`` is set, the result is still persisted
            but without a ``model_run_id`` link (the caller should supply
            this when available).

        Returns
        -------
        InferenceResult
            Contains all measured metrics.  On timeout, ``timed_out=True``
            and all metric fields are ``None``.  On other errors,
            ``error`` is set to the exception message.
        """
        # Log prompt being sent (truncated if too long)
        prompt_display = prompt[:200] + "..." if len(prompt) > 200 else prompt
        _log.debug("[PROMPT] %s: %s", category, prompt_display)

        url = f"{self._base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }

        result = await self._stream_generate(
            url=url,
            payload=payload,
            prompt=prompt,
            timeout=timeout,
        )

        # Log response received (truncated if too long)
        if result.response_text:
            response_display = result.response_text[:200] + "..." if len(result.response_text) > 200 else result.response_text
            _log.debug("[RESPONSE] %s", response_display)
        elif result.timed_out:
            _log.debug("[RESPONSE] <timeout>")
        elif result.error:
            _log.debug("[RESPONSE] <error: %s>", result.error)

        # Persist to DB if wired in
        if self._db is not None and model_run_id is not None:
            try:
                self._db.save_inference_result(model_run_id, result, category)
            except Exception as exc:  # noqa: BLE001 — DB errors must not abort inference
                _log.error("Failed to persist InferenceResult: %s", exc)

        return result

    async def run_all_prompts(
        self,
        model: str,
        prompts: list[tuple[str, "PromptEntry"]],
        timeout: int,
        model_run_id: Optional[int] = None,
    ) -> list[InferenceResult]:
        """Run all prompts sequentially in config-defined order.

        Parameters
        ----------
        model:
            Ollama model name (e.g. ``"llama3"``).
        prompts:
            Ordered list of ``(category, PromptEntry)`` pairs as defined in
            the benchmark config.  Prompts are executed in the exact order
            given — no reordering.
        timeout:
            Per-prompt request timeout in seconds.
        model_run_id:
            Optional model run ID forwarded to ``run_prompt`` for DB
            persistence of each individual result.

        Returns
        -------
        list[InferenceResult]
            One result per prompt, in the same order as the input list.
            After this call, ``self.last_avg_latency_ms`` holds the mean
            of non-None ``total_response_ms`` values, or 0 when all timed
            out / errored.
        """
        results: list[InferenceResult] = []

        for idx, (category, prompt_entry) in enumerate(prompts, start=1):
            result = await self.run_prompt(
                model,
                prompt_entry.text,
                timeout,
                category=category,
                model_run_id=model_run_id,
            )
            results.append(result)
            if self.on_prompt_complete:
                self.on_prompt_complete(model, idx, len(prompts), category)

        # Compute avg_latency_ms from non-None total_response_ms values
        # (Requirements 5.5 and 5.6)
        non_none_latencies = [
            r.total_response_ms
            for r in results
            if r.total_response_ms is not None
        ]
        if non_none_latencies:
            self._last_avg_latency_ms = sum(non_none_latencies) / len(non_none_latencies)
        else:
            self._last_avg_latency_ms = 0.0

        return results

    @property
    def last_avg_latency_ms(self) -> float:
        """Average latency from the most recent run_all_prompts call."""
        return self._last_avg_latency_ms

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _stream_generate(
        self,
        url: str,
        payload: dict,
        prompt: str,
        timeout: int,
    ) -> InferenceResult:
        """Core streaming logic.

        Opens an ``httpx.AsyncClient``, POSTs the payload, and iterates
        the response line-by-line.  Each line is a JSON object emitted by
        Ollama.  The final object (``done == True``) carries ``eval_count``
        and ``eval_duration`` used to compute throughput metrics.

        The entire streaming operation is wrapped in ``asyncio.wait_for`` with
        the caller-supplied *timeout* as a hard wall-clock deadline.  This
        guarantees that slow but continuously-streaming models (which would
        otherwise never trigger the httpx read timeout) are still cut off
        after *timeout* seconds total.

        Catches ``asyncio.TimeoutError``, ``httpx.TimeoutException`` and
        generic exceptions, returning a gracefully-degraded ``InferenceResult``
        in all cases.
        """
        import asyncio

        t_start = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self._do_stream(url, payload, prompt, t_start),
                timeout=float(timeout),
            )
            return result
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - t_start) * 1_000
            _log.warning(
                "Inference hard timeout (%ds) exceeded for prompt (model=%s) — skipping",
                timeout,
                payload.get("model"),
            )
            return InferenceResult(
                prompt_text=prompt,
                response_text=None,
                ttft_ms=None,
                total_response_ms=elapsed_ms,
                tokens_generated=None,
                tokens_per_second=None,
                avg_inter_token_ms=None,
                timed_out=True,
                error=f"Hard timeout after {timeout}s",
            )
        except httpx.TimeoutException as exc:
            _log.warning("Inference timeout for prompt (model=%s): %s", payload.get("model"), exc)
            return InferenceResult(
                prompt_text=prompt,
                response_text=None,
                ttft_ms=None,
                total_response_ms=None,
                tokens_generated=None,
                tokens_per_second=None,
                avg_inter_token_ms=None,
                timed_out=True,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("Inference error for prompt (model=%s): %s", payload.get("model"), exc)
            return InferenceResult(
                prompt_text=prompt,
                response_text=None,
                ttft_ms=None,
                total_response_ms=None,
                tokens_generated=None,
                tokens_per_second=None,
                avg_inter_token_ms=None,
                timed_out=False,
                error=str(exc),
            )

    async def _do_stream(
        self,
        url: str,
        payload: dict,
        prompt: str,
        t_start: float,
    ) -> InferenceResult:
        """Inner streaming coroutine — called inside asyncio.wait_for."""
        response_chunks: list[str] = []
        ttft_ms: Optional[float] = None
        tokens_generated: Optional[int] = None
        tokens_per_second: Optional[float] = None
        avg_inter_token_ms: Optional[float] = None

        # Connect / write timeouts remain short; the total wall-clock deadline
        # is enforced by the asyncio.wait_for in _stream_generate.
        http_timeout = httpx.Timeout(
            connect=10.0,
            read=30.0,
            write=10.0,
            pool=5.0,
        )

        try:
            async with httpx.AsyncClient(timeout=http_timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError as exc:
                            _log.warning("Skipping non-JSON line from Ollama stream: %s (%s)", line[:80], exc)
                            continue

                        chunk_response: str = chunk.get("response", "")
                        is_done: bool = chunk.get("done", False)

                        # Track TTFT on first non-empty response chunk
                        if ttft_ms is None and chunk_response:
                            ttft_ms = (time.perf_counter() - t_start) * 1_000

                        if chunk_response:
                            response_chunks.append(chunk_response)

                        # Final chunk carries throughput metadata
                        if is_done:
                            eval_count: Optional[int] = chunk.get("eval_count")
                            eval_duration: Optional[int] = chunk.get("eval_duration")

                            if eval_count is not None and eval_duration is not None and eval_count > 0:
                                tokens_generated = eval_count
                                tokens_per_second = eval_count / (eval_duration / 1e9)
                                avg_inter_token_ms = eval_duration / eval_count / 1e6
                            elif eval_count is not None:
                                tokens_generated = eval_count

                            break  # stream is complete

            total_response_ms = (time.perf_counter() - t_start) * 1_000
            response_text = "".join(response_chunks) if response_chunks else None

            return InferenceResult(
                prompt_text=prompt,
                response_text=response_text,
                ttft_ms=ttft_ms,
                total_response_ms=total_response_ms,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                avg_inter_token_ms=avg_inter_token_ms,
                timed_out=False,
                error=None,
            )

        except httpx.TimeoutException as exc:
            _log.warning("Inference timeout for prompt (model=%s): %s", payload.get("model"), exc)
            return InferenceResult(
                prompt_text=prompt,
                response_text=None,
                ttft_ms=None,
                total_response_ms=None,
                tokens_generated=None,
                tokens_per_second=None,
                avg_inter_token_ms=None,
                timed_out=True,
                error=str(exc),
            )

        except Exception as exc:  # noqa: BLE001
            _log.error("Inference error for prompt (model=%s): %s", payload.get("model"), exc)
            return InferenceResult(
                prompt_text=prompt,
                response_text=None,
                ttft_ms=None,
                total_response_ms=None,
                tokens_generated=None,
                tokens_per_second=None,
                avg_inter_token_ms=None,
                timed_out=False,
                error=str(exc),
            )
