"""
Model lifecycle manager for ollama-performance-studio.

Wraps all `ollama` CLI subprocess calls. All commands are launched
automatically as subprocesses — no user confirmation, no interactive
prompts, no `input()`, no `pause()`, no `confirm()` calls anywhere.

Components
----------
ModelManager   — pull, start_and_verify, stop_and_remove
ColdStartTimeoutError — raised by start_and_verify on timeout
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Optional

import httpx

from ollama_benchmark.models import PullResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: ping the model via /api/generate
# ---------------------------------------------------------------------------


def _ping_model(base_url: str, model: str) -> bool:
    """Return True if *model* responds to a minimal prompt.

    Uses a short per-request timeout (10 s) so the polling loop stays
    responsive.  Any exception (connection refused, malformed JSON, etc.)
    is silently swallowed — the caller will retry.
    """
    try:
        resp = httpx.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": "ping", "stream": False},
            timeout=10.0,
        )
        data = resp.json()
        return bool(data.get("response", "").strip())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ColdStartTimeoutError(RuntimeError):
    """Raised when the model does not respond within the cold-start timeout."""


# ---------------------------------------------------------------------------
# Helper: query model size after a successful pull
# ---------------------------------------------------------------------------


def _get_model_size_gb(model: str) -> Optional[float]:
    """Return the on-disk size of *model* in GB, or None if it cannot be determined.

    Runs ``ollama show <model>`` and looks for a line containing the word
    "size" (case-insensitive).  The value is extracted and converted to GB.

    The subprocess is given a fixed 30-second timeout; if it fails for any
    reason the function returns None without raising.
    """
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        output = result.stdout + result.stderr
        for line in output.splitlines():
            lower = line.lower()
            if "size" in lower:
                # Lines look like:
                #   Size            4.7 GB
                #   Model size: 4.7 GB
                #   size     4661751424
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() in ("size", "size:"):
                        # The numeric value should be the next token
                        if i + 1 < len(parts):
                            try:
                                value = float(parts[i + 1].rstrip(","))
                                unit = parts[i + 2].upper() if i + 2 < len(parts) else "B"
                                if unit.startswith("GB"):
                                    return value
                                elif unit.startswith("MB"):
                                    return value / 1024.0
                                elif unit.startswith("KB"):
                                    return value / (1024.0 ** 2)
                                else:
                                    # Raw bytes
                                    return value / (1024.0 ** 3)
                            except (ValueError, IndexError):
                                continue
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not determine model size for %r: %s", model, exc)
    return None


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class ModelManager:
    """Wraps all ``ollama`` CLI subprocess calls.

    All methods launch subprocesses automatically without any user interaction.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url
        #: The running ``ollama run`` subprocess, set by ``start_and_verify``
        #: and consumed by ``stop_and_remove`` (task 7.3).
        self._active_process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Task 7.1 — pull
    # ------------------------------------------------------------------

    def pull(self, model: str, timeout: int = 3600) -> PullResult:
        """Run ``ollama pull <model>`` as an automatic subprocess.

        Behaviour
        ---------
        - Streams stdout/stderr to the console in real time so the user can
          observe download progress without any prompting.
        - Records ``download_time_s`` from subprocess start to completion.
        - After a successful pull, queries ``ollama show`` for the on-disk
          size and populates ``model_size_gb`` (``None`` when unavailable).
        - Returns ``PullResult(success=True, ...)`` on zero exit code.
        - Returns ``PullResult(success=False, ...)`` on non-zero exit or
          ``subprocess.TimeoutExpired``; the failing model is **not** re-tried
          here — the caller (BenchmarkRunner) decides whether to skip it.

        Parameters
        ----------
        model:
            The Ollama model tag to pull, e.g. ``"llama3:8b"``.
        timeout:
            Maximum seconds to wait for the pull to complete.
            Defaults to 3600 s when not configured.

        Returns
        -------
        PullResult
            Populated DTO describing the outcome of the pull operation.
        """
        logger.info("Pulling model %r (timeout=%ds) …", model, timeout)
        logger.debug("[CMD] ollama pull %s", model)

        output_lines: list[str] = []
        t_start = time.perf_counter()

        try:
            process = subprocess.Popen(
                ["ollama", "pull", model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",  # Replace problematic characters instead of failing
                bufsize=1,  # line-buffered
            )

            # Stream output to console in real time
            assert process.stdout is not None  # always set when PIPE is used
            for line in process.stdout:
                print(line, end="", flush=True)
                output_lines.append(line)

            process.wait(timeout=timeout)
            download_time_s = time.perf_counter() - t_start

            if process.returncode != 0:
                last_line = output_lines[-1].strip() if output_lines else "unknown error"
                logger.error(
                    "ollama pull %r failed (exit %d): %s",
                    model,
                    process.returncode,
                    last_line,
                )
                return PullResult(
                    model_name=model,
                    success=False,
                    download_time_s=download_time_s,
                    model_size_gb=None,
                    error=last_line,
                )

        except subprocess.TimeoutExpired:
            download_time_s = time.perf_counter() - t_start
            logger.error(
                "ollama pull %r timed out after %ds", model, timeout
            )
            try:
                process.kill()
                process.wait()
            except Exception:  # noqa: BLE001
                pass
            return PullResult(
                model_name=model,
                success=False,
                download_time_s=download_time_s,
                model_size_gb=None,
                error="timeout",
            )

        except Exception as exc:  # noqa: BLE001
            download_time_s = time.perf_counter() - t_start
            logger.error("ollama pull %r raised an unexpected error: %s", model, exc)
            return PullResult(
                model_name=model,
                success=False,
                download_time_s=download_time_s,
                model_size_gb=None,
                error=str(exc),
            )

        # Pull succeeded — try to get disk size
        model_size_gb = _get_model_size_gb(model)
        logger.info(
            "Pull of %r completed in %.1fs (size=%.2f GB)",
            model,
            download_time_s,
            model_size_gb if model_size_gb is not None else 0.0,
        )
        return PullResult(
            model_name=model,
            success=True,
            download_time_s=download_time_s,
            model_size_gb=model_size_gb,
            error=None,
        )

    # ------------------------------------------------------------------
    # Task 7.2 — start_and_verify
    # ------------------------------------------------------------------

    def start_and_verify(self, model: str, timeout: int = 120) -> float:
        """Launch ``ollama run <model>`` and verify readiness via a ping prompt.

        Behaviour
        ---------
        1. Spawns ``ollama run <model>`` as a background subprocess with
           stdout/stderr piped.
        2. A daemon thread streams that output to the console in real time
           so the user can observe startup progress without being prompted.
        3. The main thread polls ``/api/generate`` every 0.5 s with a fixed
           "ping" prompt until a non-empty text response is received or
           *timeout* seconds elapse.
        4. On success the process is left running (the model stays loaded in
           memory) and ``cold_start_s`` is returned.
        5. On timeout: logs the event, kills the ``ollama run`` process, runs
           ``ollama rm <model>`` automatically (no prompts), then raises
           ``ColdStartTimeoutError``.

        The subprocess handle is stored in ``self._active_process`` so that
        ``stop_and_remove`` (task 7.3) can terminate it later.

        Parameters
        ----------
        model:
            Ollama model tag to start, e.g. ``"llama3:8b"``.
        timeout:
            Maximum seconds to wait for the model to respond (cold-start
            timeout).  Defaults to 120 s.

        Returns
        -------
        float
            Cold-start time in seconds from process launch to first response.

        Raises
        ------
        ColdStartTimeoutError
            When the model does not respond within *timeout* seconds.
        """
        logger.info("Starting model %r (cold-start timeout=%ds) …", model, timeout)
        logger.debug("[CMD] ollama run %s", model)

        # --- 1. Launch ``ollama run <model>`` ---
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # line-buffered
        )
        self._active_process = process

        # --- 2. Background thread: stream subprocess output to console ---
        def _stream_output(proc: subprocess.Popen) -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
            except ValueError:
                # stdout was closed before the loop finished — harmless
                pass

        stream_thread = threading.Thread(
            target=_stream_output,
            args=(process,),
            daemon=True,
            name=f"ollama-run-stream-{model}",
        )
        stream_thread.start()

        # --- 3. Poll /api/generate until ready or timeout ---
        t_start = time.perf_counter()
        poll_interval = 0.5  # seconds between ping attempts

        while True:
            elapsed = time.perf_counter() - t_start

            if _ping_model(self._base_url, model):
                cold_start_s = time.perf_counter() - t_start
                logger.info(
                    "Model %r responded after %.1fs (cold-start).",
                    model,
                    cold_start_s,
                )
                return cold_start_s

            if elapsed >= timeout:
                break

            time.sleep(poll_interval)

        # --- 4. Timeout path ---
        elapsed_final = time.perf_counter() - t_start
        logger.error(
            "Model %r did not respond within %ds cold-start timeout (%.1fs elapsed). "
            "Killing process and running ollama rm.",
            model,
            timeout,
            elapsed_final,
        )

        # Kill the ollama run process
        try:
            process.kill()
            process.wait(timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error killing ollama run process for %r: %s", model, exc)
        finally:
            self._active_process = None

        # Run ``ollama rm <model>`` automatically — no prompts
        try:
            logger.debug("[CMD] ollama rm %s", model)
            rm_result = subprocess.run(
                ["ollama", "rm", model],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if rm_result.returncode != 0:
                logger.warning(
                    "ollama rm %r exited with code %d: %s",
                    model,
                    rm_result.returncode,
                    (rm_result.stdout + rm_result.stderr).strip(),
                )
            else:
                logger.info("ollama rm %r completed successfully.", model)
        except Exception as exc:  # noqa: BLE001
            logger.error("ollama rm %r raised an error: %s", model, exc)

        raise ColdStartTimeoutError(
            f"Model {model!r} did not respond within {timeout}s cold-start timeout"
        )

    # ------------------------------------------------------------------
    # Task 7.3 — stop_and_remove
    # ------------------------------------------------------------------

    def stop_and_remove(self, model: str) -> None:
        """Terminate the model process and run ``ollama rm <model>``.

        Behaviour
        ---------
        1. If ``self._active_process`` is not None, terminates it:
           - Attempts ``process.terminate()`` first.
           - Waits up to 5 seconds for graceful shutdown.
           - If the process is still alive, calls ``process.kill()``.
           - Sets ``self._active_process = None`` afterward.
        2. Runs ``ollama rm <model>`` as a subprocess automatically without
           user confirmation.
        3. Streams the command output to console in real time for visibility.
        4. If ``ollama rm`` exits with a non-zero code, logs the error with
           full details but does **not** raise — execution continues.
        5. If any exception occurs during ``ollama rm``, logs the full error
           and continues without raising.

        Parameters
        ----------
        model:
            Ollama model tag to remove, e.g. ``"llama3:8b"``.
        """
        logger.info("Stopping and removing model %r …", model)

        # --- 1. Terminate self._active_process if it exists ---
        if self._active_process is not None:
            proc = self._active_process
            try:
                logger.debug("Terminating ollama run process for %r …", model)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    logger.debug("Process for %r exited gracefully.", model)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Process for %r did not exit after 5s; killing it.", model
                    )
                    proc.kill()
                    proc.wait()  # no timeout; kill is forceful
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Error terminating ollama run process for %r: %s", model, exc
                )
            finally:
                self._active_process = None

        # --- 2. Run ``ollama rm <model>`` automatically without prompts ---
        try:
            logger.info("Running ollama rm %r …", model)
            logger.debug("[CMD] ollama rm %s", model)
            rm_proc = subprocess.Popen(
                ["ollama", "rm", model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            # --- 3. Print output to console for visibility ---
            assert rm_proc.stdout is not None
            for line in rm_proc.stdout:
                print(line, end="", flush=True)

            rm_proc.wait(timeout=60)

            # --- 4. Log error if non-zero exit, but do NOT raise ---
            if rm_proc.returncode != 0:
                logger.error(
                    "ollama rm %r exited with code %d. "
                    "The model may not have been removed. Continuing anyway.",
                    model,
                    rm_proc.returncode,
                )
            else:
                logger.info("ollama rm %r completed successfully.", model)

        # --- 5. Catch any exception during ``ollama rm``, log, and continue ---
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ollama rm %r raised an error: %s. Continuing anyway.", model, exc
            )
