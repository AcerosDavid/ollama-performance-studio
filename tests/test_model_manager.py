"""
Unit tests for ModelManager.

Task 7.4: Validates Requirements 3.1, 3.6, 3.7, 3.9
Tests verify that:
  - pull() success/failure paths work correctly
  - start_and_verify() handles cold-start timeout
  - stop_and_remove() handles ollama rm failure
  - NO input(), pause, or confirmation prompts exist anywhere
"""
from __future__ import annotations

import inspect
import subprocess
from unittest.mock import MagicMock, patch, call

import pytest

from ollama_benchmark.model_manager import (
    ColdStartTimeoutError,
    ModelManager,
    _ping_model,
)
from ollama_benchmark.models import PullResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc(returncode: int = 0, stdout_lines: list[str] | None = None) -> MagicMock:
    """Build a mock subprocess.Popen instance."""
    proc = MagicMock()
    proc.returncode = returncode
    lines = stdout_lines or []
    proc.stdout = iter(lines)
    proc.wait = MagicMock(return_value=returncode)
    proc.kill = MagicMock()
    proc.terminate = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Task 7.1 — pull()
# Validates: Requirements 3.1, 3.2, 3.6
# ---------------------------------------------------------------------------

class TestPullSuccess:
    """pull() succeeds: returns PullResult(success=True) with timing info."""

    def test_returns_pull_result_success(self) -> None:
        proc = _make_proc(returncode=0, stdout_lines=["pulling…\n", "done\n"])

        with (
            patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=proc),
            patch("ollama_benchmark.model_manager._get_model_size_gb", return_value=4.1),
        ):
            mm = ModelManager()
            result = mm.pull("llama3", timeout=30)

        assert result.success is True
        assert result.model_name == "llama3"
        assert result.download_time_s >= 0.0
        assert result.model_size_gb == pytest.approx(4.1)
        assert result.error is None

    def test_streams_stdout_to_console(self, capsys) -> None:
        """pull() must print each stdout line in real time."""
        proc = _make_proc(returncode=0, stdout_lines=["progress 50%\n", "done\n"])

        with (
            patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=proc),
            patch("ollama_benchmark.model_manager._get_model_size_gb", return_value=None),
        ):
            mm = ModelManager()
            mm.pull("llama3", timeout=30)

        captured = capsys.readouterr()
        assert "progress 50%" in captured.out or "done" in captured.out

    def test_no_user_prompts_in_pull(self) -> None:
        """pull() source code must NOT call input(), pause, or confirm()."""
        import ollama_benchmark.model_manager as mod
        source = inspect.getsource(mod.ModelManager.pull)
        assert "input(" not in source
        assert "pause(" not in source
        assert "confirm(" not in source


class TestPullFailure:
    """pull() failure: non-zero exit returns PullResult(success=False)."""

    def test_nonzero_exit_returns_failure(self) -> None:
        proc = _make_proc(returncode=1, stdout_lines=["unknown model\n"])

        with patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=proc):
            mm = ModelManager()
            result = mm.pull("bad-model:latest", timeout=30)

        assert result.success is False
        assert result.error is not None

    def test_timeout_returns_failure(self) -> None:
        proc = _make_proc(returncode=0, stdout_lines=[])
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="ollama pull", timeout=5)
        proc.kill = MagicMock()
        proc.wait = MagicMock(side_effect=[subprocess.TimeoutExpired(cmd="ollama pull", timeout=5), None])

        with patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=proc):
            mm = ModelManager()
            result = mm.pull("big-model", timeout=5)

        assert result.success is False
        assert result.error == "timeout"

    def test_exception_during_popen_returns_failure(self) -> None:
        with patch(
            "ollama_benchmark.model_manager.subprocess.Popen",
            side_effect=FileNotFoundError("ollama not found"),
        ):
            mm = ModelManager()
            result = mm.pull("llama3", timeout=30)

        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# Task 7.2 — start_and_verify()
# Validates: Requirements 3.3, 3.4, 3.7
# ---------------------------------------------------------------------------

class TestStartAndVerifySuccess:
    """start_and_verify() pings model and returns cold_start_s on success."""

    def test_returns_positive_cold_start_s(self) -> None:
        proc = _make_proc(returncode=0, stdout_lines=["loading model\n"])

        with (
            patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=proc),
            patch("ollama_benchmark.model_manager._ping_model", return_value=True),
            patch("ollama_benchmark.model_manager.time.sleep"),
        ):
            mm = ModelManager()
            cold_start = mm.start_and_verify("llama3", timeout=30)

        assert cold_start >= 0.0

    def test_sets_active_process(self) -> None:
        proc = _make_proc(returncode=0, stdout_lines=[])

        with (
            patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=proc),
            patch("ollama_benchmark.model_manager._ping_model", return_value=True),
            patch("ollama_benchmark.model_manager.time.sleep"),
        ):
            mm = ModelManager()
            mm.start_and_verify("llama3", timeout=30)

        assert mm._active_process is proc

    def test_no_user_prompts_in_start_and_verify(self) -> None:
        """start_and_verify() source code must NOT call input(), pause, or confirm()."""
        import ollama_benchmark.model_manager as mod
        source = inspect.getsource(mod.ModelManager.start_and_verify)
        assert "input(" not in source
        assert "pause(" not in source
        assert "confirm(" not in source


class TestStartAndVerifyTimeout:
    """start_and_verify() times out → kills proc, runs ollama rm, raises ColdStartTimeoutError."""

    def test_raises_cold_start_timeout_error(self) -> None:
        proc = _make_proc(returncode=0, stdout_lines=[])

        with (
            patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=proc),
            patch("ollama_benchmark.model_manager._ping_model", return_value=False),
            patch("ollama_benchmark.model_manager.time.sleep"),
            patch("ollama_benchmark.model_manager.time.perf_counter", side_effect=[0.0, 999.0, 999.0, 999.0]),
            patch("ollama_benchmark.model_manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            mm = ModelManager()
            with pytest.raises(ColdStartTimeoutError):
                mm.start_and_verify("slow-model", timeout=5)

    def test_kills_process_on_timeout(self) -> None:
        proc = _make_proc(returncode=0, stdout_lines=[])

        with (
            patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=proc),
            patch("ollama_benchmark.model_manager._ping_model", return_value=False),
            patch("ollama_benchmark.model_manager.time.sleep"),
            patch("ollama_benchmark.model_manager.time.perf_counter", side_effect=[0.0, 999.0, 999.0, 999.0]),
            patch("ollama_benchmark.model_manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            mm = ModelManager()
            with pytest.raises(ColdStartTimeoutError):
                mm.start_and_verify("slow-model", timeout=5)

        proc.kill.assert_called()

    def test_runs_ollama_rm_on_timeout(self) -> None:
        """On cold-start timeout, ollama rm is run automatically without any prompt."""
        proc = _make_proc(returncode=0, stdout_lines=[])

        with (
            patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=proc),
            patch("ollama_benchmark.model_manager._ping_model", return_value=False),
            patch("ollama_benchmark.model_manager.time.sleep"),
            patch("ollama_benchmark.model_manager.time.perf_counter", side_effect=[0.0, 999.0, 999.0, 999.0]),
            patch("ollama_benchmark.model_manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            mm = ModelManager()
            with pytest.raises(ColdStartTimeoutError):
                mm.start_and_verify("slow-model", timeout=5)

        # Verify ollama rm was called
        assert mock_run.called
        rm_call_args = [str(a) for call_args in mock_run.call_args_list for a in call_args[0]]
        assert any("ollama" in arg or "rm" in arg for arg in rm_call_args), \
            "Expected ollama rm to be called on cold-start timeout"


# ---------------------------------------------------------------------------
# Task 7.3 — stop_and_remove()
# Validates: Requirements 3.5, 3.9
# ---------------------------------------------------------------------------

class TestStopAndRemove:
    """stop_and_remove() terminates process and runs ollama rm."""

    def test_terminates_active_process(self) -> None:
        active_proc = _make_proc(returncode=0, stdout_lines=[])
        active_proc.wait = MagicMock(return_value=0)

        rm_proc = _make_proc(returncode=0, stdout_lines=["deleted\n"])

        mm = ModelManager()
        mm._active_process = active_proc

        with patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=rm_proc):
            mm.stop_and_remove("llama3")

        active_proc.terminate.assert_called_once()
        assert mm._active_process is None

    def test_runs_ollama_rm(self) -> None:
        rm_proc = _make_proc(returncode=0, stdout_lines=["deleted successfully\n"])

        mm = ModelManager()

        with patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=rm_proc) as mock_popen:
            mm.stop_and_remove("llama3")

        # Verify Popen was called with ollama rm
        mock_popen.assert_called_once()
        cmd_args = mock_popen.call_args[0][0]
        assert "ollama" in cmd_args[0]
        assert "rm" in cmd_args[1]
        assert "llama3" in cmd_args[2]

    def test_ollama_rm_failure_does_not_raise(self) -> None:
        """When ollama rm exits with non-zero code, stop_and_remove must NOT raise."""
        rm_proc = _make_proc(returncode=1, stdout_lines=["error removing\n"])

        mm = ModelManager()

        with patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=rm_proc):
            # Must not raise
            mm.stop_and_remove("llama3")

    def test_ollama_rm_exception_does_not_raise(self) -> None:
        """When ollama rm raises an exception, stop_and_remove must NOT propagate it."""
        mm = ModelManager()

        with patch(
            "ollama_benchmark.model_manager.subprocess.Popen",
            side_effect=FileNotFoundError("ollama not found"),
        ):
            # Must not raise
            mm.stop_and_remove("llama3")

    def test_streams_ollama_rm_output_to_console(self, capsys) -> None:
        """ollama rm output is printed to console for visibility."""
        rm_proc = _make_proc(returncode=0, stdout_lines=["Model llama3 removed\n"])

        mm = ModelManager()

        with patch("ollama_benchmark.model_manager.subprocess.Popen", return_value=rm_proc):
            mm.stop_and_remove("llama3")

        captured = capsys.readouterr()
        assert "removed" in captured.out or "llama3" in captured.out

    def test_no_user_prompts_in_stop_and_remove(self) -> None:
        """stop_and_remove() source code must NOT call input(), pause, or confirm()."""
        import ollama_benchmark.model_manager as mod
        source = inspect.getsource(mod.ModelManager.stop_and_remove)
        assert "input(" not in source
        assert "pause(" not in source
        assert "confirm(" not in source


# ---------------------------------------------------------------------------
# No interactive prompts anywhere in model_manager module
# Validates: Requirement 3.1 (fully automatic)
# ---------------------------------------------------------------------------

class TestNoInteractivePrompts:
    """The entire model_manager module must not contain interactive prompt calls."""

    def _get_code_lines(self, source: str) -> str:
        """Strip docstrings and comments, keep only executable code."""
        import ast
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        
        # Collect line numbers that are string constants (docstrings)
        docstring_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    # This is a docstring — mark its lines
                    start = node.lineno
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start
                    docstring_lines.update(range(start, end + 1))
        
        lines = source.splitlines()
        code_lines = []
        for i, line in enumerate(lines, start=1):
            if i in docstring_lines:
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            code_lines.append(line)
        return "\n".join(code_lines)

    def test_no_input_calls_in_module(self) -> None:
        import ollama_benchmark.model_manager as mod
        source = inspect.getsource(mod)
        code_text = self._get_code_lines(source)
        assert "input(" not in code_text, "input() found in model_manager module code"

    def test_no_pause_calls_in_module(self) -> None:
        import ollama_benchmark.model_manager as mod
        source = inspect.getsource(mod)
        code_text = self._get_code_lines(source)
        assert "pause()" not in code_text, "pause() found in model_manager module code"

    def test_no_confirm_calls_in_module(self) -> None:
        import ollama_benchmark.model_manager as mod
        source = inspect.getsource(mod)
        code_text = self._get_code_lines(source)
        assert "confirm()" not in code_text, "confirm() found in model_manager module code"


# ---------------------------------------------------------------------------
# Default timeout fallback
# Validates: Requirement 3.6 (default 3600s timeout)
# ---------------------------------------------------------------------------

class TestDefaultTimeouts:
    """Default timeout for pull is 3600s when not configured."""

    def test_pull_default_timeout_is_3600(self) -> None:
        """pull() signature must default to timeout=3600."""
        import inspect as _inspect
        sig = _inspect.signature(ModelManager.pull)
        assert sig.parameters["timeout"].default == 3600

    def test_start_and_verify_default_timeout_is_120(self) -> None:
        """start_and_verify() signature must default to timeout=120."""
        import inspect as _inspect
        sig = _inspect.signature(ModelManager.start_and_verify)
        assert sig.parameters["timeout"].default == 120
