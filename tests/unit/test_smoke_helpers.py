import os
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

import pytest

from tests import smoke_helpers
from tests.smoke import test_real_tools
from tests.smoke_helpers import (
    assert_a3m_query,
    failure_details,
    open_process_handle,
    pidfd_supported,
    process_handle_is_running,
    resolve_executable,
    resolve_path,
    run_with_timeout,
)


@pytest.mark.skipif(
    sys.platform != "linux" or not pidfd_supported(), reason="pidfd is unavailable"
)
def test_process_handle_tracks_exact_process_identity(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"], cwd=tmp_path)
    handle = open_process_handle(process.pid)
    try:
        process.wait(timeout=3)

        assert not process_handle_is_running(handle)
    finally:
        os.close(handle)


def test_failure_details_reads_run_log_from_explicit_tmp_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "published"
    tmp_dir = tmp_path / "tmp"
    run_log = tmp_dir / "cluster-msa-work" / "standard-one" / "output-one" / "run.log"
    run_log.parent.mkdir(parents=True)
    run_log.write_text("real colabfold failure\n", encoding="utf-8")

    details = failure_details(output_dir, tmp_dir, "cli stdout", "cli stderr")

    assert "cli stdout" in details
    assert "cli stderr" in details
    assert "real colabfold failure" in details
    assert str(run_log) in details


def test_relative_paths_are_resolved_from_parent_cwd(tmp_path: Path) -> None:
    parent_cwd = tmp_path / "parent"
    parent_cwd.mkdir()

    assert resolve_path("relative/database", parent_cwd) == (
        parent_cwd / "relative/database"
    ).resolve()
    assert resolve_executable("tools/colabfold_search", parent_cwd) == str(
        (parent_cwd / "tools/colabfold_search").resolve()
    )


def test_bare_executable_is_resolved_from_supplied_path(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "colabfold_search"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    assert resolve_executable(
        "colabfold_search", tmp_path, path=str(bin_dir)
    ) == str(executable)


def test_missing_bare_executable_is_retained_for_config_diagnostics(tmp_path: Path) -> None:
    assert resolve_executable("missing_search", tmp_path, path="") == "missing_search"


def test_assert_a3m_query_accepts_wrapped_query_sequence(tmp_path: Path) -> None:
    alignment = tmp_path / "query.a3m"
    alignment.write_text(">query description\nACDE\nFGHI\n>hit\nAC-EFGHI\n", encoding="utf-8")

    assert_a3m_query(alignment, "query", "ACDEFGHI")


def test_assert_a3m_query_rejects_wrong_query_identifier(tmp_path: Path) -> None:
    alignment = tmp_path / "query.a3m"
    alignment.write_text(">other\nACDEFGHI\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="identifier"):
        assert_a3m_query(alignment, "query", "ACDEFGHI")


def test_assert_a3m_query_rejects_wrong_query_sequence(tmp_path: Path) -> None:
    alignment = tmp_path / "query.a3m"
    alignment.write_text(">query\nACDEYGHI\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="query sequence"):
        assert_a3m_query(alignment, "query", "ACDEFGHI")


@pytest.mark.parametrize(
    "content",
    [
        "",
        "# comment only\n",
        "ACDE\n",
        ">\nACDE\n",
        ">query\n",
        ">query\n\n>hit\nACDE\n",
    ],
)
def test_assert_a3m_query_rejects_missing_or_empty_first_record(
    tmp_path: Path, content: str
) -> None:
    alignment = tmp_path / "query.a3m"
    alignment.write_text(content, encoding="utf-8")

    with pytest.raises(AssertionError, match="first A3M record|query sequence"):
        assert_a3m_query(alignment, "query", "ACDE")


@pytest.mark.parametrize("query_sequence", ["ACdeFGHI", "AC-DEFGHI"])
def test_assert_a3m_query_rejects_insertions_and_gaps_in_query(
    tmp_path: Path, query_sequence: str
) -> None:
    alignment = tmp_path / "query.a3m"
    alignment.write_text(f">query\n{query_sequence}\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="query sequence"):
        assert_a3m_query(alignment, "query", "ACDEFGHI")


def test_search_environment_resolves_default_from_child_path(tmp_path: Path) -> None:
    bin_dir = tmp_path / "relative-bin"
    bin_dir.mkdir()
    executable = bin_dir / "colabfold_search"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    child_env = {"PATH": str(bin_dir)}

    resolved = smoke_helpers.configure_search_environment(child_env, tmp_path)

    assert resolved == str(executable.resolve())
    assert child_env["COLABFOLD_SEARCH"] == resolved


def test_real_tool_smoke_defers_missing_bare_search_tool_to_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SubprocessReached(Exception):
        pass

    def subprocess_reached(*args: object, **kwargs: object) -> NoReturn:
        raise SubprocessReached

    monkeypatch.setenv("CLUSTER_MSA_SMOKE_DB", "missing-db")
    monkeypatch.setenv("COLABFOLD_SEARCH", "missing_search")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(test_real_tools, "run_with_timeout", subprocess_reached)

    with pytest.raises(SubprocessReached):
        test_real_tools.test_standard_mode_with_real_tools(tmp_path)


class ControlledBaseException(BaseException):
    pass


@pytest.mark.skipif(
    sys.platform != "linux" or not pidfd_supported(), reason="pidfd is unavailable"
)
@pytest.mark.parametrize(
    "interruption",
    [KeyboardInterrupt(), SystemExit(7), ControlledBaseException("stop")],
)
def test_base_exception_kills_and_reaps_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: BaseException
) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import pathlib, subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    original_communicate = subprocess.Popen.communicate
    observed_process: subprocess.Popen[str] | None = None
    parent_handle: int | None = None
    child_handle: int | None = None
    interrupted = False

    def communicate_once_then_delegate(
        process: subprocess.Popen[str], *args: object, **kwargs: object
    ) -> tuple[str, str]:
        nonlocal observed_process, parent_handle, child_handle, interrupted
        observed_process = process
        if not interrupted:
            interrupted = True
            deadline = time.monotonic() + 3
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert child_pid_path.exists(), "parent did not publish child PID"
            parent_handle = open_process_handle(process.pid)
            child_handle = open_process_handle(
                int(child_pid_path.read_text(encoding="utf-8"))
            )
            raise interruption
        return original_communicate(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", communicate_once_then_delegate)

    try:
        with pytest.raises(type(interruption)) as caught:
            run_with_timeout(
                [sys.executable, "-c", parent_code],
                cwd=tmp_path,
                timeout=5,
            )

        assert parent_handle is not None
        assert child_handle is not None
        deadline = time.monotonic() + 3
        while process_handle_is_running(child_handle) and time.monotonic() < deadline:
            time.sleep(0.05)

        assert caught.value is interruption
        assert observed_process is not None
        assert observed_process.returncode == -9
        assert not process_handle_is_running(parent_handle)
        assert not process_handle_is_running(child_handle)
    finally:
        if parent_handle is not None:
            os.close(parent_handle)
        if child_handle is not None:
            os.close(child_handle)


def test_base_exception_cleanup_handles_process_exit_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_communicate = subprocess.Popen.communicate
    interruption = ControlledBaseException("after exit")
    interrupted = False

    def reap_then_interrupt(
        process: subprocess.Popen[str], *args: object, **kwargs: object
    ) -> NoReturn | tuple[str, str]:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            original_communicate(process, *args, **kwargs)
            raise interruption
        return original_communicate(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", reap_then_interrupt)

    with pytest.raises(ControlledBaseException) as caught:
        run_with_timeout([sys.executable, "-c", "pass"], cwd=tmp_path, timeout=5)

    assert caught.value is interruption


@pytest.mark.skipif(
    sys.platform != "linux" or not pidfd_supported(), reason="pidfd is unavailable"
)
def test_timeout_kills_process_group_and_preserves_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import time; print('child output', flush=True); time.sleep(60)"
    )
    parent_code = (
        "import pathlib, subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "print('parent output', flush=True); time.sleep(60)"
    )
    original_communicate = subprocess.Popen.communicate
    parent_handle: int | None = None
    child_handle: int | None = None

    def capture_handles_then_delegate(
        process: subprocess.Popen[str], *args: object, **kwargs: object
    ) -> tuple[str, str]:
        nonlocal parent_handle, child_handle
        if parent_handle is None:
            deadline = time.monotonic() + 3
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert child_pid_path.exists(), "parent did not publish child PID"
            parent_handle = open_process_handle(process.pid)
            child_handle = open_process_handle(
                int(child_pid_path.read_text(encoding="utf-8"))
            )
        return original_communicate(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", capture_handles_then_delegate)

    try:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            run_with_timeout([sys.executable, "-c", parent_code], cwd=tmp_path, timeout=0.5)

        assert parent_handle is not None
        assert child_handle is not None
        deadline = time.monotonic() + 3
        while process_handle_is_running(child_handle) and time.monotonic() < deadline:
            time.sleep(0.05)

        assert not process_handle_is_running(parent_handle)
        assert not process_handle_is_running(child_handle)
        assert "parent output" in (caught.value.stdout or "")
        assert "child output" in (caught.value.stdout or "")
    finally:
        if parent_handle is not None:
            os.close(parent_handle)
        if child_handle is not None:
            os.close(child_handle)


def test_failure_details_survives_log_disappearing(tmp_path: Path) -> None:
    output_dir = tmp_path / "published"
    tmp_dir = tmp_path / "tmp"
    details = failure_details(output_dir, tmp_dir, "stdout", "stderr")

    assert "stdout" in details
    assert "stderr" in details
    assert "not available" in details
