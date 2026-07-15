import os
from pathlib import Path

import pytest

from cluster_msa.errors import ExternalToolError
from cluster_msa.tools import get_tool_version, run_command


def script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_run_command_logs_command_and_output_without_mutating_parent_env(tmp_path, monkeypatch):
    tool = script(tmp_path / "env-tool", 'printf "%s\\n" "$CUDA_VISIBLE_DEVICES"')
    log = tmp_path / "run.log"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent")

    result = run_command(
        [str(tool)], stage="gpu-test", log_path=log, env={"CUDA_VISIBLE_DEVICES": "2"}
    )

    assert result.stdout == "2\n"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent"
    contents = log.read_text(encoding="utf-8")
    assert str(tool) in contents
    assert "2" in contents


def test_run_command_gpu_with_empty_devices_preserves_inherited_visibility(tmp_path, monkeypatch):
    tool = script(tmp_path / "env-tool", 'printf "%s" "$CUDA_VISIBLE_DEVICES"')
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent")

    result = run_command([str(tool)], stage="gpu-test", log_path=tmp_path / "run.log", env=None)

    assert result.stdout == "parent"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent"


def test_run_command_cpu_removes_inherited_gpu_environment(tmp_path, monkeypatch):
    tool = script(tmp_path / "env-tool", 'printf "%s" "${CUDA_VISIBLE_DEVICES-unset}"')
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent")

    result = run_command([str(tool)], stage="cpu-test", log_path=tmp_path / "run.log", env={})

    assert result.stdout == "unset"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent"


def test_run_command_raises_typed_error_and_logs_failure(tmp_path):
    tool = script(tmp_path / "fail", 'printf "bad output\\n"; exit 7')
    log = tmp_path / "failure.log"

    with pytest.raises(ExternalToolError) as error:
        run_command([str(tool)], stage="stage", log_path=log)

    message = str(error.value)
    assert "stage" in message
    assert tool.name in message
    assert "return code 7" in message
    assert str(log) in message
    assert "bad output" in log.read_text(encoding="utf-8")


def test_get_tool_version_returns_first_nonempty_output_line(tmp_path):
    tool = script(tmp_path / "versioned", 'printf "\\nversion 1.2\\nmore\\n"')

    assert get_tool_version(tool, tmp_path / "version.log") == "version 1.2"


def test_get_tool_version_reports_failure(tmp_path):
    tool = script(tmp_path / "broken", "exit 3")

    with pytest.raises(ExternalToolError, match="version"):
        get_tool_version(tool, tmp_path / "version.log")
