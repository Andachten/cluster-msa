import subprocess
import sys

import pytest

from cluster_msa.cli import main
from cluster_msa.errors import (
    ConfigurationError,
    ExternalToolError,
    InputValidationError,
    OutputValidationError,
)


def test_cli_subprocess_argparse_usage_returns_2():
    result = subprocess.run(
        [sys.executable, "-m", "cluster_msa"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (InputValidationError("bad input"), 3),
        (ConfigurationError("bad config"), 3),
        (ExternalToolError("search stage failed; log: /tmp/run.log"), 4),
        (OutputValidationError("missing output"), 5),
    ],
)
def test_cli_expected_error_codes_are_concise(monkeypatch, capsys, error, expected):
    monkeypatch.setattr("cluster_msa.cli.build_run_config", lambda *args: (_ for _ in ()).throw(error))

    code = main(["standard", "--input", "in.csv", "--output-dir", "out"])

    assert code == expected
    stderr = capsys.readouterr().err
    assert str(error) in stderr
    assert "Traceback" not in stderr


def test_cli_keyboard_interrupt_returns_130(monkeypatch, capsys):
    monkeypatch.setattr(
        "cluster_msa.cli.build_run_config",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert main(["standard", "--input", "in.csv", "--output-dir", "out"]) == 130
    assert "interrupted" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("verbose", [False, True])
def test_cli_unexpected_exception_traceback_only_when_verbose(monkeypatch, capsys, verbose):
    monkeypatch.setattr(
        "cluster_msa.cli.build_run_config",
        lambda *args: (_ for _ in ()).throw(RuntimeError("private implementation detail")),
    )
    arguments = ["standard", "--input", "in.csv", "--output-dir", "out"]
    if verbose:
        arguments.append("--verbose")

    assert main(arguments) == 1

    stderr = capsys.readouterr().err
    assert ("Traceback" in stderr) is verbose
    assert ("private implementation detail" in stderr) is verbose
    if not verbose:
        assert stderr.strip() == "cluster-msa: unexpected error"


def test_cli_verbose_expected_error_includes_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        "cluster_msa.cli.build_run_config",
        lambda *args: (_ for _ in ()).throw(ConfigurationError("bad config")),
    )
    assert main(
        ["standard", "--input", "in.csv", "--output-dir", "out", "--verbose"]
    ) == 3
    assert "Traceback" in capsys.readouterr().err
