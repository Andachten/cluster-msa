import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cluster_msa import __version__
from tests.smoke_helpers import (
    assert_a3m_query,
    configure_search_environment,
    failure_details,
    resolve_path,
    run_with_timeout,
)


pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not os.environ.get("CLUSTER_MSA_SMOKE_DB"),
        reason="set CLUSTER_MSA_SMOKE_DB to run real-tool smoke tests",
    ),
]


def test_standard_mode_with_real_tools(tmp_path: Path) -> None:
    parent_cwd = Path.cwd()
    sequences = {
        "protein_one": "ACDEFGHIKLMNPQRSTVWY",
        "protein_two": "MKTIIALSYIFCLVFADYKDDDDK",
    }
    timeout_value = os.environ.get("CLUSTER_MSA_SMOKE_TIMEOUT", "3600")
    try:
        timeout = float(timeout_value)
    except ValueError:
        pytest.fail(
            f"CLUSTER_MSA_SMOKE_TIMEOUT must be a positive finite number, got {timeout_value!r}"
        )
    if not math.isfinite(timeout) or timeout <= 0:
        pytest.fail(
            f"CLUSTER_MSA_SMOKE_TIMEOUT must be a positive finite number, got {timeout_value!r}"
        )

    input_path = tmp_path / "input.csv"
    input_path.write_text(
        "id,sequence\nprotein_one,ACDEFGHIKLMNPQRSTVWY\nprotein_two,MKTIIALSYIFCLVFADYKDDDDK\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    tmp_dir = tmp_path / "tmp"
    subprocess_cwd = tmp_path / "cwd"
    subprocess_cwd.mkdir()
    subprocess_cwd.chmod(0o555)
    command = [
        sys.executable,
        "-m",
        "cluster_msa",
        "standard",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--db-path",
        str(resolve_path(os.environ["CLUSTER_MSA_SMOKE_DB"], parent_cwd)),
        "--threads",
        "2",
        "--tmp-dir",
        str(tmp_dir),
    ]
    if os.environ.get("CLUSTER_MSA_SMOKE_CPU") == "1":
        command.append("--no-gpu")

    child_env = os.environ.copy()
    expected_search = configure_search_environment(child_env, parent_cwd)

    try:
        completed = run_with_timeout(
            command, cwd=subprocess_cwd, timeout=timeout, env=child_env
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else error.stdout or ""
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else error.stderr or ""
        )
        pytest.fail(
            f"real-tool smoke test timed out after {timeout:g} seconds\n"
            f"{failure_details(output_dir, tmp_dir, stdout, stderr)}",
            pytrace=False,
        )

    diagnostics = failure_details(output_dir, tmp_dir, completed.stdout, completed.stderr)
    assert completed.returncode == 0, diagnostics
    assert {path.name for path in output_dir.glob("*.a3m")} == {
        "protein_one.a3m",
        "protein_two.a3m",
    }, diagnostics
    for sequence_id, sequence in sequences.items():
        alignment = output_dir / f"{sequence_id}.a3m"
        assert alignment.is_file(), diagnostics
        assert alignment.stat().st_size > 0, diagnostics
        try:
            assert_a3m_query(alignment, sequence_id, sequence)
        except (OSError, UnicodeError, AssertionError) as error:
            pytest.fail(f"{error}\n{diagnostics}", pytrace=False)

    manifest_path = output_dir / "run_manifest.json"
    assert manifest_path.is_file(), diagnostics
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        pytest.fail(f"cannot read run manifest: {error}\n{diagnostics}", pytrace=False)
    assert manifest["status"] == "success", diagnostics
    assert manifest["mode"] == "standard", diagnostics
    assert manifest["schema_version"] == 1, diagnostics
    assert manifest["package"] == {"name": "cluster-msa", "version": __version__}, diagnostics
    assert manifest["input"] == {"path": str(input_path), "count": 2}, diagnostics
    expected_db = str(resolve_path(os.environ["CLUSTER_MSA_SMOKE_DB"], parent_cwd))
    assert manifest["database"] == {
        "path": expected_db,
        "resolved_path": expected_db,
    }, diagnostics
    assert manifest["parameters"] == {
        "threads": 2,
        "gpu": os.environ.get("CLUSTER_MSA_SMOKE_CPU") != "1",
        "gpus": "",
        "af3": False,
    }, diagnostics
    search_tool = manifest["tools"]["colabfold_search"]
    assert search_tool["path"] == expected_search, diagnostics
    assert search_tool["name"] == Path(expected_search).name, diagnostics
    assert isinstance(search_tool["version"], str) and search_tool["version"].strip(), diagnostics
    timing = manifest["timing"]
    assert timing["timing_scope"] == "through_pre_manifest_finalization", diagnostics
    assert isinstance(timing["started_at"], str) and timing["started_at"], diagnostics
    assert isinstance(timing["finished_at"], str) and timing["finished_at"], diagnostics
    durations = timing["stage_durations_seconds"]
    assert set(durations) == {"full_database_search", "output_validation", "total"}, diagnostics
    assert all(
        not isinstance(duration, bool)
        and isinstance(duration, (int, float))
        and math.isfinite(duration)
        and duration >= 0
        for duration in durations.values()
    ), diagnostics
    assert manifest["result"]["expected_count"] == 2, diagnostics
    assert manifest["result"]["generated_count"] == 2, diagnostics
