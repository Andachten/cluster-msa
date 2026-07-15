import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not os.environ.get("CLUSTER_MSA_SMOKE_DB"),
        reason="set CLUSTER_MSA_SMOKE_DB to run real-tool smoke tests",
    ),
]


def test_standard_mode_with_real_tools(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text(
        "id,sequence\n"
        "protein_one,ACDEFGHIKLMNPQRSTVWY\n"
        "protein_two,MKTIIALSYIFCLVFADYKDDDDK\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
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
        os.environ["CLUSTER_MSA_SMOKE_DB"],
        "--threads",
        "2",
    ]
    if os.environ.get("CLUSTER_MSA_SMOKE_CPU") == "1":
        command.append("--no-gpu")

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    alignments = sorted(output_dir.glob("*.a3m"))
    assert len(alignments) == 2
    assert all(path.stat().st_size > 0 for path in alignments)

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["mode"] == "standard"
    assert manifest["result"]["expected_count"] == 2
    assert manifest["result"]["generated_count"] == 2
