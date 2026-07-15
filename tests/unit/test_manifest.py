import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cluster_msa.errors import OutputValidationError
from cluster_msa.manifest import write_manifest
from cluster_msa.models import RunConfig, RunResult, Toolchain


def make_config(tmp_path: Path, *, mode: str = "standard") -> RunConfig:
    return RunConfig(
        mode=mode,
        input_path=tmp_path / "inputs.csv",
        output_dir=tmp_path / "output",
        db_path=tmp_path / "database",
        toolchain=Toolchain(
            tmp_path / "bin" / "colabfold_search",
            tmp_path / "bin" / "mmseqs" if mode == "accelerated" else None,
        ),
        threads=8,
        gpu=True,
        gpus="0,2",
        af3_json=True,
        tmp_dir=tmp_path / "private-tmp",
        work_dir=tmp_path / "private-work",
        keep_work=True,
        overwrite=True,
        verbose=True,
        cluster_identity=0.75,
        cluster_coverage=0.85,
        cluster_mode=2,
    )


def test_write_manifest_has_stable_v1_schema_and_no_sensitive_content(tmp_path, monkeypatch):
    config = make_config(tmp_path, mode="accelerated")
    result = RunResult("accelerated", 3, 3, 1, 2, None)
    started = datetime(2026, 7, 15, 9, 30, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=12)
    monkeypatch.setenv("CLUSTER_MSA_TOKEN", "super-secret-token")
    monkeypatch.setenv("USER", "private-user")

    path = tmp_path / "run_manifest.json"
    write_manifest(
        path,
        config=config,
        result=result,
        tool_versions={"colabfold_search": "colabfold 1.5", "mmseqs": "MMseqs2 15"},
        started_at=started,
        finished_at=finished,
        stage_durations={"clustering": 1.25, "total": 12.0},
    )

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {
        "schema_version": 1,
        "package": {"name": "cluster-msa", "version": "0.1.0"},
        "status": "success",
        "mode": "accelerated",
        "input": {"path": str(config.input_path), "count": 3},
        "database": {
            "path": str(config.db_path),
            "resolved_path": str(config.db_path.resolve()),
        },
        "parameters": {
            "threads": 8,
            "gpu": True,
            "gpus": "0,2",
            "af3": True,
            "cluster_identity": 0.75,
            "cluster_coverage": 0.85,
            "cluster_mode": 2,
        },
        "tools": {
            "colabfold_search": {
                "path": str(config.toolchain.colabfold_search),
                "name": "colabfold_search",
                "version": "colabfold 1.5",
            },
            "mmseqs": {
                "path": str(config.toolchain.mmseqs),
                "name": "mmseqs",
                "version": "MMseqs2 15",
            },
        },
        "timing": {
            "started_at": "2026-07-15T09:30:00Z",
            "finished_at": "2026-07-15T09:30:12Z",
            "stage_durations_seconds": {"clustering": 1.25, "total": 12.0},
        },
        "result": {
            "expected_count": 3,
            "generated_count": 3,
            "representative_count": 1,
            "nonrepresentative_count": 2,
            "fallback_reason": None,
        },
    }
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n  \"package\"" in text
    for forbidden in (
        "ACDE",
        "super-secret-token",
        "private-user",
        str(config.tmp_dir),
        str(config.work_dir),
        str(config.output_dir),
    ):
        assert forbidden not in text


def test_standard_manifest_omits_accelerated_only_fields_and_mmseqs(tmp_path):
    config = make_config(tmp_path)
    now = datetime.now(timezone.utc)
    path = tmp_path / "manifest.json"

    write_manifest(
        path,
        config=config,
        result=RunResult("standard", 1, 1),
        tool_versions={"colabfold_search": "version"},
        started_at=now,
        finished_at=now,
        stage_durations={"total": 0.0},
    )

    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document["parameters"]) == {"threads", "gpu", "gpus", "af3"}
    assert set(document["tools"]) == {"colabfold_search"}
    assert document["result"] == {"expected_count": 1, "generated_count": 1}


@pytest.mark.parametrize(
    ("started", "finished", "durations", "message"),
    [
        (datetime(2026, 1, 1), datetime.now(timezone.utc), {"total": 1.0}, "timezone-aware"),
        (datetime.now(timezone.utc), datetime(2026, 1, 1), {"total": 1.0}, "timezone-aware"),
        (
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            {"total": 1.0},
            "before",
        ),
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            {"total": -0.1},
            "nonnegative",
        ),
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            {"total": float("nan")},
            "finite",
        ),
    ],
)
def test_write_manifest_rejects_invalid_timing(
    tmp_path, started, finished, durations, message
):
    with pytest.raises(OutputValidationError, match=message):
        write_manifest(
            tmp_path / "manifest.json",
            config=make_config(tmp_path),
            result=RunResult("standard", 1, 1),
            tool_versions={"colabfold_search": "version"},
            started_at=started,
            finished_at=finished,
            stage_durations=durations,
        )


def test_write_manifest_is_atomic_and_preserves_destination_on_replace_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "manifest.json"
    path.write_text("old\n", encoding="utf-8")
    now = datetime.now(timezone.utc)

    def fail_replace(source, destination):
        raise OSError("replace denied")

    monkeypatch.setattr("cluster_msa.manifest.os.replace", fail_replace)
    with pytest.raises(OutputValidationError, match="manifest"):
        write_manifest(
            path,
            config=make_config(tmp_path),
            result=RunResult("standard", 1, 1),
            tool_versions={"colabfold_search": "version"},
            started_at=now,
            finished_at=now,
            stage_durations={"total": 0.0},
        )

    assert path.read_text(encoding="utf-8") == "old\n"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["manifest.json"]


def test_write_manifest_is_stable_across_mapping_order(tmp_path):
    now = datetime.now(timezone.utc)
    arguments = dict(
        config=make_config(tmp_path),
        result=RunResult("standard", 1, 1),
        tool_versions={"colabfold_search": "version"},
        started_at=now,
        finished_at=now,
    )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_manifest(first, **arguments, stage_durations={"search": 1.0, "total": 2.0})
    write_manifest(second, **arguments, stage_durations={"total": 2.0, "search": 1.0})

    assert first.read_bytes() == second.read_bytes()
