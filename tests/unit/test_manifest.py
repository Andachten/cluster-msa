import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cluster_msa.errors import OutputValidationError
from cluster_msa.manifest import mark_manifest_failed, write_manifest
from cluster_msa.models import RunConfig, RunResult, Toolchain


STANDARD_TIMING = {
    "full_database_search": 1.0,
    "output_validation": 0.5,
    "total": 2.0,
}
ACCELERATED_TIMING = {
    "clustering": 1.0,
    "representative_search": 1.0,
    "compact_database": 1.0,
    "nonrepresentative_search": 1.0,
    "merge_and_staging": 1.0,
    "output_validation": 1.0,
    "total": 7.0,
}
FALLBACK_TIMING = {
    "clustering": 1.0,
    "standard_search": 1.0,
    "output_validation": 1.0,
    "total": 4.0,
}


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
        stage_durations={**ACCELERATED_TIMING, "total": 12.0},
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
            "timing_scope": "through_pre_manifest_finalization",
            "started_at": "2026-07-15T09:30:00Z",
            "finished_at": "2026-07-15T09:30:12Z",
            "stage_durations_seconds": {**ACCELERATED_TIMING, "total": 12.0},
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
        stage_durations=STANDARD_TIMING,
    )

    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document["parameters"]) == {"threads", "gpu", "gpus", "af3"}
    assert set(document["tools"]) == {"colabfold_search"}
    assert document["result"] == {"expected_count": 1, "generated_count": 1}


def test_manifest_prefers_exact_supplied_database_spelling(tmp_path):
    config = replace(make_config(tmp_path), db_path_supplied="~/database")
    now = datetime.now(timezone.utc)
    path = tmp_path / "manifest.json"

    write_manifest(
        path,
        config=config,
        result=RunResult("standard", 1, 1),
        tool_versions={"colabfold_search": "version"},
        started_at=now,
        finished_at=now,
        stage_durations=STANDARD_TIMING,
    )

    assert json.loads(path.read_text(encoding="utf-8"))["database"] == {
        "path": "~/database",
        "resolved_path": str(config.db_path.resolve()),
    }


@pytest.mark.parametrize(
    ("started", "finished", "durations", "message"),
    [
        (datetime(2026, 1, 1), datetime.now(timezone.utc), STANDARD_TIMING, "timezone-aware"),
        (datetime.now(timezone.utc), datetime(2026, 1, 1), STANDARD_TIMING, "timezone-aware"),
        (
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            STANDARD_TIMING,
            "before",
        ),
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            {**STANDARD_TIMING, "total": -0.1},
            "nonnegative",
        ),
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            {**STANDARD_TIMING, "total": float("nan")},
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
            stage_durations=STANDARD_TIMING,
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
    write_manifest(first, **arguments, stage_durations=STANDARD_TIMING)
    write_manifest(
        second,
        **arguments,
        stage_durations={"total": 2.0, "output_validation": 0.5, "full_database_search": 1.0},
    )

    assert first.read_bytes() == second.read_bytes()


def test_mark_manifest_failed_atomically_updates_only_diagnostic_status(tmp_path):
    now = datetime.now(timezone.utc)
    path = tmp_path / "manifest.json"
    write_manifest(
        path,
        config=make_config(tmp_path),
        result=RunResult("standard", 1, 1),
        tool_versions={"colabfold_search": "version"},
        started_at=now,
        finished_at=now,
        stage_durations=STANDARD_TIMING,
    )
    original = json.loads(path.read_text(encoding="utf-8"))

    mark_manifest_failed(path, "publication", OutputValidationError("token=secret-value"))

    failed = json.loads(path.read_text(encoding="utf-8"))
    assert failed == {
        **original,
        "status": "failed",
        "failure_stage": "publication",
        "error": "output publication failed",
    }
    assert "secret-value" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("config_changes", "result", "versions", "durations"),
    [
        ({"mode": "invalid"}, RunResult("invalid", 1, 1), {"colabfold_search": "v"}, {"total": 0.0}),
        ({}, RunResult("accelerated", 1, 1, 1, 0), {"colabfold_search": "v"}, {"total": 0.0}),
        ({}, RunResult("standard", -1, 0), {"colabfold_search": "v"}, {"total": 0.0}),
        ({}, RunResult("standard", True, 1), {"colabfold_search": "v"}, {"total": 0.0}),
        ({}, RunResult("standard", 1, 2), {"colabfold_search": "v"}, {"total": 0.0}),
        ({}, RunResult("standard", 1, 1, 0, None), {"colabfold_search": "v"}, {"total": 0.0}),
        ({}, RunResult("standard", 1, 1, None, None, "fallback"), {"colabfold_search": "v"}, {"total": 0.0}),
        ({}, RunResult("standard", 1, 1), {}, {"total": 0.0}),
        ({}, RunResult("standard", 1, 1), {"colabfold_search": ""}, {"total": 0.0}),
        ({}, RunResult("standard", 1, 1), {"colabfold_search": 7}, {"total": 0.0}),
        ({}, RunResult("standard", 1, 1), {"colabfold_search": "v"}, {"": 0.0}),
        ({}, RunResult("standard", 1, 1), {"colabfold_search": "v"}, {7: 0.0}),
        ({}, RunResult("standard", 1, 1), {"colabfold_search": "v"}, {"total": True}),
        ({}, RunResult("standard", 1, 1), {"colabfold_search": "v"}, {"total": "0"}),
        ({}, RunResult("standard", 1, 1), {"colabfold_search": "v"}, {"total": float("inf")}),
    ],
)
def test_write_manifest_rejects_malformed_standard_inputs(
    tmp_path, config_changes, result, versions, durations
):
    config = replace(make_config(tmp_path), **config_changes)
    now = datetime.now(timezone.utc)

    with pytest.raises(OutputValidationError):
        write_manifest(
            tmp_path / "manifest.json",
            config=config,
            result=result,
            tool_versions=versions,
            started_at=now,
            finished_at=now,
            stage_durations=durations,
        )


@pytest.mark.parametrize(
    ("config", "result", "versions"),
    [
        ("missing_mmseqs", RunResult("accelerated", 2, 2, 1, 1), {"colabfold_search": "v", "mmseqs": "m"}),
        ("valid", RunResult("accelerated", 2, 2, None, 1), {"colabfold_search": "v", "mmseqs": "m"}),
        ("valid", RunResult("accelerated", 2, 2, 1, -1), {"colabfold_search": "v", "mmseqs": "m"}),
        ("valid", RunResult("accelerated", 2, 2, 1, True), {"colabfold_search": "v", "mmseqs": "m"}),
        ("valid", RunResult("accelerated", 2, 2, 1, 1, 7), {"colabfold_search": "v", "mmseqs": "m"}),
        ("valid", RunResult("accelerated", 2, 2, 1, 1), {"colabfold_search": "v"}),
        ("valid", RunResult("accelerated", 2, 2, 1, 1), {"colabfold_search": "v", "mmseqs": ""}),
    ],
)
def test_write_manifest_rejects_malformed_accelerated_inputs(
    tmp_path, config, result, versions
):
    run_config = make_config(tmp_path, mode="accelerated")
    if config == "missing_mmseqs":
        run_config = replace(run_config, toolchain=replace(run_config.toolchain, mmseqs=None))
    now = datetime.now(timezone.utc)

    with pytest.raises(OutputValidationError):
        write_manifest(
            tmp_path / "manifest.json",
            config=run_config,
            result=result,
            tool_versions=versions,
            started_at=now,
            finished_at=now,
            stage_durations=ACCELERATED_TIMING,
        )


def test_write_manifest_rejects_empty_executable_name(tmp_path):
    config = make_config(tmp_path)
    config = replace(config, toolchain=replace(config.toolchain, colabfold_search=Path(".")))
    now = datetime.now(timezone.utc)

    with pytest.raises(OutputValidationError):
        write_manifest(
            tmp_path / "manifest.json",
            config=config,
            result=RunResult("standard", 1, 1),
            tool_versions={"colabfold_search": "version"},
            started_at=now,
            finished_at=now,
            stage_durations=STANDARD_TIMING,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"threads": 0},
        {"threads": True},
        {"gpu": 1},
        {"af3_json": 0},
        {"gpus": None},
        {"input_path": "input.csv"},
        {"output_dir": "output"},
        {"db_path": "database"},
        {"tmp_dir": "tmp"},
        {"work_dir": "work"},
        {"db_path_supplied": ""},
        {"db_path_supplied": 7},
    ],
)
def test_write_manifest_rejects_invalid_emitted_standard_config(tmp_path, changes):
    config = replace(make_config(tmp_path), **changes)
    now = datetime.now(timezone.utc)

    with pytest.raises(OutputValidationError):
        write_manifest(
            tmp_path / "manifest.json",
            config=config,
            result=RunResult("standard", 1, 1),
            tool_versions={"colabfold_search": "version"},
            started_at=now,
            finished_at=now,
            stage_durations=STANDARD_TIMING,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"cluster_identity": 0},
        {"cluster_identity": 1.1},
        {"cluster_identity": True},
        {"cluster_identity": float("nan")},
        {"cluster_coverage": 0},
        {"cluster_coverage": float("inf")},
        {"cluster_coverage": "0.8"},
        {"cluster_mode": -1},
        {"cluster_mode": True},
    ],
)
def test_write_manifest_rejects_invalid_emitted_accelerated_config(tmp_path, changes):
    config = replace(make_config(tmp_path, mode="accelerated"), **changes)
    now = datetime.now(timezone.utc)

    with pytest.raises(OutputValidationError):
        write_manifest(
            tmp_path / "manifest.json",
            config=config,
            result=RunResult("accelerated", 2, 2, 1, 1),
            tool_versions={"colabfold_search": "version", "mmseqs": "version"},
            started_at=now,
            finished_at=now,
            stage_durations=ACCELERATED_TIMING,
        )


@pytest.mark.parametrize(
    "result",
    [
        RunResult("standard", 0, 0),
        RunResult("standard", 2, 1),
        RunResult("accelerated", 3, 3, 1, 1),
        RunResult("accelerated", 2, 2, 2, 0, "unexpected"),
        RunResult("accelerated", 2, 2, 1, 1, "no_non_representatives"),
        RunResult("accelerated", 2, 2, 1, 0, "no_non_representatives"),
        RunResult("accelerated", 2, 2, 2, 0),
        RunResult("accelerated", 2, 2, 0, 2),
    ],
)
def test_write_manifest_rejects_inconsistent_success_counts(tmp_path, result):
    config = make_config(tmp_path, mode=result.mode)
    now = datetime.now(timezone.utc)
    versions = {"colabfold_search": "version"}
    if result.mode == "accelerated":
        versions["mmseqs"] = "version"

    with pytest.raises(OutputValidationError):
        write_manifest(
            tmp_path / "manifest.json",
            config=config,
            result=result,
            tool_versions=versions,
            started_at=now,
            finished_at=now,
            stage_durations=(ACCELERATED_TIMING if result.mode == "accelerated" else STANDARD_TIMING),
        )


def test_write_manifest_accepts_consistent_accelerated_fallback(tmp_path):
    now = datetime.now(timezone.utc)
    write_manifest(
        tmp_path / "manifest.json",
        config=make_config(tmp_path, mode="accelerated"),
        result=RunResult("accelerated", 2, 2, 2, 0, "no_non_representatives"),
        tool_versions={"colabfold_search": "version", "mmseqs": "version"},
        started_at=now,
        finished_at=now,
        stage_durations=FALLBACK_TIMING,
    )


def test_write_manifest_wraps_nonfinite_json_serialization(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("cluster_msa.manifest.__version__", float("nan"))

    with pytest.raises(OutputValidationError):
        write_manifest(
            tmp_path / "manifest.json",
            config=make_config(tmp_path),
            result=RunResult("standard", 1, 1),
            tool_versions={"colabfold_search": "version"},
            started_at=now,
            finished_at=now,
            stage_durations=STANDARD_TIMING,
        )


@pytest.mark.parametrize(
    ("mode", "result", "durations"),
    [
        ("standard", RunResult("standard", 1, 1), {}),
        ("standard", RunResult("standard", 1, 1), {"total": 1.0}),
        (
            "standard",
            RunResult("standard", 1, 1),
            {**STANDARD_TIMING, "unexpected": 0.0},
        ),
        (
            "standard",
            RunResult("standard", 1, 1),
            {**STANDARD_TIMING, "full_database_search": 3.0},
        ),
        (
            "standard",
            RunResult("standard", 1, 1),
            {"full_database_search": 1.1, "output_validation": 1.0, "total": 2.0},
        ),
        (
            "accelerated",
            RunResult("accelerated", 2, 2, 1, 1),
            {**ACCELERATED_TIMING, "merge_and_staging": 3.0, "total": 7.0},
        ),
        (
            "accelerated",
            RunResult("accelerated", 2, 2, 2, 0, "no_non_representatives"),
            ACCELERATED_TIMING,
        ),
        (
            "accelerated",
            RunResult("accelerated", 2, 2, 2, 0, "no_non_representatives"),
            {"clustering": 2.0, "standard_search": 2.0, "output_validation": 2.0, "total": 5.9},
        ),
    ],
)
def test_write_manifest_rejects_invalid_stage_schema(tmp_path, mode, result, durations):
    now = datetime.now(timezone.utc)
    versions = {"colabfold_search": "version"}
    if mode == "accelerated":
        versions["mmseqs"] = "version"

    with pytest.raises(OutputValidationError):
        write_manifest(
            tmp_path / "manifest.json",
            config=make_config(tmp_path, mode=mode),
            result=result,
            tool_versions=versions,
            started_at=now,
            finished_at=now,
            stage_durations=durations,
        )
