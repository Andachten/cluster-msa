import argparse
from pathlib import Path

import pytest

from cluster_msa.config import (
    build_run_config,
    resolve_database,
    resolve_executable,
    validate_database,
)
from cluster_msa.errors import ConfigurationError


def executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_resolve_executable_uses_explicit_then_environment_then_path(tmp_path, monkeypatch):
    explicit = executable(tmp_path / "explicit")
    from_env = executable(tmp_path / "from-env")
    from_path = executable(tmp_path / "from-path")
    monkeypatch.setenv("PATH", str(tmp_path))

    assert resolve_executable(str(explicit), "TOOL", "from-path") == explicit
    assert resolve_executable(None, "TOOL", "from-path") == from_path

    monkeypatch.setenv("TOOL", str(from_env))
    assert resolve_executable(None, "TOOL", "from-path") == from_env


@pytest.mark.parametrize("kind", ["missing", "directory", "not-executable"])
def test_resolve_executable_rejects_invalid_configured_path(tmp_path, monkeypatch, kind):
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()
    elif kind == "not-executable":
        path.write_text("tool", encoding="utf-8")
        path.chmod(0o644)

    monkeypatch.setenv("TOOL", str(path))
    with pytest.raises(ConfigurationError):
        resolve_executable(None, "TOOL", "unused")


def test_resolve_database_precedence_and_component_validation(tmp_path):
    explicit = tmp_path / "explicit-db"
    environ_db = tmp_path / "environment-db"
    for path in (explicit, environ_db):
        path.mkdir()
        (path / "uniref30_2024.tar.gz").write_text("", encoding="utf-8")
        (path / "colabfold_envdb_2024.tar.gz").write_text("", encoding="utf-8")

    assert resolve_database(explicit, {"CLUSTER_MSA_DB": str(environ_db)}) == explicit
    assert resolve_database(None, {"CLUSTER_MSA_DB": str(environ_db)}) == environ_db

    validate_database(environ_db)


@pytest.mark.parametrize("bad", ["missing", "file", "incomplete"])
def test_validate_database_rejects_invalid_database(tmp_path, bad):
    path = tmp_path / bad
    if bad == "file":
        path.write_text("", encoding="utf-8")
    elif bad == "incomplete":
        path.mkdir()
        (path / "uniref30_only").write_text("", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        validate_database(path)


def test_validate_database_rejects_prefixed_directories(tmp_path):
    (tmp_path / "uniref30_directory").mkdir()
    (tmp_path / "colabfold_envdb_directory").mkdir()

    with pytest.raises(ConfigurationError):
        validate_database(tmp_path)


def args_for(tmp_path, **overrides):
    (tmp_path / "input.csv").write_text("id,sequence\nexample,ACDE\n", encoding="utf-8")
    values = {
        "mode": "standard",
        "input": tmp_path / "input.csv",
        "output": tmp_path / "output",
        "db": None,
        "colabfold_search": None,
        "mmseqs": None,
        "threads": 1,
        "gpu": None,
        "gpus": None,
        "af3_json": False,
        "tmp": None,
        "work": None,
        "keep_work": False,
        "overwrite": False,
        "verbose": False,
        "cluster_identity": 0.7,
        "cluster_coverage": 0.8,
        "cluster_mode": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_run_config_resolves_tools_and_portable_defaults(tmp_path, monkeypatch):
    search = executable(tmp_path / "colabfold_search")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path))
    config = build_run_config(
        args_for(tmp_path),
        {"CLUSTER_MSA_DB": str(db), "CLUSTER_MSA_COLABFOLD_SEARCH": str(search)},
    )

    assert config.toolchain.colabfold_search == search
    assert config.toolchain.mmseqs is None
    assert config.threads == 1
    assert config.gpus == "0"
    assert config.tmp_dir == Path(".cluster-msa-tmp")
    assert config.work_dir == Path(".cluster-msa-work")


def test_build_run_config_requires_mmseqs_for_accelerated(tmp_path, monkeypatch):
    search = executable(tmp_path / "colabfold_search")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ConfigurationError, match="mmseqs"):
        build_run_config(
            args_for(tmp_path, mode="accelerated"),
            {"CLUSTER_MSA_COLABFOLD_SEARCH": str(search)},
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("threads", 0),
        ("cluster_identity", 0),
        ("cluster_identity", 1.1),
        ("cluster_coverage", 0),
        ("cluster_mode", -1),
    ],
)
def test_build_run_config_rejects_invalid_values(tmp_path, field, value):
    with pytest.raises(ConfigurationError):
        build_run_config(args_for(tmp_path, **{field: value}), {})


def test_build_run_config_gpu_flag_and_environment_precedence(tmp_path, monkeypatch):
    search = executable(tmp_path / "search")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    monkeypatch.setenv("CLUSTER_MSA_THREADS", "8")

    config = build_run_config(
        args_for(tmp_path, threads=2, gpu=True, gpus="2,3"),
        {"CLUSTER_MSA_DB": str(db), "CLUSTER_MSA_COLABFOLD_SEARCH": str(search)},
    )

    assert config.threads == 2
    assert config.gpu is True
    assert config.gpus == "2,3"


def test_build_run_config_honors_no_gpu_flag(tmp_path):
    search = executable(tmp_path / "search")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    args = args_for(tmp_path, gpu=None, no_gpu=True)
    config = build_run_config(
        args,
        {"CLUSTER_MSA_DB": str(db), "CLUSTER_MSA_COLABFOLD_SEARCH": str(search)},
    )

    assert config.gpu is False
