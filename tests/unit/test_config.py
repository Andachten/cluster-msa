import argparse
import tempfile
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


@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_resolve_database_preserves_relative_user_spelling(tmp_path, monkeypatch, source):
    database = tmp_path / "relative-db"
    database.mkdir()
    (database / "uniref30_component").write_text("", encoding="utf-8")
    (database / "colabfold_envdb_component").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    explicit = Path("relative-db") if source == "explicit" else None
    environ = {"CLUSTER_MSA_DB": "relative-db"} if source == "environment" else {}

    resolved = resolve_database(explicit, environ)
    assert resolved == database
    assert resolved.is_absolute()


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
        "gpu": True,
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
        {"CLUSTER_MSA_DB": str(db), "COLABFOLD_SEARCH": str(search)},
    )

    assert config.toolchain.colabfold_search == search
    assert config.toolchain.mmseqs is None
    assert config.threads == 1
    assert config.gpu is True
    assert config.gpus == ""
    assert config.tmp_dir == Path(".cluster-msa-tmp")
    assert config.work_dir == Path(".cluster-msa-tmp") / "cluster-msa-work"
    assert config.db_path_supplied == str(db)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("threads", True, "threads"),
        ("threads", 1.5, "threads"),
        ("threads", "1", "threads"),
        ("cluster_identity", True, "cluster"),
        ("cluster_identity", "0.7", "cluster"),
        ("cluster_identity", float("nan"), "cluster"),
        ("cluster_identity", 0, "cluster"),
        ("cluster_coverage", False, "cluster"),
        ("cluster_coverage", None, "cluster"),
        ("cluster_coverage", 1.1, "cluster"),
        ("cluster_mode", True, "cluster"),
        ("cluster_mode", 1.5, "cluster"),
        ("cluster_mode", "0", "cluster"),
        ("gpus", 7, "GPU IDs"),
        ("gpu", 1, "GPU flag"),
        ("gpu", "true", "GPU flag"),
        ("af3_json", 0, "AF3 flag"),
        ("af3_json", "false", "AF3 flag"),
    ],
)
def test_build_run_config_rejects_invalid_typed_values(tmp_path, name, value, message):
    search = executable(tmp_path / "search")
    database = tmp_path / "database"
    database.mkdir()
    (database / "uniref30_component").write_text("", encoding="utf-8")
    (database / "colabfold_envdb_component").write_text("", encoding="utf-8")
    arguments = args_for(
        tmp_path,
        colabfold_search=str(search),
        db=database,
        **{name: value},
    )

    with pytest.raises(ConfigurationError, match=message):
        build_run_config(arguments, {})


@pytest.mark.parametrize(
    ("explicit", "environment", "supplied"),
    [
        ("./database", None, "./database"),
        (None, "./database", "./database"),
        ("~/database", None, "~/database"),
    ],
)
def test_build_run_config_preserves_exact_database_spelling(
    tmp_path, monkeypatch, explicit, environment, supplied
):
    home = tmp_path / "home"
    home.mkdir()
    database_parent = home if supplied.startswith("~") else tmp_path
    database = database_parent / "database"
    database.mkdir()
    (database / "uniref30_component").write_text("", encoding="utf-8")
    (database / "colabfold_envdb_component").write_text("", encoding="utf-8")
    search = executable(tmp_path / "search")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    environ = {"COLABFOLD_SEARCH": str(search)}
    if environment is not None:
        environ["CLUSTER_MSA_DB"] = environment

    config = build_run_config(args_for(tmp_path, db=explicit), environ)

    assert config.db_path_supplied == supplied
    assert config.db_path == database.resolve()


@pytest.mark.parametrize(
    ("output", "tmp"),
    [(Path("."), Path(".cluster-msa-tmp")), (Path(".cluster-msa-work"), Path("tmp"))],
)
def test_standard_work_root_is_resolved_outside_output(tmp_path, monkeypatch, output, tmp):
    search = executable(tmp_path / "search")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = build_run_config(
        args_for(tmp_path, output=output, tmp=tmp, colabfold_search=str(search), db=db), {}
    )

    assert config.work_dir.resolve() != config.output_dir.resolve()
    assert not config.work_dir.resolve().is_relative_to(config.output_dir.resolve())


def test_standard_work_root_falls_back_when_explicit_tmp_is_inside_output(tmp_path):
    search = executable(tmp_path / "search")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    output = tmp_path / "output"

    config = build_run_config(
        args_for(
            tmp_path,
            output=output,
            tmp=output / "tmp",
            colabfold_search=str(search),
            db=db,
        ),
        {},
    )

    assert config.work_dir == Path(tempfile.gettempdir()) / "cluster-msa-work"
    assert not config.work_dir.resolve().is_relative_to(output.resolve())


@pytest.mark.parametrize("work_suffix", [Path("."), Path("work")])
def test_accelerated_rejects_work_root_inside_output(tmp_path, work_suffix):
    search = executable(tmp_path / "search")
    mmseqs = executable(tmp_path / "mmseqs")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ConfigurationError, match="work.*output"):
        build_run_config(
            args_for(
                tmp_path,
                mode="accelerated",
                output=output,
                work=output / work_suffix,
                colabfold_search=str(search),
                mmseqs=str(mmseqs),
                db=db,
            ),
            {},
        )


def test_accelerated_rejects_work_root_that_is_a_file(tmp_path):
    search = executable(tmp_path / "search")
    mmseqs = executable(tmp_path / "mmseqs")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    work = tmp_path / "work"
    work.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="work directory is not a directory"):
        build_run_config(
            args_for(
                tmp_path,
                mode="accelerated",
                work=work,
                colabfold_search=str(search),
                mmseqs=str(mmseqs),
                db=db,
            ),
            {},
        )


def test_build_run_config_requires_mmseqs_for_accelerated(tmp_path, monkeypatch):
    search = executable(tmp_path / "colabfold_search")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ConfigurationError, match="mmseqs"):
        build_run_config(
            args_for(tmp_path, mode="accelerated"),
            {"COLABFOLD_SEARCH": str(search)},
        )


def test_build_run_config_tool_arguments_override_documented_environment(tmp_path):
    explicit_search = executable(tmp_path / "explicit-search")
    explicit_mmseqs = executable(tmp_path / "explicit-mmseqs")
    environment_search = executable(tmp_path / "environment-search")
    environment_mmseqs = executable(tmp_path / "environment-mmseqs")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")

    config = build_run_config(
        args_for(
            tmp_path,
            mode="accelerated",
            colabfold_search=str(explicit_search),
            mmseqs=str(explicit_mmseqs),
        ),
        {
            "CLUSTER_MSA_DB": str(db),
            "COLABFOLD_SEARCH": str(environment_search),
            "MMSEQS": str(environment_mmseqs),
        },
    )

    assert config.toolchain.colabfold_search == explicit_search
    assert config.toolchain.mmseqs == explicit_mmseqs


def test_build_run_config_uses_documented_tool_environment(tmp_path, monkeypatch):
    search = executable(tmp_path / "environment-search")
    mmseqs = executable(tmp_path / "environment-mmseqs")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", "")

    config = build_run_config(
        args_for(tmp_path, mode="accelerated"),
        {
            "CLUSTER_MSA_DB": str(db),
            "COLABFOLD_SEARCH": str(search),
            "MMSEQS": str(mmseqs),
        },
    )

    assert config.toolchain.colabfold_search == search
    assert config.toolchain.mmseqs == mmseqs


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


@pytest.mark.parametrize("field", ["input", "output"])
def test_build_run_config_reports_missing_required_paths(tmp_path, field):
    with pytest.raises(ConfigurationError, match=field):
        build_run_config(args_for(tmp_path, **{field: None}), {})


def test_build_run_config_expands_all_user_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    input_path = home / "input.csv"
    input_path.write_text("id,sequence\nexample,ACDE\n", encoding="utf-8")
    executable(home / "search")
    db = home / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    config = build_run_config(
        args_for(
            tmp_path,
            input=Path("~/input.csv"),
            output=Path("~/output"),
            db=Path("~/db"),
            colabfold_search="~/search",
            tmp=Path("~/tmp"),
            work=Path("~/work"),
        ),
        {},
    )

    assert config.input_path == home / "input.csv"
    assert config.output_dir == home / "output"
    assert config.db_path == home / "db"
    assert config.toolchain.colabfold_search == home / "search"
    assert config.tmp_dir == home / "tmp"
    assert config.work_dir == home / "tmp" / "cluster-msa-work"


def test_build_run_config_gpu_flag_and_environment_precedence(tmp_path, monkeypatch):
    search = executable(tmp_path / "search")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    monkeypatch.setenv("CLUSTER_MSA_THREADS", "8")

    config = build_run_config(
        args_for(tmp_path, threads=2, gpu=True, gpus="2,3"),
        {"CLUSTER_MSA_DB": str(db), "COLABFOLD_SEARCH": str(search)},
    )

    assert config.threads == 2
    assert config.gpu is True
    assert config.gpus == "2,3"


def test_build_run_config_honors_gpu_value(tmp_path):
    search = executable(tmp_path / "search")
    db = tmp_path / "db"
    db.mkdir()
    (db / "uniref30_component").write_text("", encoding="utf-8")
    (db / "colabfold_envdb_component").write_text("", encoding="utf-8")
    args = args_for(tmp_path, gpu=False)
    config = build_run_config(
        args,
        {"CLUSTER_MSA_DB": str(db), "COLABFOLD_SEARCH": str(search)},
    )

    assert config.gpu is False
