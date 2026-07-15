import pytest

from cluster_msa.errors import OutputValidationError
from cluster_msa.models import RunConfig, SequenceRecord, Toolchain
from cluster_msa.standard import run_full_database_search, run_standard


def test_run_full_database_search_uses_exact_colabfold_command(tmp_path, monkeypatch):
    executable = tmp_path / "colabfold_search"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    database = tmp_path / "db"
    database.mkdir()
    input_csv = tmp_path / "inputs.csv"
    destination = tmp_path / "destination"
    log_path = tmp_path / "run.log"
    config = RunConfig(
        mode="standard",
        input_path=input_csv,
        output_dir=destination,
        db_path=database,
        toolchain=Toolchain(executable, None),
        threads=4,
        gpu=True,
        gpus="2,3",
        af3_json=True,
        tmp_dir=tmp_path / "tmp",
        work_dir=tmp_path / "work",
        keep_work=False,
        overwrite=False,
        verbose=False,
    )
    captured = {}

    def fake_run_command(command, *, stage, log_path, env, verbose):
        captured.update(command=command, stage=stage, log_path=log_path, env=env, verbose=verbose)

    monkeypatch.setattr("cluster_msa.standard.run_command", fake_run_command)

    run_full_database_search(
        (SequenceRecord("one", "ACDE"),), input_csv, destination, config, log_path
    )

    assert captured["command"] == [
        str(executable),
        str(input_csv),
        str(database),
        str(destination),
        "--threads",
        "4",
        "--gpu",
        "1",
        "--af3-json",
    ]
    assert captured["env"] == {"CUDA_VISIBLE_DEVICES": "2,3"}
    assert captured["log_path"] == log_path


@pytest.mark.parametrize(
    ("gpu", "gpus", "expected"),
    [(True, "", None), (False, "", {}), (True, "5", {"CUDA_VISIBLE_DEVICES": "5"})],
)
def test_run_full_database_search_sets_child_cuda_policy(
    tmp_path, monkeypatch, gpu, gpus, expected
):
    executable = tmp_path / "search"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    config = RunConfig(
        "standard", tmp_path / "in.csv", tmp_path / "out", tmp_path, Toolchain(executable, None),
        1, gpu, gpus, False, tmp_path / "tmp", tmp_path / "work", False, False, False,
    )
    captured = {}
    monkeypatch.setattr(
        "cluster_msa.standard.run_command",
        lambda command, **kwargs: captured.update(kwargs),
    )

    run_full_database_search(
        (SequenceRecord("one", "ACDE"),), config.input_path, config.output_dir, config,
        tmp_path / "run.log",
    )

    assert captured["env"] == expected


def test_run_standard_preflights_before_work_or_external_tool(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing").write_text("old", encoding="utf-8")
    config = RunConfig(
        "standard", tmp_path / "in.csv", output, tmp_path, Toolchain(tmp_path / "missing", None),
        1, False, "", False, tmp_path / "tmp", tmp_path / "work", False, False, False,
    )
    monkeypatch.setattr("cluster_msa.standard.run_command", lambda *args, **kwargs: pytest.fail())

    with pytest.raises(OutputValidationError):
        run_standard(config, (SequenceRecord("one", "ACDE"),))
    assert not (tmp_path / "work").exists()


def test_run_standard_retains_failure_diagnostics_and_cleans_success(tmp_path, monkeypatch):
    records = (SequenceRecord("one", "ACDE"),)
    config = RunConfig(
        "standard", tmp_path / "in.csv", tmp_path / "output", tmp_path, Toolchain(tmp_path / "search", None),
        1, False, "", False, tmp_path / "tmp", tmp_path / "work", False, False, False,
    )
    monkeypatch.setattr(
        "cluster_msa.standard.run_full_database_search",
        lambda records, input_csv, destination, config, log_path: log_path.write_text("failed", encoding="utf-8"),
    )
    with pytest.raises(OutputValidationError):
        run_standard(config, records)
    assert list((tmp_path / "work").rglob("run.log"))

    config = RunConfig(
        "standard", tmp_path / "in.csv", tmp_path / "success", tmp_path, Toolchain(tmp_path / "search", None),
        1, False, "", False, tmp_path / "tmp", tmp_path / "success-work", False, False, False,
    )
    def successful(records, input_csv, destination, config, log_path):
        (destination / "one.a3m").write_text(">one\nACDE\n", encoding="utf-8")
        log_path.write_text("ok", encoding="utf-8")
    monkeypatch.setattr("cluster_msa.standard.run_full_database_search", successful)
    result = run_standard(config, records)
    assert (result.mode, result.expected_count, result.generated_count) == ("standard", 1, 1)
    assert not list((tmp_path / "success-work").iterdir())


def test_run_standard_keep_work_retains_successful_staging(tmp_path, monkeypatch):
    records = (SequenceRecord("one", "ACDE"),)
    config = RunConfig(
        "standard", tmp_path / "in.csv", tmp_path / "output", tmp_path,
        Toolchain(tmp_path / "search", None), 1, False, "", False, tmp_path / "tmp",
        tmp_path / "work", True, False, False,
    )

    def successful(records, input_csv, destination, config, log_path):
        input_csv.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
        (destination / "one.a3m").write_text(">one\nACDE\n", encoding="utf-8")
        log_path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("cluster_msa.standard.run_full_database_search", successful)
    run_standard(config, records)

    retained = list((tmp_path / "work").glob("standard-*/retained"))
    assert len(retained) == 1
    assert (retained[0] / "canonical-input.csv").exists()
    assert (retained[0] / "one.a3m").exists()
    assert (retained[0] / "run.log").exists()
