import json
from pathlib import Path

import pytest

from cluster_msa.cli import main
from cluster_msa.errors import OutputValidationError


def accelerated_args(input_path, output, database, colabfold, mmseqs, work):
    return [
        "accelerated", "--input", str(input_path), "--output-dir", str(output),
        "--db-path", str(database), "--colabfold-search", str(colabfold),
        "--mmseqs", str(mmseqs), "--work-dir", str(work),
        "--tmp-dir", str(work.parent / "tmp"),
    ]


def test_accelerated_cli_runs_documented_pipeline_once_per_batch(
    tmp_path: Path, fake_database, fake_colabfold_search, fake_mmseqs, monkeypatch
):
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\ntwo,FGHI\nthree,KLMN\n", encoding="utf-8")
    output = tmp_path / "output"
    work = tmp_path / "work"
    monkeypatch.setenv("FAKE_COLABFOLD_ADD_HIT", "1")
    monkeypatch.setenv("FAKE_MMSEQS_CLUSTER_TSV", "one\tone\none\ttwo\nthree\tthree\n")

    assert main(accelerated_args(
        input_path, output, fake_database, fake_colabfold_search.executable,
        fake_mmseqs.executable, work
    )) == 0

    assert sorted(path.name for path in output.iterdir()) == [
        "one.a3m", "run.log", "run_manifest.json", "three.a3m", "two.a3m"
    ]
    assert len(fake_colabfold_search.invocations()) == 2
    commands = fake_mmseqs.invocations()
    assert commands[0] == ["--version"]
    pipeline_commands = [command for command in commands if command != ["--version"]]
    assert [command[0] for command in pipeline_commands] == [
        "easy-cluster", "createdb", "createindex", "createdb", "search", "result2msa"
    ]
    run_roots = {
        next(parent for parent in Path(argument).parents if parent.name.startswith("accelerated-"))
        for command in pipeline_commands
        for argument in command[1:]
        if argument.startswith("/") and "accelerated-" in argument
    }
    assert len(run_roots) == 1
    run_root = run_roots.pop()
    for command in pipeline_commands:
        for argument in command[1:]:
            if argument.startswith("/") and "accelerated-" in argument:
                assert Path(argument).is_relative_to(run_root)
    assert all(str(work.parent / "tmp") not in command for command in pipeline_commands)
    assert not list(work.iterdir())
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tools"]["mmseqs"]["version"] == "fake-mmseqs 1.0"
    assert manifest["result"] == {
        "expected_count": 3, "generated_count": 3, "representative_count": 2,
        "nonrepresentative_count": 1, "fallback_reason": None,
    }
    assert set(manifest["timing"]["stage_durations_seconds"]) == {
        "clustering", "representative_search", "compact_database",
        "nonrepresentative_search", "merge_and_staging", "output_validation", "total"
    }
    assert manifest["timing"]["timing_scope"] == "through_pre_manifest_finalization"
    durations = manifest["timing"]["stage_durations_seconds"]
    assert durations["total"] >= max(
        value for name, value in durations.items() if name != "total"
    ) >= 0


def test_accelerated_cli_generates_af3_for_every_record(
    tmp_path: Path, fake_database, fake_colabfold_search, fake_mmseqs, monkeypatch
):
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setenv("FAKE_COLABFOLD_ADD_HIT", "1")
    args = accelerated_args(
        input_path, output, fake_database, fake_colabfold_search.executable,
        fake_mmseqs.executable, tmp_path / "work"
    )

    assert main([*args, "--af3-json"]) == 0

    assert "--af3-json" in fake_colabfold_search.invocation()["argv"]
    for record_id in ("one", "two"):
        assert (output / f"{record_id}.a3m").exists()
        assert json.loads((output / f"{record_id}_data.json").read_text())["name"] == record_id


def test_accelerated_cli_fallback_searches_original_csv_without_compact_commands(
    tmp_path: Path, fake_database, fake_colabfold_search, fake_mmseqs, monkeypatch
):
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,acde\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setenv("FAKE_MMSEQS_CLUSTER_TSV", "one\tone\ntwo\ttwo\n")

    assert main(accelerated_args(
        input_path, output, fake_database, fake_colabfold_search.executable,
        fake_mmseqs.executable, tmp_path / "work"
    )) == 0

    assert fake_colabfold_search.invocation()["input_csv"] == "id,sequence\none,ACDE\ntwo,FGHI\n"
    assert [command[0] for command in fake_mmseqs.invocations()] == ["--version", "easy-cluster"]
    log = (output / "run.log").read_text(encoding="utf-8")
    assert "fallback_reason: no_non_representatives" in log
    assert "standard fallback" in log
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["result"]["fallback_reason"] == "no_non_representatives"
    assert set(manifest["timing"]["stage_durations_seconds"]) == {
        "clustering", "standard_search", "output_validation", "total"
    }


def test_accelerated_cli_retains_failed_run_and_publishes_nothing(
    tmp_path: Path, fake_database, fake_colabfold_search, fake_mmseqs, monkeypatch
):
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"
    work = tmp_path / "work"
    monkeypatch.setenv("FAKE_COLABFOLD_ADD_HIT", "1")
    monkeypatch.setenv("FAKE_MMSEQS_FAIL", "search")

    assert main(accelerated_args(
        input_path, output, fake_database, fake_colabfold_search.executable,
        fake_mmseqs.executable, work
    )) == 4
    assert not output.exists()
    assert list(work.glob("accelerated-*"))
    assert not list(work.rglob("run_manifest.json"))


@pytest.mark.parametrize(
    "failure",
    ["easy-cluster:1", "createdb:1", "createindex:1", "createdb:2", "search:1", "result2msa:1"],
)
def test_accelerated_cli_retains_diagnostics_for_every_mmseqs_failure(
    tmp_path: Path, fake_database, fake_colabfold_search, fake_mmseqs, monkeypatch,
    capsys, failure
):
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"
    work = tmp_path / "work"
    monkeypatch.setenv("FAKE_COLABFOLD_ADD_HIT", "1")
    monkeypatch.setenv("FAKE_MMSEQS_FAIL_AT", failure)

    result = main(accelerated_args(
        input_path, output, fake_database, fake_colabfold_search.executable,
        fake_mmseqs.executable, work
    ))

    assert result == 4
    assert not output.exists()
    runs = list(work.glob("accelerated-*"))
    assert len(runs) == 1
    assert list(runs[0].rglob("run.log"))
    assert "Traceback" not in capsys.readouterr().err


def test_accelerated_cli_rejects_malformed_result2msa_without_publishing(
    tmp_path: Path, fake_database, fake_colabfold_search, fake_mmseqs, monkeypatch, capsys
):
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"
    work = tmp_path / "work"
    monkeypatch.setenv("FAKE_COLABFOLD_ADD_HIT", "1")
    monkeypatch.setenv("FAKE_MMSEQS_RESULT2MSA_EMPTY", "1")

    result = main(accelerated_args(
        input_path, output, fake_database, fake_colabfold_search.executable,
        fake_mmseqs.executable, work
    ))

    assert result == 5
    assert not output.exists()
    assert list(work.glob("accelerated-*/output-*/run.log"))
    assert "Traceback" not in capsys.readouterr().err


def test_accelerated_cli_marks_retained_manifest_failed_when_publication_fails(
    tmp_path: Path, fake_database, fake_colabfold_search, fake_mmseqs, monkeypatch
):
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"
    work = tmp_path / "work"
    monkeypatch.setenv("FAKE_COLABFOLD_ADD_HIT", "1")

    def fail_publication(*args, **kwargs):
        raise OutputValidationError("private publication details")

    monkeypatch.setattr("cluster_msa.accelerated.publish_outputs", fail_publication)
    result = main(accelerated_args(
        input_path, output, fake_database, fake_colabfold_search.executable,
        fake_mmseqs.executable, work
    ))

    assert result == 5
    assert not output.exists()
    manifests = list(work.rglob("run_manifest.json"))
    assert len(manifests) == 1
    document = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert (document["status"], document["failure_stage"], document["error"]) == (
        "failed", "publication", "output publication failed"
    )
