import json
import shutil
from pathlib import Path

from cluster_msa.cli import main
from cluster_msa.errors import OutputValidationError


def test_standard_cli_publishes_all_msas_and_log(
    tmp_path: Path, fake_database: Path, fake_colabfold_search
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,acde\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"
    mmseqs = tmp_path / "mmseqs"
    tmp_dir = tmp_path / "tmp"
    mmseqs_marker = tmp_path / "mmseqs-invoked"
    mmseqs.write_text(f"#!/bin/sh\ntouch {mmseqs_marker}\nexit 99\n", encoding="utf-8")
    mmseqs.chmod(0o755)

    result = main(
        [
            "standard",
            "--input",
            str(input_path),
            "--output-dir",
            str(output),
            "--db-path",
            str(fake_database),
            "--colabfold-search",
            str(fake_colabfold_search.executable),
            "--mmseqs",
            str(mmseqs),
            "--tmp-dir",
            str(tmp_dir),
        ]
    )

    assert result == 0
    assert (output / "one.a3m").read_text(encoding="utf-8") == ">one\nACDE\n"
    assert (output / "two.a3m").read_text(encoding="utf-8") == ">two\nFGHI\n"
    assert "fake search complete" in (output / "run.log").read_text(encoding="utf-8")
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["mode"] == "standard"
    assert manifest["input"] == {"path": str(input_path), "count": 2}
    assert manifest["tools"]["colabfold_search"]["version"] == "fake-colabfold-search 1.0"
    assert set(manifest["timing"]["stage_durations_seconds"]) == {
        "full_database_search", "output_validation", "total"
    }
    assert manifest["timing"]["timing_scope"] == "through_pre_manifest_finalization"
    durations = manifest["timing"]["stage_durations_seconds"]
    assert durations["total"] >= max(
        durations["full_database_search"], durations["output_validation"]
    ) >= 0
    assert all(value >= 0 for value in manifest["timing"]["stage_durations_seconds"].values())
    derived_work = tmp_dir / "cluster-msa-work"
    assert derived_work.is_dir()
    assert not list(derived_work.iterdir())
    assert fake_colabfold_search.invocations()[0]["argv"] == ["--version"]
    assert len(fake_colabfold_search.invocations()) == 2
    assert not mmseqs_marker.exists()
    staging = Path(fake_colabfold_search.invocation()["argv"][2]).resolve()
    assert not staging.is_relative_to(output.resolve())


def test_standard_cli_supports_af3_json_gpu_and_cpu_environment(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    output = tmp_path / "af3-output"
    tmp_dir = tmp_path / "tmp"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "inherited")

    assert (
        main(
            [
                "standard",
                "--input",
                str(input_path),
                "--output-dir",
                str(output),
                "--db-path",
                str(fake_database),
                "--colabfold-search",
                str(fake_colabfold_search.executable),
                "--gpus",
                "2,3",
                "--af3-json",
                "--tmp-dir",
                str(tmp_dir),
            ]
        )
        == 0
    )
    invocation = fake_colabfold_search.invocation()
    assert invocation["cuda_visible_devices"] == "2,3"
    assert "--af3-json" in invocation["argv"]
    assert (output / "one_data.json").exists()

    cpu_output = tmp_path / "cpu-output"
    assert (
        main(
            [
                "standard",
                "--input",
                str(input_path),
                "--output-dir",
                str(cpu_output),
                "--db-path",
                str(fake_database),
                "--colabfold-search",
                str(fake_colabfold_search.executable),
                "--no-gpu",
                "--gpus",
                "9",
                "--tmp-dir",
                str(tmp_dir),
            ]
        )
        == 0
    )
    cpu_invocation = fake_colabfold_search.invocation()
    assert cpu_invocation["cuda_visible_devices"] is None
    assert cpu_invocation["argv"][-2:] == ["--gpu", "0"]
    assert len(fake_colabfold_search.invocations()) == 4
    assert "ignored" in (cpu_output / "run.log").read_text(encoding="utf-8").lower()


def test_standard_cli_retains_work_on_tool_failure_and_never_publishes_partial_output(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"
    tmp_dir = tmp_path / "tmp"
    work = tmp_dir / "cluster-msa-work"
    monkeypatch.setenv("FAKE_COLABFOLD_SKIP_ID", "two")

    assert (
        main(
            [
                "standard",
                "--input",
                str(input_path),
                "--output-dir",
                str(output),
                "--db-path",
                str(fake_database),
                "--colabfold-search",
                str(fake_colabfold_search.executable),
                "--tmp-dir",
                str(tmp_dir),
            ]
        )
        == 5
    )
    assert not output.exists()
    assert list(work.rglob("one.a3m"))
    assert list(work.rglob("run.log"))

    monkeypatch.setenv("FAKE_COLABFOLD_FAIL", "1")
    failure_output = tmp_path / "failure-output"
    assert (
        main(
            [
                "standard",
                "--input",
                str(input_path),
                "--output-dir",
                str(failure_output),
                "--db-path",
                str(fake_database),
                "--colabfold-search",
                str(fake_colabfold_search.executable),
                "--tmp-dir",
                str(tmp_dir),
            ]
        )
        == 4
    )
    assert not failure_output.exists()
    assert not list(work.rglob("run_manifest.json"))
    assert len(list(work.rglob("run.log"))) == 2


def test_standard_cli_rejects_nonempty_output_before_external_compute(
    tmp_path: Path, fake_database: Path, fake_colabfold_search
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "old").write_text("old", encoding="utf-8")

    assert (
        main(
            [
                "standard",
                "--input",
                str(input_path),
                "--output-dir",
                str(output),
                "--db-path",
                str(fake_database),
                "--colabfold-search",
                str(fake_colabfold_search.executable),
            ]
        )
        == 5
    )
    assert fake_colabfold_search.invocation_path.exists() is False


def test_standard_cli_rejects_file_tmp_dir_without_traceback(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, capsys
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    invalid_tmp = tmp_path / "tmp-file"
    invalid_tmp.write_text("not a directory", encoding="utf-8")

    result = main(
        [
            "standard",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--db-path",
            str(fake_database),
            "--colabfold-search",
            str(fake_colabfold_search.executable),
            "--tmp-dir",
            str(invalid_tmp),
        ]
    )

    captured = capsys.readouterr()
    assert result == 3
    assert "Traceback" not in captured.err
    assert "tmp" in captured.err.lower()
    assert not fake_colabfold_search.invocation_path.exists()


def test_standard_cli_treats_version_failure_as_external_error(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    output = tmp_path / "output"
    work = tmp_path / "tmp" / "cluster-msa-work"
    monkeypatch.setenv("FAKE_COLABFOLD_VERSION_FAIL", "1")

    result = main(
        [
            "standard",
            "--input",
            str(input_path),
            "--output-dir",
            str(output),
            "--db-path",
            str(fake_database),
            "--colabfold-search",
            str(fake_colabfold_search.executable),
            "--tmp-dir",
            str(tmp_path / "tmp"),
        ]
    )

    stderr = capsys.readouterr().err
    assert result == 4
    assert "version" in stderr
    assert "log:" in stderr
    assert not output.exists()
    assert list(work.rglob("run.log"))
    assert not list(work.rglob("run_manifest.json"))


def test_standard_cli_marks_retained_manifest_failed_when_publication_fails(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    output = tmp_path / "output"
    tmp_dir = tmp_path / "tmp"

    def fail_publication(*args, **kwargs):
        raise OutputValidationError("publication failed with token=secret-value")

    monkeypatch.setattr("cluster_msa.standard.publish_outputs", fail_publication)
    result = main(
        [
            "standard", "--input", str(input_path), "--output-dir", str(output),
            "--db-path", str(fake_database), "--colabfold-search",
            str(fake_colabfold_search.executable), "--tmp-dir", str(tmp_dir), "--keep-work",
        ]
    )

    assert result == 5
    assert not output.exists()
    manifests = list((tmp_dir / "cluster-msa-work").rglob("run_manifest.json"))
    assert len(manifests) == 2
    for manifest in manifests:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        assert document["status"] == "failed"
        assert document["failure_stage"] == "publication"
        assert document["error"] == "output publication failed"
        assert "secret-value" not in manifest.read_text(encoding="utf-8")


def test_standard_cli_manifest_preserves_relative_database_spelling(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    output = tmp_path / "output"
    relative_database = f"./{fake_database.name}"
    monkeypatch.chdir(tmp_path)

    result = main(
        [
            "standard", "--input", str(input_path), "--output-dir", str(output),
            "--db-path", relative_database, "--colabfold-search",
            str(fake_colabfold_search.executable), "--tmp-dir", str(tmp_path / "tmp"),
        ]
    )

    assert result == 0
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["database"] == {
        "path": relative_database,
        "resolved_path": str(fake_database.resolve()),
    }
    search = [
        invocation for invocation in fake_colabfold_search.invocations()
        if invocation["argv"] != ["--version"]
    ][0]
    assert search["argv"][1] == str(fake_database.resolve())


def test_standard_keep_work_failure_marks_staging_manifest_failed(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    output = tmp_path / "output"
    tmp_dir = tmp_path / "tmp"

    monkeypatch.setattr(
        "cluster_msa.standard.shutil.copytree",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("retention denied")),
    )
    result = main(
        [
            "standard", "--input", str(input_path), "--output-dir", str(output),
            "--db-path", str(fake_database), "--colabfold-search",
            str(fake_colabfold_search.executable), "--tmp-dir", str(tmp_dir), "--keep-work",
        ]
    )

    assert result == 5
    assert not output.exists()
    manifests = list((tmp_dir / "cluster-msa-work").rglob("run_manifest.json"))
    assert len(manifests) == 1
    document = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert (document["status"], document["failure_stage"]) == ("failed", "work_retention")
    assert document["error"] == "work retention failed"


def test_standard_partial_retention_failure_marks_both_manifests_failed(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    output = tmp_path / "output"
    tmp_dir = tmp_path / "tmp"
    real_copytree = shutil.copytree

    def copy_then_fail(source, destination, *args, **kwargs):
        real_copytree(source, destination, *args, **kwargs)
        raise OSError("post-copy retention failure")

    monkeypatch.setattr("cluster_msa.standard.shutil.copytree", copy_then_fail)
    result = main(
        [
            "standard", "--input", str(input_path), "--output-dir", str(output),
            "--db-path", str(fake_database), "--colabfold-search",
            str(fake_colabfold_search.executable), "--tmp-dir", str(tmp_dir), "--keep-work",
        ]
    )

    assert result == 5
    assert not output.exists()
    manifests = list((tmp_dir / "cluster-msa-work").rglob("run_manifest.json"))
    assert len(manifests) == 2
    for manifest in manifests:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        assert (document["status"], document["failure_stage"]) == (
            "failed", "work_retention"
        )
