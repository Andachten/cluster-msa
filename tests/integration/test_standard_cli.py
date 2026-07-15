from pathlib import Path

from cluster_msa.cli import main


def test_standard_cli_publishes_all_msas_and_log(
    tmp_path: Path, fake_database: Path, fake_colabfold_search
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,acde\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"

    result = main(
        [
            "standard",
            str(input_path),
            str(output),
            "--db",
            str(fake_database),
            "--colabfold-search",
            str(fake_colabfold_search.executable),
            "--work",
            str(tmp_path / "work"),
        ]
    )

    assert result == 0
    assert (output / "one.a3m").read_text(encoding="utf-8") == ">one\nACDE\n"
    assert (output / "two.a3m").read_text(encoding="utf-8") == ">two\nFGHI\n"
    assert "fake search complete" in (output / "run.log").read_text(encoding="utf-8")


def test_standard_cli_supports_af3_json_gpu_and_cpu_environment(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    output = tmp_path / "af3-output"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "inherited")

    assert main(
        [
            "standard", str(input_path), str(output), "--db", str(fake_database),
            "--colabfold-search", str(fake_colabfold_search.executable), "--gpus", "2,3",
            "--af3-json", "--work", str(tmp_path / "work-1"),
        ]
    ) == 0
    invocation = fake_colabfold_search.invocation()
    assert invocation["cuda_visible_devices"] == "2,3"
    assert "--af3-json" in invocation["argv"]
    assert (output / "one_data.json").exists()

    cpu_output = tmp_path / "cpu-output"
    assert main(
        [
            "standard", str(input_path), str(cpu_output), "--db", str(fake_database),
            "--colabfold-search", str(fake_colabfold_search.executable), "--no-gpu",
            "--work", str(tmp_path / "work-2"),
        ]
    ) == 0
    cpu_invocation = fake_colabfold_search.invocation()
    assert cpu_invocation["cuda_visible_devices"] is None
    assert cpu_invocation["argv"][-2:] == ["--gpu", "0"]


def test_standard_cli_retains_work_on_tool_failure_and_never_publishes_partial_output(
    tmp_path: Path, fake_database: Path, fake_colabfold_search, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\ntwo,FGHI\n", encoding="utf-8")
    output = tmp_path / "output"
    work = tmp_path / "work"
    monkeypatch.setenv("FAKE_COLABFOLD_SKIP_ID", "two")

    assert main(
        [
            "standard", str(input_path), str(output), "--db", str(fake_database),
            "--colabfold-search", str(fake_colabfold_search.executable), "--work", str(work),
        ]
    ) == 1
    assert not output.exists()
    assert list(work.rglob("one.a3m"))
    assert list(work.rglob("run.log"))

    monkeypatch.setenv("FAKE_COLABFOLD_FAIL", "1")
    failure_output = tmp_path / "failure-output"
    assert main(
        [
            "standard", str(input_path), str(failure_output), "--db", str(fake_database),
            "--colabfold-search", str(fake_colabfold_search.executable), "--work",
            str(tmp_path / "failure-work"),
        ]
    ) == 1
    assert not failure_output.exists()
    assert list((tmp_path / "failure-work").rglob("run.log"))


def test_standard_cli_rejects_nonempty_output_before_external_compute(
    tmp_path: Path, fake_database: Path, fake_colabfold_search
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "old").write_text("old", encoding="utf-8")

    assert main(
        [
            "standard", str(input_path), str(output), "--db", str(fake_database),
            "--colabfold-search", str(fake_colabfold_search.executable),
        ]
    ) == 1
    assert fake_colabfold_search.invocation_path.exists() is False
