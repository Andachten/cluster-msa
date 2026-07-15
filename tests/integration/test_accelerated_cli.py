import json
from pathlib import Path

from cluster_msa.cli import main


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
        "one.a3m", "run.log", "three.a3m", "two.a3m"
    ]
    assert len(fake_colabfold_search.invocations()) == 1
    assert fake_mmseqs.count("easy-cluster") == 1
    assert fake_mmseqs.count("search") == 1
    assert fake_mmseqs.count("result2msa") == 1
    assert not list(work.iterdir())


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
    assert [command[0] for command in fake_mmseqs.invocations()] == ["easy-cluster"]
    log = (output / "run.log").read_text(encoding="utf-8")
    assert "fallback_reason: no_non_representatives" in log
    assert "standard fallback" in log


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
    )) == 1
    assert not output.exists()
    assert list(work.glob("accelerated-*"))
