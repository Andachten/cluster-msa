from pathlib import Path

import pytest

from cluster_msa.errors import ConfigurationError, ExternalToolError, OutputValidationError
from cluster_msa.models import ClusterResult, RunConfig, SequenceRecord, Toolchain


RECORDS = (
    SequenceRecord("one", "ACDE"),
    SequenceRecord("two", "FGHI"),
    SequenceRecord("three", "KLMN"),
)


def make_config(tmp_path: Path, **changes) -> RunConfig:
    values = dict(
        mode="accelerated",
        input_path=tmp_path / "input.csv",
        output_dir=tmp_path / "output",
        db_path=tmp_path / "database",
        toolchain=Toolchain(tmp_path / "colabfold_search", tmp_path / "mmseqs"),
        threads=4,
        gpu=False,
        gpus="",
        af3_json=False,
        tmp_dir=tmp_path / "tmp",
        work_dir=tmp_path / "work",
        keep_work=False,
        overwrite=False,
        verbose=False,
    )
    values.update(changes)
    return RunConfig(**values)


def test_run_accelerated_orders_phases_merges_outputs_and_returns_counts(tmp_path, monkeypatch):
    from cluster_msa.accelerated import run_accelerated

    events = []

    def cluster(records, **kwargs):
        events.append("cluster")
        return ClusterResult((records[0],), ((records[1], "one"), (records[2], "one")))

    def full(records, input_csv, destination, config, log_path):
        events.append("representatives")
        input_csv.parent.mkdir(parents=True, exist_ok=True)
        input_csv.write_text("id,sequence\none,ACDE\n", encoding="utf-8")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "one.a3m").write_text(">one\nACDE\n>hit\nAAAA\n", encoding="utf-8")
        log_path.write_text("representative search\n", encoding="utf-8")

    def compact(rep_dir, work_dir, config, log_path):
        events.append("compact")
        path = work_dir / "compactDB"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("db", encoding="utf-8")
        return path

    def search(records, compact_db, output_dir, config, log_path):
        events.append("nonrepresentatives")
        assert config.work_dir.is_relative_to(tmp_path / "work")
        output_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            (output_dir / f"{record.id}.a3m").write_text(
                f">{record.id}\n{record.sequence}\n>hit\nAAAA\n", encoding="utf-8"
            )

    monkeypatch.setattr("cluster_msa.accelerated.cluster_sequences", cluster)
    monkeypatch.setattr("cluster_msa.accelerated.run_full_database_search", full)
    monkeypatch.setattr("cluster_msa.accelerated.build_compact_database", compact)
    monkeypatch.setattr("cluster_msa.accelerated.search_compact_database", search)

    result = run_accelerated(make_config(tmp_path), RECORDS)

    assert events == ["cluster", "representatives", "compact", "nonrepresentatives"]
    assert result.mode == "accelerated"
    assert (result.expected_count, result.generated_count) == (3, 3)
    assert (result.representative_count, result.nonrepresentative_count) == (1, 2)
    assert result.fallback_reason is None
    assert sorted(path.name for path in (tmp_path / "output").iterdir()) == [
        "one.a3m", "run.log", "three.a3m", "two.a3m"
    ]
    assert "copied" in (tmp_path / "output" / "run.log").read_text(encoding="utf-8")
    assert not list((tmp_path / "work").iterdir())


def test_run_accelerated_af3_uses_colabfold_for_reps_and_writes_nonrep_json(tmp_path, monkeypatch):
    from cluster_msa.accelerated import run_accelerated

    observed = {"full": 0, "json": []}
    monkeypatch.setattr(
        "cluster_msa.accelerated.cluster_sequences",
        lambda records, **kwargs: ClusterResult((records[0],), ((records[1], "one"),)),
    )

    def full(records, input_csv, destination, config, log_path):
        observed["full"] += 1
        assert config.af3_json is True
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "one.a3m").write_text(">one\nACDE\n>hit\nAAAA\n", encoding="utf-8")
        (destination / "one_data.json").write_text('{"name":"one"}', encoding="utf-8")
        log_path.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr("cluster_msa.accelerated.run_full_database_search", full)
    monkeypatch.setattr("cluster_msa.accelerated.build_compact_database", lambda *args: tmp_path / "db")

    def search(records, compact_db, output_dir, config, log_path):
        output_dir.mkdir(parents=True)
        (output_dir / "two.a3m").write_text(">two\nFGHI\n>hit\nAAAA\n", encoding="utf-8")

    def json_writer(record, a3m, output):
        observed["json"].append(record.id)
        output.write_text('{"name":"two"}', encoding="utf-8")

    monkeypatch.setattr("cluster_msa.accelerated.search_compact_database", search)
    monkeypatch.setattr("cluster_msa.accelerated.write_af3_json", json_writer)

    run_accelerated(make_config(tmp_path, af3_json=True), RECORDS[:2])

    assert observed == {"full": 1, "json": ["two"]}
    assert sorted(path.name for path in (tmp_path / "output").glob("*_data.json")) == [
        "one_data.json", "two_data.json"
    ]


def test_run_accelerated_falls_back_with_original_records_and_no_compact_commands(
    tmp_path, monkeypatch
):
    from cluster_msa.accelerated import run_accelerated

    observed = {}
    monkeypatch.setattr(
        "cluster_msa.accelerated.cluster_sequences",
        lambda records, **kwargs: ClusterResult(tuple(records), ()),
    )

    def full(records, input_csv, destination, config, log_path):
        observed["records"] = tuple(records)
        observed["csv"] = input_csv
        input_csv.parent.mkdir(parents=True, exist_ok=True)
        input_csv.write_text(
            "id,sequence\n" + "".join(f"{r.id},{r.sequence}\n" for r in records), encoding="utf-8"
        )
        observed["csv_content"] = input_csv.read_text(encoding="utf-8")
        destination.mkdir(parents=True, exist_ok=True)
        for record in records:
            (destination / f"{record.id}.a3m").write_text(
                f">{record.id}\n{record.sequence}\n", encoding="utf-8"
            )

    monkeypatch.setattr("cluster_msa.accelerated.run_full_database_search", full)
    monkeypatch.setattr(
        "cluster_msa.accelerated.build_compact_database", lambda *args: pytest.fail("compact DB")
    )
    monkeypatch.setattr(
        "cluster_msa.accelerated.search_compact_database", lambda *args: pytest.fail("compact search")
    )

    result = run_accelerated(make_config(tmp_path), RECORDS)

    assert observed["records"] == RECORDS
    assert observed["csv_content"] == (
        "id,sequence\none,ACDE\ntwo,FGHI\nthree,KLMN\n"
    )
    assert result.mode == "accelerated"
    assert result.fallback_reason == "no_non_representatives"
    assert (result.representative_count, result.nonrepresentative_count) == (3, 0)
    log = (tmp_path / "output" / "run.log").read_text(encoding="utf-8")
    assert "fallback_reason: no_non_representatives" in log
    assert "standard fallback" in log


def test_run_accelerated_preflights_before_clustering(tmp_path, monkeypatch):
    from cluster_msa.accelerated import run_accelerated

    output = tmp_path / "output"
    output.mkdir()
    (output / "existing").write_text("old", encoding="utf-8")
    monkeypatch.setattr("cluster_msa.accelerated.cluster_sequences", lambda *args, **kwargs: pytest.fail())

    with pytest.raises(OutputValidationError):
        run_accelerated(make_config(tmp_path), RECORDS)
    assert not (tmp_path / "work").exists()


def test_run_accelerated_rejects_work_inside_output_before_clustering(tmp_path, monkeypatch):
    from cluster_msa.accelerated import run_accelerated

    monkeypatch.setattr("cluster_msa.accelerated.cluster_sequences", lambda *args, **kwargs: pytest.fail())

    with pytest.raises(ConfigurationError, match="outside output"):
        run_accelerated(
            make_config(tmp_path, work_dir=tmp_path / "output" / "work"), RECORDS
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("phase", ["cluster", "representatives", "compact", "nonrepresentatives", "af3"])
def test_run_accelerated_propagates_each_phase_failure_and_retains_work(tmp_path, monkeypatch, phase):
    from cluster_msa.accelerated import run_accelerated

    def fail(name):
        if phase == name:
            raise ExternalToolError(f"{name} failed")

    def cluster(records, **kwargs):
        fail("cluster")
        return ClusterResult((records[0],), ((records[1], "one"),))

    def full(records, input_csv, destination, config, log_path):
        fail("representatives")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "one.a3m").write_text(">one\nACDE\n>hit\nAAAA\n", encoding="utf-8")

    def compact(rep_dir, work_dir, config, log_path):
        fail("compact")
        return tmp_path / "db"

    def search(records, compact_db, output_dir, config, log_path):
        fail("nonrepresentatives")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "two.a3m").write_text(">two\nFGHI\n>hit\nAAAA\n", encoding="utf-8")

    def af3(*args):
        fail("af3")

    monkeypatch.setattr("cluster_msa.accelerated.cluster_sequences", cluster)
    monkeypatch.setattr("cluster_msa.accelerated.run_full_database_search", full)
    monkeypatch.setattr("cluster_msa.accelerated.build_compact_database", compact)
    monkeypatch.setattr("cluster_msa.accelerated.search_compact_database", search)
    monkeypatch.setattr("cluster_msa.accelerated.write_af3_json", af3)

    with pytest.raises(ExternalToolError, match="failed"):
        run_accelerated(make_config(tmp_path, af3_json=phase == "af3"), RECORDS[:2])
    assert list((tmp_path / "work").glob("accelerated-*"))
    assert not (tmp_path / "output").exists()


def test_run_accelerated_keep_work_retains_success(tmp_path, monkeypatch):
    from cluster_msa.accelerated import run_accelerated

    monkeypatch.setattr(
        "cluster_msa.accelerated.cluster_sequences",
        lambda records, **kwargs: ClusterResult(tuple(records), ()),
    )

    def full(records, input_csv, destination, config, log_path):
        destination.mkdir(parents=True, exist_ok=True)
        for record in records:
            (destination / f"{record.id}.a3m").write_text(
                f">{record.id}\n{record.sequence}\n", encoding="utf-8"
            )

    monkeypatch.setattr("cluster_msa.accelerated.run_full_database_search", full)
    run_accelerated(make_config(tmp_path, keep_work=True), RECORDS[:1])

    retained = list((tmp_path / "work").glob("accelerated-*/retained"))
    assert len(retained) == 1
    assert (retained[0] / "one.a3m").exists()


def test_run_accelerated_wraps_af3_decode_failure(tmp_path, monkeypatch):
    from cluster_msa.accelerated import run_accelerated

    monkeypatch.setattr(
        "cluster_msa.accelerated.cluster_sequences",
        lambda records, **kwargs: ClusterResult((records[0],), ((records[1], "one"),)),
    )

    def full(records, input_csv, destination, config, log_path):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "one.a3m").write_text(">one\nACDE\n>hit\nAAAA\n", encoding="utf-8")
        (destination / "one_data.json").write_text("{}", encoding="utf-8")

    def search(records, compact_db, output_dir, config, log_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "two.a3m").write_bytes(b"\xff")

    monkeypatch.setattr("cluster_msa.accelerated.run_full_database_search", full)
    monkeypatch.setattr("cluster_msa.accelerated.build_compact_database", lambda *args: tmp_path / "db")
    monkeypatch.setattr("cluster_msa.accelerated.search_compact_database", search)

    with pytest.raises(OutputValidationError, match="AlphaFold3 JSON"):
        run_accelerated(make_config(tmp_path, af3_json=True), RECORDS[:2])
