from pathlib import Path

import pytest

from cluster_msa.compact_db import (
    build_compact_database,
    parse_a3m_hits,
    search_compact_database,
    split_combined_msa,
)
from cluster_msa.errors import ExternalToolError, InputValidationError, OutputValidationError
from cluster_msa.models import RunConfig, SequenceRecord, Toolchain


RECORDS = (SequenceRecord("one", "ACDE"), SequenceRecord("two", "FGHI"))


def make_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        mode="accelerated",
        input_path=tmp_path / "inputs.csv",
        output_dir=tmp_path / "output",
        db_path=tmp_path / "database",
        toolchain=Toolchain(tmp_path / "colabfold_search", tmp_path / "mmseqs"),
        threads=7,
        gpu=False,
        gpus="",
        af3_json=False,
        tmp_dir=tmp_path / "tmp",
        work_dir=tmp_path / "work",
        keep_work=False,
        overwrite=False,
        verbose=False,
    )


def test_parse_a3m_hits_excludes_query_normalizes_multiline_hits_and_deduplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rep.a3m"
    path.write_bytes(
        b"# metadata\n>query\nACd-E\n>first\nACd-\x00E\nFG\n>duplicate\nACEFG\n>second\nBXz-JUO\n"
    )

    assert parse_a3m_hits(path) == ("ACEFG", "BXJUO")


@pytest.mark.parametrize(
    "content",
    [
        "ACDE\n",
        ">query\nACDE\n>empty\n",
        ">query\nACDE\n>bad\nAC1E\n",
        ">query\nACDE\n>bad\nacde\n",
        ">query\nACDE\n>bad\nACéDE\n",
    ],
)
def test_parse_a3m_hits_rejects_malformed_or_empty_records(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "bad.a3m"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(InputValidationError):
        parse_a3m_hits(path)


def test_build_compact_database_deduplicates_hits_in_sorted_file_order_and_indexes(
    tmp_path: Path, monkeypatch
) -> None:
    rep_dir = tmp_path / "representatives"
    rep_dir.mkdir()
    (rep_dir / "z.a3m").write_text(
        ">query-z\nAAAA\n>z-first\nFGHI\n>duplicate\nACDE\n", encoding="utf-8"
    )
    (rep_dir / "a.a3m").write_text(
        ">query-a\nBBBB\n>a-first\nACd-DE\n>a-second\nKLMN\n", encoding="utf-8"
    )
    config = make_config(tmp_path)
    work_dir = tmp_path / "compact-work"
    log_path = tmp_path / "run.log"
    calls = []

    def fake_run_command(command, *, stage, log_path):
        calls.append((command, stage, log_path))
        prefix = Path(command[3] if command[1] == "createdb" else command[2])
        if command[1] == "createdb":
            prefix.write_text("db", encoding="utf-8")
            prefix.with_suffix(".dbtype").write_text("type", encoding="utf-8")
            prefix.with_suffix(".index").write_text("index", encoding="utf-8")
        else:
            prefix.with_suffix(".idx").write_text("index-db", encoding="utf-8")
            prefix.with_suffix(".idx.dbtype").write_text("index-type", encoding="utf-8")
            prefix.with_suffix(".idx.index").write_text("index-index", encoding="utf-8")

    monkeypatch.setattr("cluster_msa.compact_db.run_command", fake_run_command)

    result = build_compact_database(rep_dir, work_dir, config, log_path)

    fasta = work_dir / "hits_dedup.fasta"
    compact_db = work_dir / "compactDB"
    assert result == compact_db
    assert fasta.read_bytes() == b">hit_0\nACDE\n>hit_1\nKLMN\n>hit_2\nFGHI\n"
    assert calls == [
        (
            [str(config.toolchain.mmseqs), "createdb", str(fasta), str(compact_db), "--dbtype", "1"],
            "mmseqs createdb compact database",
            log_path,
        ),
        (
            [
                str(config.toolchain.mmseqs),
                "createindex",
                str(compact_db),
                str(work_dir / "tmp"),
                "--remove-tmp-files",
                "1",
            ],
            "mmseqs createindex compact database",
            log_path,
        ),
    ]


def test_build_compact_database_rejects_no_hits_before_running(tmp_path: Path, monkeypatch) -> None:
    rep_dir = tmp_path / "representatives"
    rep_dir.mkdir()
    (rep_dir / "only-query.a3m").write_text(">query\nACDE\n", encoding="utf-8")
    monkeypatch.setattr("cluster_msa.compact_db.run_command", lambda *args, **kwargs: pytest.fail())

    with pytest.raises(InputValidationError, match="no hits"):
        build_compact_database(rep_dir, tmp_path / "work", make_config(tmp_path), tmp_path / "log")


def test_build_compact_database_treats_createindex_failure_as_fatal(
    tmp_path: Path, monkeypatch
) -> None:
    rep_dir = tmp_path / "representatives"
    rep_dir.mkdir()
    (rep_dir / "rep.a3m").write_text(">query\nACDE\n>hit\nFGHI\n", encoding="utf-8")

    def fake_run_command(command, **kwargs):
        if command[1] == "createindex":
            raise ExternalToolError("index failed")
        prefix = Path(command[3])
        prefix.write_text("db", encoding="utf-8")
        prefix.with_suffix(".dbtype").write_text("type", encoding="utf-8")

    monkeypatch.setattr("cluster_msa.compact_db.run_command", fake_run_command)

    with pytest.raises(ExternalToolError, match="index failed"):
        build_compact_database(rep_dir, tmp_path / "work", make_config(tmp_path), tmp_path / "log")


def test_build_compact_database_rejects_missing_index_artifacts(tmp_path: Path, monkeypatch) -> None:
    rep_dir = tmp_path / "representatives"
    rep_dir.mkdir()
    (rep_dir / "rep.a3m").write_text(">query\nACDE\n>hit\nFGHI\n", encoding="utf-8")

    def fake_run_command(command, **kwargs):
        prefix = Path(command[3] if command[1] == "createdb" else command[2])
        if command[1] == "createdb":
            prefix.write_text("db", encoding="utf-8")
            prefix.with_suffix(".dbtype").write_text("type", encoding="utf-8")
            prefix.with_suffix(".index").write_text("createdb-index", encoding="utf-8")

    monkeypatch.setattr("cluster_msa.compact_db.run_command", fake_run_command)

    with pytest.raises(OutputValidationError, match="artifact"):
        build_compact_database(rep_dir, tmp_path / "work", make_config(tmp_path), tmp_path / "log")


def test_search_compact_database_runs_one_batched_search_and_splits_results(
    tmp_path: Path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    compact_db = tmp_path / "compactDB"
    compact_db.write_text("db", encoding="utf-8")
    compact_db.with_suffix(".dbtype").write_text("type", encoding="utf-8")
    output_dir = tmp_path / "output"
    calls = []

    def fake_run_command(command, *, stage, log_path):
        calls.append((command, stage, log_path))
        if command[1] == "result2msa":
            Path(command[5]).write_text(
                ">two\nFGHI\n>hit-two\nFG-HI\n>one\nACDE\n>hit-one\nACdDE\n",
                encoding="utf-8",
            )

    monkeypatch.setattr("cluster_msa.compact_db.run_command", fake_run_command)

    search_compact_database(RECORDS, compact_db, output_dir, config, tmp_path / "run.log")

    work = config.work_dir / "compact-search"
    query_fasta = work / "queries.fasta"
    query_db = work / "queryDB"
    result_db = work / "resultDB"
    combined = work / "combined.a3m"
    assert query_fasta.read_bytes() == b">one\nACDE\n>two\nFGHI\n"
    assert calls == [
        (
            [str(config.toolchain.mmseqs), "createdb", str(query_fasta), str(query_db), "--dbtype", "1"],
            "mmseqs createdb compact queries",
            tmp_path / "run.log",
        ),
        (
            [
                str(config.toolchain.mmseqs),
                "search",
                str(query_db),
                str(compact_db),
                str(result_db),
                str(work / "tmp"),
                "--threads",
                "7",
                "--db-load-mode",
                "2",
            ],
            "mmseqs search compact database",
            tmp_path / "run.log",
        ),
        (
            [
                str(config.toolchain.mmseqs),
                "result2msa",
                str(query_db),
                str(compact_db),
                str(result_db),
                str(combined),
                "--msa-format-mode",
                "2",
            ],
            "mmseqs result2msa compact database",
            tmp_path / "run.log",
        ),
    ]
    assert (output_dir / "one.a3m").read_text(encoding="utf-8") == (
        ">one\nACDE\n>hit-one\nACdDE\n"
    )
    assert (output_dir / "two.a3m").read_text(encoding="utf-8") == (
        ">two\nFGHI\n>hit-two\nFG-HI\n"
    )
    assert sorted(path.name for path in output_dir.iterdir()) == ["one.a3m", "two.a3m"]


def test_split_combined_msa_preserves_complete_lines_and_crlf(tmp_path: Path) -> None:
    combined = tmp_path / "combined.a3m"
    combined.write_bytes(
        b">one description\r\nACd-E\r\n# comment\r\n>hit-one extra\r\nFG-HI\r\n"
        b">two\r\nFGHI\r\n>hit-two\r\nKLMN\r\n"
    )

    split_combined_msa(combined, RECORDS, tmp_path / "output")

    assert (tmp_path / "output" / "one.a3m").read_bytes() == (
        b">one description\r\nACd-E\r\n# comment\r\n>hit-one extra\r\nFG-HI\r\n"
    )


def test_split_combined_msa_rejects_empty_hit_record_without_outputs(tmp_path: Path) -> None:
    combined = tmp_path / "combined.a3m"
    combined.write_text(
        ">one\nACDE\n>empty-hit\n>two\nFGHI\n>hit-two\nKLMN\n", encoding="utf-8"
    )
    output_dir = tmp_path / "output"

    with pytest.raises(OutputValidationError, match="empty"):
        split_combined_msa(combined, RECORDS, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (">one\nACDE\n>hit\nFGHI\n", "missing"),
        ("", "empty"),
        (">one\nACDE\n>two\nFGHI\n>hit\nKLMN\n", "query-only"),
        (">one\nACDE\n>hit\nFGHI\n>one\nACDE\n>hit2\nKLMN\n>two\nFGHI\n>x\nAAAA\n", "duplicate"),
        ("garbage\n>one\nACDE\n>hit\nFGHI\n>two\nFGHI\n>x\nAAAA\n", "header"),
    ],
)
def test_split_combined_msa_rejects_invalid_results_without_partial_outputs(
    tmp_path: Path, content: str, message: str
) -> None:
    combined = tmp_path / "combined.a3m"
    combined.write_text(content, encoding="utf-8")
    output_dir = tmp_path / "output"

    with pytest.raises(OutputValidationError, match=message):
        split_combined_msa(combined, RECORDS, output_dir)

    assert not output_dir.exists() or not list(output_dir.iterdir())


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_split_combined_msa_rejects_nonregular_input(tmp_path: Path, kind: str) -> None:
    combined = tmp_path / "combined.a3m"
    if kind == "directory":
        combined.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.a3m"
        target.write_text(">one\nACDE\n", encoding="utf-8")
        combined.symlink_to(target)

    with pytest.raises(OutputValidationError):
        split_combined_msa(combined, RECORDS, tmp_path / "output")
