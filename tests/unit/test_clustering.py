from pathlib import Path

import pytest

from cluster_msa.clustering import cluster_sequences, parse_clusters, write_fasta
from cluster_msa.errors import InputValidationError, OutputValidationError
from cluster_msa.models import ClusterResult, SequenceRecord


RECORDS = (
    SequenceRecord("first", "ACDE"),
    SequenceRecord("second", "FGHI"),
    SequenceRecord("third", "KLMN"),
    SequenceRecord("fourth", "PQRS"),
)


def write_tsv(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "clusters_cluster.tsv"
    path.write_text(content, encoding="utf-8")
    return path


def test_write_fasta_preserves_records_headers_and_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "inputs.fasta"

    write_fasta(RECORDS[:2], path)

    assert path.read_bytes() == b">first\nACDE\n>second\nFGHI\n"


@pytest.mark.parametrize("record_id", ["", "../escape", "space id", "-leading"])
def test_write_fasta_rejects_invalid_record_ids(tmp_path: Path, record_id: str) -> None:
    with pytest.raises(InputValidationError, match="invalid.*ID"):
        write_fasta((SequenceRecord(record_id, "ACDE"),), tmp_path / "inputs.fasta")


def test_write_fasta_rejects_empty_records(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="empty"):
        write_fasta((), tmp_path / "inputs.fasta")


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ((SequenceRecord("bad", ""),), "empty"),
        ((SequenceRecord("bad", "AC1E"),), "residue"),
        ((SequenceRecord("bad", "AC\n>injected\nAAAA"),), "whitespace"),
        ((SequenceRecord("bad", "AC DE"),), "whitespace"),
        ((SequenceRecord("bad", "AC:DE"),), "colon"),
        ((SequenceRecord("bad", "ACÉE"),), "residue"),
        (
            (SequenceRecord("duplicate", "ACDE"), SequenceRecord("duplicate", "FGHI")),
            "duplicate.*ID",
        ),
    ],
)
def test_write_fasta_validates_all_records_before_writing(
    tmp_path: Path, records: tuple[SequenceRecord, ...], message: str
) -> None:
    path = tmp_path / "inputs.fasta"

    with pytest.raises(InputValidationError, match=message):
        write_fasta((SequenceRecord("valid", "ACDE"), *records), path)

    assert not path.exists()


def test_write_fasta_wraps_filesystem_errors(tmp_path: Path) -> None:
    parent = tmp_path / "not-a-directory"
    parent.write_text("occupied", encoding="utf-8")

    with pytest.raises(OutputValidationError, match="cannot write FASTA"):
        write_fasta(RECORDS, parent / "inputs.fasta")


def test_parse_clusters_orders_results_by_original_input(tmp_path: Path) -> None:
    path = write_tsv(
        tmp_path,
        "third\tfourth\nfirst\tsecond\nthird\tthird\nfirst\tfirst\n",
    )

    assert parse_clusters(path, RECORDS) == ClusterResult(
        representatives=(RECORDS[0], RECORDS[2]),
        nonrepresentatives=((RECORDS[1], "first"), (RECORDS[3], "third")),
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("first\tfirst\textra\nsecond\tsecond\nthird\tthird\nfourth\tfourth\n", "two fields"),
        ("first\tfirst\nfirst\tsecond\nfirst\tsecond\nthird\tthird\nfourth\tfourth\n", "duplicate"),
        ("first\tfirst\nfirst\tunknown\nsecond\tsecond\nthird\tthird\nfourth\tfourth\n", "unknown"),
        ("unknown\tunknown\nfirst\tfirst\nsecond\tsecond\nthird\tthird\nfourth\tfourth\n", "unknown"),
        ("first\tfirst\nfirst\tsecond\nthird\tthird\n", "missing"),
        ("first\tsecond\nthird\tfirst\nthird\tthird\nfourth\tfourth\n", "maps to itself"),
        ("first\tfirst\nsecond\tfourth\nthird\tsecond\nthird\tthird\n", "maps to itself"),
    ],
)
def test_parse_clusters_rejects_invalid_cluster_membership(
    tmp_path: Path, content: str, message: str
) -> None:
    path = write_tsv(tmp_path, content)

    with pytest.raises(OutputValidationError, match=message):
        parse_clusters(path, RECORDS)


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "empty"])
def test_parse_clusters_rejects_invalid_tsv_paths(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "clusters_cluster.tsv"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.tsv"
        target.write_text("first\tfirst\n", encoding="utf-8")
        path.symlink_to(target)
    elif kind == "empty":
        path.write_text("", encoding="utf-8")

    with pytest.raises(OutputValidationError):
        parse_clusters(path, RECORDS)


def test_parse_clusters_rejects_empty_records(tmp_path: Path) -> None:
    path = write_tsv(tmp_path, "first\tfirst\n")

    with pytest.raises(InputValidationError, match="empty"):
        parse_clusters(path, ())


def test_cluster_sequences_writes_fasta_runs_exact_command_and_parses_output(
    tmp_path: Path, monkeypatch
) -> None:
    mmseqs = tmp_path / "mmseqs"
    work_dir = tmp_path / "work"
    tmp_dir = tmp_path / "tmp"
    log_path = tmp_path / "logs" / "cluster.log"
    captured = {}

    def fake_run_command(command, *, stage, log_path):
        captured.update(command=command, stage=stage, log_path=log_path)
        prefix = Path(command[3])
        prefix.with_name(f"{prefix.name}_cluster.tsv").write_text(
            "first\tsecond\nfirst\tfirst\nthird\tthird\nthird\tfourth\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("cluster_msa.clustering.run_command", fake_run_command)

    result = cluster_sequences(
        RECORDS,
        mmseqs=mmseqs,
        work_dir=work_dir,
        tmp_dir=tmp_dir,
        min_seq_id=0.7,
        coverage=0.8,
        cluster_mode=2,
        threads=13,
        log_path=log_path,
    )

    fasta = work_dir / "cluster-input.fasta"
    prefix = work_dir / "cluster"
    assert captured == {
        "command": [
            str(mmseqs),
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(tmp_dir),
            "--min-seq-id",
            "0.7",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--cluster-mode",
            "2",
            "--threads",
            "13",
        ],
        "stage": "mmseqs easy-cluster",
        "log_path": log_path,
    }
    assert fasta.read_text(encoding="utf-8") == (
        ">first\nACDE\n>second\nFGHI\n>third\nKLMN\n>fourth\nPQRS\n"
    )
    assert result == ClusterResult(
        representatives=(RECORDS[0], RECORDS[2]),
        nonrepresentatives=((RECORDS[1], "first"), (RECORDS[3], "third")),
    )


def test_cluster_sequences_rejects_empty_records_before_running(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("cluster_msa.clustering.run_command", lambda *args, **kwargs: pytest.fail())

    with pytest.raises(InputValidationError, match="empty"):
        cluster_sequences(
            (),
            mmseqs=tmp_path / "mmseqs",
            work_dir=tmp_path / "work",
            tmp_dir=tmp_path / "tmp",
            min_seq_id=0.7,
            coverage=0.8,
            cluster_mode=0,
            threads=1,
            log_path=tmp_path / "run.log",
        )


@pytest.mark.parametrize(
    "records",
    [
        (SequenceRecord("bad", ""),),
        (SequenceRecord("bad", "AC1E"),),
        (SequenceRecord("bad", "AC\n>injected\nAAAA"),),
        (SequenceRecord("bad", "AC DE"),),
        (SequenceRecord("bad", "AC:DE"),),
        (SequenceRecord("bad", "ACÉE"),),
        (SequenceRecord("duplicate", "ACDE"), SequenceRecord("duplicate", "FGHI")),
    ],
)
def test_cluster_sequences_rejects_invalid_records_before_writing_or_running(
    tmp_path: Path,
    monkeypatch,
    records: tuple[SequenceRecord, ...],
) -> None:
    invoked = False

    def fake_run_command(*args, **kwargs):
        nonlocal invoked
        invoked = True

    monkeypatch.setattr("cluster_msa.clustering.run_command", fake_run_command)
    work_dir = tmp_path / "work"

    with pytest.raises(InputValidationError):
        cluster_sequences(
            (SequenceRecord("valid", "ACDE"), *records),
            mmseqs=tmp_path / "mmseqs",
            work_dir=work_dir,
            tmp_dir=tmp_path / "tmp",
            min_seq_id=0.7,
            coverage=0.8,
            cluster_mode=0,
            threads=1,
            log_path=tmp_path / "run.log",
        )

    assert not invoked
    assert not (work_dir / "cluster-input.fasta").exists()
