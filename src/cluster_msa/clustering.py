import re
from pathlib import Path
from typing import Sequence

from cluster_msa.errors import InputValidationError, OutputValidationError
from cluster_msa.models import ClusterResult, SequenceRecord
from cluster_msa.tools import run_command


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def write_fasta(records: Sequence[SequenceRecord], path: Path) -> None:
    """Write validated records as an ordered, unwrapped FASTA file."""
    _validate_records(records)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(f">{record.id}\n{record.sequence}\n")
    except OSError as error:
        raise OutputValidationError(f"cannot write FASTA file: {path}: {error}") from error


def parse_clusters(
    cluster_tsv: Path, records: Sequence[SequenceRecord]
) -> ClusterResult:
    """Validate mmseqs cluster membership and restore original input order."""
    records_by_id = _validate_records(records)
    if cluster_tsv.is_symlink():
        raise OutputValidationError(f"cluster TSV is not a regular file: {cluster_tsv}")
    try:
        if not cluster_tsv.is_file():
            raise OutputValidationError(f"cluster TSV is not a regular file: {cluster_tsv}")
        lines = cluster_tsv.read_text(encoding="utf-8").splitlines()
    except OutputValidationError:
        raise
    except (OSError, UnicodeError) as error:
        raise OutputValidationError(f"cannot read cluster TSV: {cluster_tsv}: {error}") from error

    membership: dict[str, str] = {}
    representatives: set[str] = set()
    row_count = 0
    for row_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row_count += 1
        fields = line.split("\t")
        if len(fields) != 2:
            raise OutputValidationError(
                f"invalid cluster TSV row {row_number}: expected exactly two fields"
            )
        representative_id, member_id = fields
        if representative_id not in records_by_id:
            raise OutputValidationError(
                f"invalid cluster TSV row {row_number}: unknown representative ID"
            )
        if member_id not in records_by_id:
            raise OutputValidationError(
                f"invalid cluster TSV row {row_number}: unknown member ID"
            )
        if member_id in membership:
            raise OutputValidationError(
                f"invalid cluster TSV row {row_number}: duplicate member ID {member_id!r}"
            )
        representatives.add(representative_id)
        membership[member_id] = representative_id

    if not row_count:
        raise OutputValidationError(f"cluster TSV is empty: {cluster_tsv}")

    missing = records_by_id.keys() - membership.keys()
    if missing:
        raise OutputValidationError(f"cluster TSV has missing input IDs: {', '.join(sorted(missing))}")
    for representative_id in representatives:
        if membership.get(representative_id) != representative_id:
            raise OutputValidationError(
                f"cluster representative {representative_id!r} must map to itself; "
                "every representative maps to itself"
            )

    ordered_representatives = tuple(
        record for record in records if record.id in representatives
    )
    ordered_nonrepresentatives = tuple(
        (record, membership[record.id])
        for record in records
        if record.id not in representatives
    )
    return ClusterResult(ordered_representatives, ordered_nonrepresentatives)


def cluster_sequences(
    records: Sequence[SequenceRecord],
    *,
    mmseqs: Path,
    work_dir: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cluster_mode: int,
    threads: int,
    log_path: Path,
) -> ClusterResult:
    """Cluster records with mmseqs and return deterministic membership."""
    fasta = work_dir / "cluster-input.fasta"
    prefix = work_dir / "cluster"
    write_fasta(records, fasta)
    run_command(
        [
            str(mmseqs),
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(tmp_dir),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "--cluster-mode",
            str(cluster_mode),
            "--threads",
            str(threads),
        ],
        stage="mmseqs easy-cluster",
        log_path=log_path,
    )
    return parse_clusters(prefix.with_name(f"{prefix.name}_cluster.tsv"), records)


def _validate_records(records: Sequence[SequenceRecord]) -> dict[str, SequenceRecord]:
    if not records:
        raise InputValidationError("cannot cluster an empty sequence collection")
    records_by_id: dict[str, SequenceRecord] = {}
    for record in records:
        if not _ID_PATTERN.fullmatch(record.id):
            raise InputValidationError(f"invalid sequence record ID: {record.id!r}")
        if record.id in records_by_id:
            raise InputValidationError(f"duplicate sequence record ID: {record.id!r}")
        records_by_id[record.id] = record
    return records_by_id
