import shutil
import stat
import tempfile
from pathlib import Path
from typing import Sequence

from cluster_msa.clustering import write_fasta
from cluster_msa.errors import ConfigurationError, InputValidationError, OutputValidationError
from cluster_msa.input import normalize_sequence_record
from cluster_msa.models import RunConfig, SequenceRecord
from cluster_msa.output import publish_outputs, validate_outputs
from cluster_msa.tools import run_command


_PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def parse_a3m_hits(path: Path) -> tuple[str, ...]:
    """Extract normalized, unique non-query sequences from one A3M file."""
    _require_regular_nonempty(path, InputValidationError, "A3M input")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise InputValidationError(f"cannot read A3M input: {path}: {error}") from error

    records: list[str] = []
    sequence_parts: list[str] | None = None
    for line_number, line in enumerate(lines, start=1):
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith(">"):
            if not line[1:].strip():
                raise InputValidationError(f"{path}: line {line_number}: empty A3M header")
            if sequence_parts is not None:
                records.append(_normalize_a3m_sequence(sequence_parts, path))
            sequence_parts = []
        elif sequence_parts is None:
            raise InputValidationError(f"{path}: line {line_number}: sequence before A3M header")
        else:
            sequence_parts.append(line)

    if sequence_parts is None:
        raise InputValidationError(f"{path}: A3M contains no records")
    records.append(_normalize_a3m_sequence(sequence_parts, path))

    return tuple(dict.fromkeys(records[1:]))


def build_compact_database(
    rep_dir: Path, work_dir: Path, config: RunConfig, log_path: Path
) -> Path:
    """Build and index a deterministic mmseqs database from representative hits."""
    files = _representative_a3ms(rep_dir)
    hits = tuple(dict.fromkeys(hit for path in files for hit in parse_a3m_hits(path)))
    if not hits:
        raise InputValidationError(f"representative A3Ms contain no hits: {rep_dir}")
    mmseqs = _require_mmseqs(config)
    fasta = work_dir / "hits_dedup.fasta"
    compact_db = work_dir / "compactDB"
    _write_hits_fasta(hits, fasta)

    run_command(
        [str(mmseqs), "createdb", str(fasta), str(compact_db), "--dbtype", "1"],
        stage="mmseqs createdb compact database",
        log_path=log_path,
    )
    run_command(
        [
            str(mmseqs),
            "createindex",
            str(compact_db),
            str(config.tmp_dir),
            "--remove-tmp-files",
            "1",
        ],
        stage="mmseqs createindex compact database",
        log_path=log_path,
    )
    artifacts = (
        compact_db,
        compact_db.with_suffix(".dbtype"),
        compact_db.with_suffix(".index"),
        compact_db.with_suffix(".idx"),
        compact_db.with_suffix(".idx.dbtype"),
        compact_db.with_suffix(".idx.index"),
    )
    for artifact in artifacts:
        _require_regular_nonempty(artifact, OutputValidationError, "compact database artifact")
    return compact_db


def split_combined_msa(
    combined: Path, records: Sequence[SequenceRecord], output_dir: Path
) -> None:
    """Validate and split concatenated mmseqs A3M query blocks transactionally."""
    normalized_records = _validated_records(records)
    _require_regular_nonempty(combined, OutputValidationError, "combined MSA")
    try:
        content = combined.read_bytes()
    except (OSError, UnicodeError) as error:
        raise OutputValidationError(f"cannot read combined MSA: {combined}: {error}") from error
    if not content.strip():
        raise OutputValidationError(f"combined MSA is empty: {combined}")

    expected_ids = {record.id for record in normalized_records}
    if b"\x00" in content:
        blocks = _parse_nul_entries(content, expected_ids)
    else:
        blocks = _parse_text_blocks(content, expected_ids)

    missing = expected_ids - blocks.keys()
    if missing:
        raise OutputValidationError(f"combined MSA has missing queries: {', '.join(sorted(missing))}")
    rendered: dict[str, bytes] = {}
    for record in normalized_records:
        block = blocks[record.id]
        block_lines = _decode_block_lines(block, record.id)
        if sum(line.startswith(">") for line in block_lines) < 2:
            raise OutputValidationError(f"query-only MSA is not allowed: {record.id}")
        _validate_a3m_block(block_lines, record.id)
        rendered[record.id] = block

    staging = _create_split_staging(output_dir)
    preserve_staging = False
    try:
        for record in normalized_records:
            (staging / f"{record.id}.a3m").write_bytes(rendered[record.id])
        validate_outputs(staging, normalized_records, af3_json=False)
        publish_outputs(
            staging,
            output_dir,
            normalized_records,
            af3_json=False,
            overwrite=True,
        )
    except OutputValidationError as error:
        preserve_staging = "backup preserved at" in str(error)
        raise
    except OSError as error:
        raise OutputValidationError(f"cannot stage split MSA outputs: {staging}: {error}") from error
    finally:
        if not preserve_staging:
            shutil.rmtree(staging, ignore_errors=True)


def _parse_nul_entries(content: bytes, expected_ids: set[str]) -> dict[str, bytes]:
    entries = content.split(b"\x00")
    if entries[-1] == b"":
        entries.pop()
    parsed: dict[str, bytes] = {}
    for entry_number, entry in enumerate(entries, start=1):
        if not entry.strip():
            raise OutputValidationError(f"empty combined MSA entry: {entry_number}")
        lines = _decode_block_lines(entry, f"entry {entry_number}")
        first_header = next(
            (line for line in lines if line.startswith(">")),
            None,
        )
        if first_header is None or any(
            line.strip() and not line.startswith(("#", ">"))
            for line in lines[: lines.index(first_header)]
        ):
            raise OutputValidationError(
                f"combined MSA entry {entry_number} has no valid first header"
            )
        header = first_header[1:].strip()
        if not header:
            raise OutputValidationError(f"combined MSA entry {entry_number} has an empty header")
        query_id = header.split(maxsplit=1)[0]
        if query_id not in expected_ids:
            raise OutputValidationError(f"combined MSA entry has unknown query: {query_id}")
        if query_id in parsed:
            raise OutputValidationError(f"duplicate query block: {query_id}")
        parsed[query_id] = entry
    return parsed


def _parse_text_blocks(content: bytes, expected_ids: set[str]) -> dict[str, bytes]:
    lines = content.splitlines(keepends=True)
    decoded_lines = _decode_block_lines(content, "text output")
    blocks: dict[str, list[bytes]] = {}
    current_id: str | None = None
    for line_number, (line, decoded_line) in enumerate(zip(lines, decoded_lines), start=1):
        if decoded_line.startswith(">"):
            header = decoded_line[1:].strip()
            header_id = header.split(maxsplit=1)[0] if header else ""
            if header_id in expected_ids:
                if header_id in blocks:
                    raise OutputValidationError(f"duplicate query block: {header_id}")
                current_id = header_id
                blocks[current_id] = [line]
                continue
        if current_id is None:
            if decoded_line.strip():
                raise OutputValidationError(
                    f"combined MSA line {line_number} appears before a query header"
                )
            continue
        blocks[current_id].append(line)
    return {query_id: b"".join(block) for query_id, block in blocks.items()}


def _decode_block_lines(block: bytes, label: str) -> list[str]:
    try:
        return [line.decode("utf-8") for line in block.splitlines(keepends=True)]
    except UnicodeError as error:
        raise OutputValidationError(f"invalid UTF-8 in combined MSA {label}") from error


def search_compact_database(
    records: Sequence[SequenceRecord],
    compact_db: Path,
    output_dir: Path,
    config: RunConfig,
    log_path: Path,
) -> None:
    """Search all records against one compact database and emit per-query A3Ms."""
    mmseqs = _require_mmseqs(config)
    for artifact in (compact_db, compact_db.with_suffix(".dbtype")):
        _require_regular_nonempty(artifact, InputValidationError, "compact database artifact")
    work_dir = config.work_dir / "compact-search"
    query_fasta = work_dir / "queries.fasta"
    query_db = work_dir / "queryDB"
    result_db = work_dir / "resultDB"
    combined = work_dir / "combined.a3m"
    write_fasta(records, query_fasta)

    commands = (
        (
            [str(mmseqs), "createdb", str(query_fasta), str(query_db), "--dbtype", "1"],
            "mmseqs createdb compact queries",
        ),
        (
            [
                str(mmseqs),
                "search",
                str(query_db),
                str(compact_db),
                str(result_db),
                str(config.tmp_dir),
                "--threads",
                str(config.threads),
                "--db-load-mode",
                "2",
            ],
            "mmseqs search compact database",
        ),
        (
            [
                str(mmseqs),
                "result2msa",
                str(query_db),
                str(compact_db),
                str(result_db),
                str(combined),
                "--msa-format-mode",
                "2",
            ],
            "mmseqs result2msa compact database",
        ),
    )
    for command, stage in commands:
        run_command(command, stage=stage, log_path=log_path)
    split_combined_msa(combined, records, output_dir)


def _normalize_a3m_sequence(parts: list[str], path: Path) -> str:
    if not parts:
        raise InputValidationError(f"{path}: empty A3M record")
    raw = "".join(parts).replace("\x00", "")
    sequence = "".join(
        character for character in raw if character != "-" and character not in "abcdefghijklmnopqrstuvwxyz"
    )
    if not sequence:
        raise InputValidationError(f"{path}: empty A3M record")
    if not sequence.isascii() or not set(sequence) <= _PROTEIN_ALPHABET:
        raise InputValidationError(f"{path}: A3M record contains an invalid residue")
    return sequence


def _validate_a3m_block(lines: Sequence[str], record_id: str) -> None:
    sequence_parts: list[str] | None = None
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith(">"):
            if not line[1:].strip():
                raise OutputValidationError(f"empty A3M header in query block: {record_id}")
            if sequence_parts is not None:
                _validate_output_sequence(sequence_parts, record_id)
            sequence_parts = []
        elif sequence_parts is None:
            raise OutputValidationError(f"sequence before A3M header in query block: {record_id}")
        else:
            sequence_parts.append(line.rstrip("\r\n"))
    if sequence_parts is None:
        raise OutputValidationError(f"empty query block: {record_id}")
    _validate_output_sequence(sequence_parts, record_id)


def _validate_output_sequence(parts: list[str], record_id: str) -> None:
    try:
        _normalize_a3m_sequence(parts, Path(record_id))
    except InputValidationError as error:
        raise OutputValidationError(f"invalid or empty A3M record in query block: {record_id}") from error


def _representative_a3ms(rep_dir: Path) -> tuple[Path, ...]:
    try:
        if rep_dir.is_symlink() or not rep_dir.is_dir():
            raise InputValidationError(f"representative directory is not a directory: {rep_dir}")
        files = tuple(sorted(rep_dir.glob("*.a3m"), key=lambda path: path.name))
    except InputValidationError:
        raise
    except OSError as error:
        raise InputValidationError(f"cannot inspect representative directory: {rep_dir}: {error}") from error
    if not files:
        raise InputValidationError(f"representative directory contains no A3M files: {rep_dir}")
    for path in files:
        _require_regular_nonempty(path, InputValidationError, "representative A3M")
    return files


def _write_hits_fasta(hits: Sequence[str], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for index, hit in enumerate(hits):
                output.write(f">hit_{index}\n{hit}\n")
    except OSError as error:
        raise OutputValidationError(f"cannot write compact database FASTA: {path}: {error}") from error


def _validated_records(records: Sequence[SequenceRecord]) -> tuple[SequenceRecord, ...]:
    if not records:
        raise InputValidationError("cannot split an empty sequence collection")
    normalized: list[SequenceRecord] = []
    seen_ids: set[str] = set()
    for record in records:
        normalized_record = normalize_sequence_record(
            record, error_prefix="invalid sequence record: "
        )
        if normalized_record.id in seen_ids:
            raise InputValidationError(f"duplicate sequence record ID: {record.id!r}")
        normalized.append(normalized_record)
        seen_ids.add(normalized_record.id)
    return tuple(normalized)


def _require_mmseqs(config: RunConfig) -> Path:
    if config.toolchain.mmseqs is None:
        raise ConfigurationError("mmseqs is required for compact database operations")
    return config.toolchain.mmseqs


def _require_regular_nonempty(path: Path, error_type: type[Exception], label: str) -> None:
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.stat().st_size == 0:
            raise error_type(f"{label} is missing, nonregular, or empty: {path}")
    except FileNotFoundError as error:
        raise error_type(f"{label} is missing, nonregular, or empty: {path}") from error
    except OSError as error:
        raise error_type(f"cannot inspect {label}: {path}: {error}") from error


def _create_split_staging(output_dir: Path) -> Path:
    parent = output_dir.parent
    try:
        if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
            raise OutputValidationError(f"output directory is not a directory: {output_dir}")
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=".cluster-msa-split-", dir=parent))
    except OutputValidationError:
        raise
    except OSError as error:
        raise OutputValidationError(f"cannot create split MSA staging directory: {parent}: {error}") from error
