import csv
import re
from pathlib import Path

from cluster_msa.errors import InputValidationError
from cluster_msa.models import SequenceRecord


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def load_sequences(path: Path) -> tuple[SequenceRecord, ...]:
    """Load and validate an exact id,sequence CSV in source row order."""
    if not path.exists():
        raise InputValidationError(f"{path}: file does not exist")

    records: list[SequenceRecord] = []
    seen_ids: set[str] = set()

    try:
        with path.open("r", encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file)
            try:
                header = next(reader)
            except StopIteration as error:
                raise InputValidationError(f"{path}: empty file") from error

            if header != ["id", "sequence"]:
                raise InputValidationError(
                    f"{path}: header must contain exactly id,sequence"
                )

            for row in reader:
                row_number = reader.line_num
                if len(row) != 2:
                    detail = "empty row" if not row else "expected exactly 2 columns"
                    raise InputValidationError(f"{path}: row {row_number}: {detail}")

                record_id = row[0].strip()
                raw_sequence = row[1].strip()
                if not record_id or not raw_sequence:
                    raise InputValidationError(f"{path}: row {row_number}: empty cell")
                if not _ID_PATTERN.fullmatch(record_id):
                    raise InputValidationError(f"{path}: row {row_number}: invalid ID")
                if record_id in seen_ids:
                    raise InputValidationError(
                        f"{path}: row {row_number}: duplicate ID {record_id!r}"
                    )
                if any(character.isspace() for character in raw_sequence):
                    raise InputValidationError(
                        f"{path}: row {row_number}: internal sequence whitespace"
                    )
                if ":" in raw_sequence:
                    raise InputValidationError(
                        f"{path}: row {row_number}: sequence contains a colon"
                    )

                sequence = raw_sequence.upper()
                if not set(sequence) <= _PROTEIN_ALPHABET:
                    raise InputValidationError(
                        f"{path}: row {row_number}: sequence contains an invalid residue"
                    )

                records.append(SequenceRecord(id=record_id, sequence=sequence))
                seen_ids.add(record_id)
    except InputValidationError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise InputValidationError(f"{path}: cannot read input file: {error}") from error

    if not records:
        raise InputValidationError(f"{path}: empty input has no sequence rows")

    return tuple(records)
