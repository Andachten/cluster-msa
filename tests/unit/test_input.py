from pathlib import Path

import pytest

from cluster_msa.errors import InputValidationError
from cluster_msa.input import load_sequences
from cluster_msa.models import SequenceRecord


def write_csv(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "inputs.csv"
    path.write_text(content, encoding="utf-8")
    return path


def assert_invalid(path: Path, expected: str) -> None:
    with pytest.raises(InputValidationError, match=expected):
        load_sequences(path)


def test_load_sequences_preserves_order_strips_cells_and_uppercases(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "id,sequence\n first-1 , acde \nsecond_2,BXZJUO\n")

    assert load_sequences(path) == (
        SequenceRecord(id="first-1", sequence="ACDE"),
        SequenceRecord(id="second_2", sequence="BXZJUO"),
    )


@pytest.mark.parametrize(
    "header",
    ["sequence,id", "id", "id,sequence,description"],
)
def test_load_sequences_rejects_wrong_or_additional_header(
    tmp_path: Path, header: str
) -> None:
    path = write_csv(tmp_path, f"{header}\nexample,ACDE\n")

    assert_invalid(path, "header")


def test_load_sequences_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "id,sequence\nexample,ACDE\nexample,FGHI\n")

    assert_invalid(path, r"inputs\.csv.*row 3.*duplicate.*example")


@pytest.mark.parametrize("unsafe_id", ["../escape", "/absolute", "space id", "-leading"])
def test_load_sequences_rejects_unsafe_ids(tmp_path: Path, unsafe_id: str) -> None:
    path = write_csv(tmp_path, f"id,sequence\n{unsafe_id},ACDE\n")

    assert_invalid(path, r"inputs\.csv.*row 2.*ID")


def test_load_sequences_rejects_multichain_colon(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "id,sequence\nexample,ACDE:FGHI\n")

    assert_invalid(path, r"inputs\.csv.*row 2.*colon")


def test_load_sequences_rejects_invalid_residue_without_echoing_sequence(tmp_path: Path) -> None:
    invalid_sequence = "ACDEF1HIK"
    path = write_csv(tmp_path, f"id,sequence\nexample,{invalid_sequence}\n")

    with pytest.raises(InputValidationError, match=r"inputs\.csv.*row 2.*residue") as error:
        load_sequences(path)

    assert invalid_sequence not in str(error.value)


def test_load_sequences_rejects_empty_file(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "")

    assert_invalid(path, r"inputs\.csv.*empty")


@pytest.mark.parametrize("row", [",ACDE", "example,", ","])
def test_load_sequences_rejects_blank_cells(tmp_path: Path, row: str) -> None:
    path = write_csv(tmp_path, f"id,sequence\n{row}\n")

    assert_invalid(path, r"inputs\.csv.*row 2.*empty")


def test_load_sequences_rejects_blank_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "id,sequence\nexample,ACDE\n\n")

    assert_invalid(path, r"inputs\.csv.*row 3.*empty")


@pytest.mark.parametrize("sequence", ["AC DE", "AC\tDE"])
def test_load_sequences_rejects_internal_sequence_whitespace(
    tmp_path: Path, sequence: str
) -> None:
    path = write_csv(tmp_path, f"id,sequence\nexample,{sequence}\n")

    assert_invalid(path, r"inputs\.csv.*row 2.*whitespace")


def test_load_sequences_reports_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"

    assert_invalid(path, r"missing\.csv.*does not exist")


def test_load_sequences_reports_unreadable_path(tmp_path: Path) -> None:
    assert_invalid(tmp_path, "cannot read")


def test_load_sequences_rejects_unterminated_quoted_value(tmp_path: Path) -> None:
    path = write_csv(tmp_path, 'id,sequence\nexample,"ACDE\n')

    with pytest.raises(InputValidationError, match=r"inputs\.csv.*read") as error:
        load_sequences(path)

    assert "ACDE" not in str(error.value)
