from pathlib import Path

import pytest

from cluster_msa.errors import OutputValidationError
from cluster_msa.models import SequenceRecord
from cluster_msa.output import publish_outputs, staged_output, validate_outputs


RECORDS = (
    SequenceRecord(id="first", sequence="ACDE"),
    SequenceRecord(id="second", sequence="FGHI"),
)


def write_valid_outputs(staging: Path, *, af3_json: bool = False) -> None:
    (staging / "first.a3m").write_text(">query\nACDE\n>hit\nAC-E\n", encoding="utf-8")
    (staging / "second.a3m").write_text(">query\nFGHI\n", encoding="utf-8")
    if af3_json:
        (staging / "first_data.json").write_text('{"name": "first"}\n', encoding="utf-8")
        (staging / "second_data.json").write_text('{"name": "second"}\n', encoding="utf-8")


def test_nonempty_destination_requires_overwrite(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    write_valid_outputs(staging)

    with pytest.raises(OutputValidationError, match="not empty"):
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=False)

    assert not (output / "first.a3m").exists()
    assert (output / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_overwrite_preserves_unknown_files_and_replaces_only_expected_files(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    (output / "unknown.txt").write_text("keep", encoding="utf-8")
    (output / "first.a3m").write_text("old", encoding="utf-8")
    (output / "run_manifest.json").write_text("old manifest", encoding="utf-8")
    (output / "run.log").write_text("old log", encoding="utf-8")
    write_valid_outputs(staging)
    (staging / "run_manifest.json").write_text("new manifest", encoding="utf-8")

    publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    assert (output / "unknown.txt").read_text(encoding="utf-8") == "keep"
    assert (output / "first.a3m").read_text(encoding="utf-8").startswith(">query")
    assert not (output / "second_data.json").exists()
    assert (output / "run_manifest.json").read_text(encoding="utf-8") == "new manifest"
    assert (output / "run.log").read_text(encoding="utf-8") == "old log"


@pytest.mark.parametrize("content", ["", "   \n\t"])
def test_validate_outputs_rejects_empty_or_whitespace_a3m(tmp_path: Path, content: str) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "first.a3m").write_text(content, encoding="utf-8")
    (staging / "second.a3m").write_text(">query\nFGHI\n", encoding="utf-8")

    with pytest.raises(OutputValidationError, match="first.a3m"):
        validate_outputs(staging, RECORDS, af3_json=False)


def test_validate_outputs_rejects_missing_requested_json(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    write_valid_outputs(staging)

    with pytest.raises(OutputValidationError, match="first_data.json"):
        validate_outputs(staging, RECORDS, af3_json=True)


def test_validate_outputs_rejects_invalid_requested_json(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    write_valid_outputs(staging, af3_json=True)
    (staging / "first_data.json").write_text("not json", encoding="utf-8")

    with pytest.raises(OutputValidationError, match="first_data.json"):
        validate_outputs(staging, RECORDS, af3_json=True)


def test_validate_outputs_rejects_expected_directory(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "first.a3m").mkdir()
    (staging / "second.a3m").write_text(">query\nFGHI\n", encoding="utf-8")

    with pytest.raises(OutputValidationError, match="first.a3m"):
        validate_outputs(staging, RECORDS, af3_json=False)


def test_staged_output_does_not_publish_before_exit_and_cleans_success(tmp_path: Path) -> None:
    output = tmp_path / "output"
    work = tmp_path / "work"

    with staged_output(output, work) as staging:
        assert staging.parent == work
        assert not output.exists()
        write_valid_outputs(staging)
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=False)
        assert (output / "first.a3m").exists()
        isolated_staging = staging

    assert not isolated_staging.exists()
    assert (output / "first.a3m").exists()


def test_staged_output_preserves_failure_diagnostics(tmp_path: Path) -> None:
    work = tmp_path / "work"

    with pytest.raises(RuntimeError):
        with staged_output(tmp_path / "output", work) as staging:
            (staging / "diagnostic.txt").write_text("details", encoding="utf-8")
            failed_staging = staging
            raise RuntimeError("tool failed")

    assert failed_staging.exists()
    assert (failed_staging / "diagnostic.txt").read_text(encoding="utf-8") == "details"


def test_staged_output_allocates_unique_directories_under_work_dir(tmp_path: Path) -> None:
    work = tmp_path / "work"

    with staged_output(tmp_path / "output", work) as first:
        with staged_output(tmp_path / "output", work) as second:
            assert first.parent == work
            assert second.parent == work
            assert first != second
            assert first.exists()
            assert second.exists()


def test_publish_preflights_validation_before_moving_anything(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    (output / "first.a3m").write_text("old", encoding="utf-8")
    (staging / "first.a3m").write_text(">query\nACDE\n", encoding="utf-8")

    with pytest.raises(OutputValidationError, match="second.a3m"):
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    assert (output / "first.a3m").read_text(encoding="utf-8") == "old"
