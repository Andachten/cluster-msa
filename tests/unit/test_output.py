import os
import shutil
import tempfile
import threading
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
    (staging / "run_manifest.json").write_text('{"status": "complete"}\n', encoding="utf-8")

    publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    assert (output / "unknown.txt").read_text(encoding="utf-8") == "keep"
    assert (output / "first.a3m").read_text(encoding="utf-8").startswith(">query")
    assert not (output / "second_data.json").exists()
    assert (output / "run_manifest.json").read_text(encoding="utf-8") == (
        '{"status": "complete"}\n'
    )
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


def test_validate_outputs_rejects_a3m_symlink(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    target = tmp_path / "target.a3m"
    target.write_text(">query\nACDE\n", encoding="utf-8")
    (staging / "first.a3m").symlink_to(target)
    (staging / "second.a3m").write_text(">query\nFGHI\n", encoding="utf-8")

    with pytest.raises(OutputValidationError, match="first.a3m"):
        validate_outputs(staging, RECORDS, af3_json=False)


@pytest.mark.parametrize("name", ["run_manifest.json", "run.log"])
def test_validate_outputs_rejects_optional_metadata_symlink(tmp_path: Path, name: str) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    write_valid_outputs(staging)
    target = tmp_path / name
    target.write_text('{"status": "complete"}\n', encoding="utf-8")
    (staging / name).symlink_to(target)

    with pytest.raises(OutputValidationError, match=name):
        validate_outputs(staging, RECORDS, af3_json=False)


def test_validate_outputs_rejects_invalid_manifest_json(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    write_valid_outputs(staging)
    (staging / "run_manifest.json").write_text("not json", encoding="utf-8")

    with pytest.raises(OutputValidationError, match="run_manifest.json"):
        validate_outputs(staging, RECORDS, af3_json=False)


def test_validate_outputs_rejects_duplicate_record_ids(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "first.a3m").write_text(">query\nACDE\n", encoding="utf-8")
    duplicate_records = (RECORDS[0], SequenceRecord(id="first", sequence="FGHI"))

    with pytest.raises(OutputValidationError, match="duplicate.*first"):
        validate_outputs(staging, duplicate_records, af3_json=False)


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


@pytest.mark.parametrize("failure", ["mkdir", "mkdtemp"])
def test_staged_output_wraps_creation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    work = tmp_path / "work"
    if failure == "mkdir":
        work.write_text("not a directory", encoding="utf-8")
    else:
        monkeypatch.setattr(
            "cluster_msa.output.tempfile.mkdtemp",
            lambda **kwargs: (_ for _ in ()).throw(OSError("denied")),
        )

    with pytest.raises(OutputValidationError, match="cannot create output staging"):
        with staged_output(tmp_path / "output", work):
            pytest.fail("staging unexpectedly created")


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


def test_publish_preflights_all_targets_before_replacing_files(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    write_valid_outputs(staging)
    (output / "first.a3m").write_text("old", encoding="utf-8")
    (output / "second.a3m").mkdir()

    with pytest.raises(OutputValidationError, match="second.a3m"):
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    assert (output / "first.a3m").read_text(encoding="utf-8") == "old"
    assert (staging / "first.a3m").exists()
    assert (staging / "second.a3m").exists()


def test_publish_rejects_symlink_target_before_replacing_files(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    write_valid_outputs(staging)
    (output / "first.a3m").write_text("old", encoding="utf-8")
    target = tmp_path / "target.a3m"
    target.write_text("unsafe", encoding="utf-8")
    (output / "second.a3m").symlink_to(target)

    with pytest.raises(OutputValidationError, match="second.a3m"):
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    assert (output / "first.a3m").read_text(encoding="utf-8") == "old"
    assert (staging / "first.a3m").exists()
    assert target.read_text(encoding="utf-8") == "unsafe"


def test_publish_rejects_duplicate_record_ids_before_moving_files(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    (staging / "first.a3m").write_text(">query\nACDE\n", encoding="utf-8")
    (output / "first.a3m").write_text("old", encoding="utf-8")
    duplicate_records = (RECORDS[0], SequenceRecord(id="first", sequence="FGHI"))

    with pytest.raises(OutputValidationError, match="duplicate.*first"):
        publish_outputs(staging, output, duplicate_records, af3_json=False, overwrite=True)

    assert (output / "first.a3m").read_text(encoding="utf-8") == "old"
    assert (staging / "first.a3m").exists()


def test_publish_backup_cleanup_failure_keeps_published_outputs_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    write_valid_outputs(staging)
    (staging / "run.log").write_text("complete\n", encoding="utf-8")
    real_rmtree = __import__("shutil").rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".cluster-msa-publish-"):
            raise OSError("cleanup denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("cluster_msa.output.shutil.rmtree", fail_backup_cleanup)

    publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=False)

    assert (output / "first.a3m").exists()
    assert (output / "second.a3m").exists()
    assert "cleanup warning" in (output / "run.log").read_text().lower()


def test_publish_serializes_threads_without_mixing_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    staging_a = tmp_path / "staging-a"
    staging_b = tmp_path / "staging-b"
    staging_a.mkdir()
    staging_b.mkdir()
    for staging, generation in ((staging_a, "A"), (staging_b, "B")):
        for record in RECORDS:
            (staging / f"{record.id}.a3m").write_text(
                f">query\n{generation}-{record.id}\n", encoding="utf-8"
            )

    real_replace = os.replace
    a_first_entered = threading.Event()
    b_first_entered = threading.Event()
    a_first_finished = threading.Event()

    def force_interleaving(source: Path, destination: Path) -> None:
        if destination == output / "first.a3m" and "publish-" in source.parent.name:
            generation = source.read_text(encoding="utf-8").splitlines()[1][0]
            if generation == "A":
                a_first_entered.set()
                b_first_entered.wait(0.2)
                real_replace(source, destination)
                a_first_finished.set()
                return
            a_first_entered.wait(1)
            b_first_entered.set()
        if destination == output / "second.a3m" and "publish-" in source.parent.name:
            generation = source.read_text(encoding="utf-8").splitlines()[1][0]
            if generation == "B":
                a_first_finished.wait(1)
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", force_interleaving)
    errors = []

    def publish(staging: Path) -> None:
        try:
            publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=publish, args=(staging_a,))
    second = threading.Thread(target=publish, args=(staging_b,))
    first.start()
    assert a_first_entered.wait(1)
    second.start()
    first.join(2)
    second.join(2)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    generations = {
        (output / f"{record.id}.a3m").read_text(encoding="utf-8").splitlines()[1][0]
        for record in RECORDS
    }
    assert generations in ({"A"}, {"B"})


def test_publish_lock_failure_is_typed_and_leaves_output_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    write_valid_outputs(staging)
    (output / "first.a3m").write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        "cluster_msa.output.fcntl.flock",
        lambda *args: (_ for _ in ()).throw(OSError("lock denied")),
    )

    with pytest.raises(OutputValidationError, match="publication lock"):
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    assert (output / "first.a3m").read_text(encoding="utf-8") == "old"
    assert not (output / "second.a3m").exists()


def test_publish_copy_failure_is_typed_and_leaves_output_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    write_valid_outputs(staging)
    (output / "first.a3m").write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        shutil,
        "copy2",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copy denied")),
    )

    with pytest.raises(OutputValidationError, match="stage publication transaction"):
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    assert (output / "first.a3m").read_text(encoding="utf-8") == "old"
    assert not (output / "second.a3m").exists()
    assert not list(tmp_path.glob(".cluster-msa-publish-*"))


def test_publish_copy_cleanup_failure_preserves_original_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    write_valid_outputs(staging)
    monkeypatch.setattr(
        shutil,
        "copy2",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copy denied")),
    )
    monkeypatch.setattr(
        shutil,
        "rmtree",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup denied")),
    )

    with pytest.raises(OutputValidationError) as captured:
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=False)

    assert "copy denied" in str(captured.value)
    assert "cleanup denied" in str(captured.value)
    assert not output.exists()


def test_publish_across_filesystems_preserves_staging_and_outputs(tmp_path: Path) -> None:
    shared_memory = Path("/dev/shm")
    if not shared_memory.is_dir() or shared_memory.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("separate /dev/shm filesystem is unavailable")
    staging = Path(tempfile.mkdtemp(prefix="cluster-msa-test-", dir=shared_memory))
    output = tmp_path / "output"
    try:
        write_valid_outputs(staging)

        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=False)

        assert (output / "first.a3m").read_bytes() == (staging / "first.a3m").read_bytes()
        assert (output / "second.a3m").read_bytes() == (staging / "second.a3m").read_bytes()
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def test_rollback_backup_cleanup_failure_preserves_original_error_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    write_valid_outputs(staging)
    (output / "first.a3m").write_text("old first", encoding="utf-8")
    (output / "second.a3m").write_text("old second", encoding="utf-8")
    real_replace = os.replace
    real_rmtree = shutil.rmtree

    def fail_second_publication(source: Path, destination: Path) -> None:
        if source.name == "second.a3m" and "publish-" in source.parent.name:
            raise OSError("publication failed")
        real_replace(source, destination)

    def fail_backup_cleanup(path: Path, *args, **kwargs) -> None:
        if Path(path).name.startswith("publication-backup-"):
            raise OSError("backup cleanup denied")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_second_publication)
    monkeypatch.setattr(shutil, "rmtree", fail_backup_cleanup)

    with pytest.raises(OutputValidationError) as captured:
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    message = str(captured.value)
    assert "publication failed" in message
    assert "backup cleanup denied" in message
    assert "backup preserved at" in message
    assert (output / "first.a3m").read_text(encoding="utf-8") == "old first"
    assert (output / "second.a3m").read_text(encoding="utf-8") == "old second"


def test_rollback_transaction_cleanup_failure_preserves_original_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    write_valid_outputs(staging)
    (output / "first.a3m").write_text("old first", encoding="utf-8")
    (output / "second.a3m").write_text("old second", encoding="utf-8")
    real_replace = os.replace
    real_rmtree = shutil.rmtree

    def fail_second_publication(source: Path, destination: Path) -> None:
        if source.name == "second.a3m" and source.parent.name.startswith(
            ".cluster-msa-publish-"
        ):
            raise OSError("publication failed")
        real_replace(source, destination)

    def fail_transaction_cleanup(path: Path, *args, **kwargs) -> None:
        if Path(path).name.startswith(".cluster-msa-publish-"):
            raise OSError("transaction cleanup denied")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_second_publication)
    monkeypatch.setattr(shutil, "rmtree", fail_transaction_cleanup)

    with pytest.raises(OutputValidationError) as captured:
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    assert "publication failed" in str(captured.value)
    assert "transaction cleanup denied" in str(captured.value)
    assert (output / "first.a3m").read_text(encoding="utf-8") == "old first"
    assert (output / "second.a3m").read_text(encoding="utf-8") == "old second"


@pytest.mark.parametrize("existing_first", [True, False])
def test_publish_rolls_back_second_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_first: bool
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    write_valid_outputs(staging)
    (staging / "diagnostic.txt").write_text("details", encoding="utf-8")
    if existing_first:
        (output / "first.a3m").write_text("old first", encoding="utf-8")
    (output / "second.a3m").write_text("old second", encoding="utf-8")
    (output / "unknown.txt").write_text("keep", encoding="utf-8")
    real_replace = os.replace

    def fail_second_publication(source: Path, destination: Path) -> None:
        if (
            source.name == "second.a3m"
            and source.parent.name.startswith(".cluster-msa-publish-")
            and destination == output / "second.a3m"
        ):
            raise OSError("publication failed")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_publication)

    with pytest.raises(OutputValidationError, match="publication failed"):
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    if existing_first:
        assert (output / "first.a3m").read_text(encoding="utf-8") == "old first"
    else:
        assert not (output / "first.a3m").exists()
    assert (output / "second.a3m").read_text(encoding="utf-8") == "old second"
    assert (output / "unknown.txt").read_text(encoding="utf-8") == "keep"
    assert (staging / "first.a3m").read_text(encoding="utf-8").startswith(">query")
    assert (staging / "second.a3m").read_text(encoding="utf-8").startswith(">query")
    assert (staging / "diagnostic.txt").read_text(encoding="utf-8") == "details"


def test_rollback_attempts_every_original_restore_and_preserves_unresolved_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    write_valid_outputs(staging)
    (output / "first.a3m").write_text("old first", encoding="utf-8")
    (output / "second.a3m").write_text("old second", encoding="utf-8")
    (output / "unknown.txt").write_text("keep", encoding="utf-8")
    real_replace = os.replace

    def fail_publication_and_second_restore(source: Path, destination: Path) -> None:
        if (
            source.name == "second.a3m"
            and source.parent.name.startswith(".cluster-msa-publish-")
            and destination == output / "second.a3m"
        ):
            raise OSError("publication failed")
        if (
            source.name == "second.a3m"
            and source.parent.name.startswith("publication-backup-")
            and destination == output / "second.a3m"
        ):
            raise OSError("second restore failed")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_publication_and_second_restore)

    with pytest.raises(OutputValidationError) as captured:
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    backups = list(tmp_path.glob(".cluster-msa-publish-*/publication-backup-*"))
    assert len(backups) == 1
    backup = backups[0]
    assert str(backup) in str(captured.value)
    assert "second restore failed" in str(captured.value)
    assert (backup / "second.a3m").read_text(encoding="utf-8") == "old second"
    assert not (backup / "first.a3m").exists()
    assert (output / "first.a3m").read_text(encoding="utf-8") == "old first"
    assert not (output / "second.a3m").exists()
    assert (staging / "first.a3m").read_text(encoding="utf-8").startswith(">query")
    assert (staging / "second.a3m").read_text(encoding="utf-8").startswith(">query")
    assert (output / "unknown.txt").read_text(encoding="utf-8") == "keep"


def test_failed_published_file_rollback_does_not_overwrite_new_or_original_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    write_valid_outputs(staging)
    new_first = (staging / "first.a3m").read_text(encoding="utf-8")
    (output / "first.a3m").write_text("old first", encoding="utf-8")
    (output / "second.a3m").write_text("old second", encoding="utf-8")
    (output / "unknown.txt").write_text("keep", encoding="utf-8")
    real_replace = os.replace

    def fail_publication_and_new_file_rollback(source: Path, destination: Path) -> None:
        if (
            source.name == "second.a3m"
            and source.parent.name.startswith(".cluster-msa-publish-")
            and destination == output / "second.a3m"
        ):
            raise OSError("publication failed")
        if (
            source == output / "first.a3m"
            and destination.name == "first.a3m"
            and destination.parent.name.startswith(".cluster-msa-publish-")
        ):
            raise OSError("new file rollback failed")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_publication_and_new_file_rollback)

    with pytest.raises(OutputValidationError) as captured:
        publish_outputs(staging, output, RECORDS, af3_json=False, overwrite=True)

    backups = list(tmp_path.glob(".cluster-msa-publish-*/publication-backup-*"))
    assert len(backups) == 1
    backup = backups[0]
    assert str(backup) in str(captured.value)
    assert "new file rollback failed" in str(captured.value)
    assert (backup / "first.a3m").read_text(encoding="utf-8") == "old first"
    assert not (backup / "second.a3m").exists()
    assert (output / "first.a3m").read_text(encoding="utf-8") == new_first
    assert (output / "second.a3m").read_text(encoding="utf-8") == "old second"
    assert (staging / "first.a3m").exists()
    assert (staging / "second.a3m").exists()
    assert (output / "unknown.txt").read_text(encoding="utf-8") == "keep"
