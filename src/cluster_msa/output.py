import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from cluster_msa.errors import OutputValidationError
from cluster_msa.models import SequenceRecord


@contextmanager
def staged_output(output_dir: Path, work_dir: Path) -> Iterator[Path]:
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="output-", dir=work_dir))
    except OSError as error:
        raise OutputValidationError(f"cannot create output staging under {work_dir}: {error}") from error
    try:
        yield staging
    except BaseException:
        raise
    else:
        cleanup_after_publish(staging, output_dir)


def cleanup_after_publish(path: Path, output_dir: Path) -> bool:
    """Remove published work without turning successful results into a failed run."""
    try:
        shutil.rmtree(path)
    except OSError as error:
        try:
            with (output_dir / "run.log").open("a", encoding="utf-8") as log:
                log.write(f"cleanup warning: retained {path}: {error}\n")
        except OSError:
            pass
        return False
    return True


def validate_outputs(staging: Path, records: Sequence[SequenceRecord], af3_json: bool) -> None:
    _validate_unique_ids(records)
    for record in records:
        _validate_nonempty_file(staging / f"{record.id}.a3m")
        if af3_json:
            json_path = staging / f"{record.id}_data.json"
            _validate_nonempty_file(json_path)
            _validate_json(json_path)

    manifest = staging / "run_manifest.json"
    if _path_exists(manifest):
        _validate_nonempty_file(manifest)
        _validate_json(manifest)
    log = staging / "run.log"
    if _path_exists(log):
        _validate_regular_file(log)


def publish_outputs(
    staging: Path,
    output_dir: Path,
    records: Sequence[SequenceRecord],
    af3_json: bool,
    overwrite: bool,
) -> None:
    _validate_unique_ids(records)
    validate_outputs(staging, records, af3_json)
    _validate_destination(output_dir, overwrite)

    names = [f"{record.id}.a3m" for record in records]
    if af3_json:
        names.extend(f"{record.id}_data.json" for record in records)
    names.extend(name for name in ("run_manifest.json", "run.log") if _path_exists(staging / name))

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        _validate_destination_file(output_dir / name)

    backup = Path(tempfile.mkdtemp(prefix="publication-backup-", dir=staging))
    backed_up: list[str] = []
    published: list[str] = []
    try:
        for name in names:
            destination = output_dir / name
            if _path_exists(destination):
                os.replace(destination, backup / name)
                backed_up.append(name)
        for name in names:
            os.replace(staging / name, output_dir / name)
            published.append(name)
    except OSError as error:
        rollback_failures = _rollback_publication(staging, output_dir, backup, published, backed_up)
        if rollback_failures:
            raise OutputValidationError(
                f"output publication failed ({error}); backup preserved at {backup}; "
                f"rollback failures: {'; '.join(rollback_failures)}"
            ) from error
        shutil.rmtree(backup)
        raise OutputValidationError(f"output publication failed: {error}") from error
    cleanup_after_publish(backup, output_dir)


def _rollback_publication(
    staging: Path,
    output_dir: Path,
    backup: Path,
    published: Sequence[str],
    backed_up: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    blocked: set[str] = set()
    for name in reversed(published):
        try:
            os.replace(output_dir / name, staging / name)
        except OSError as error:
            blocked.add(name)
            failures.append(f"could not return published {name}: {error}")

    for name in reversed(backed_up):
        if name in blocked:
            failures.append(f"original {name} remains in backup because destination is occupied")
            continue
        try:
            os.replace(backup / name, output_dir / name)
        except OSError as error:
            failures.append(f"could not restore original {name}: {error}")
    return failures


def _validate_nonempty_file(path: Path) -> None:
    _validate_regular_file(path)
    try:
        if not path.read_text(encoding="utf-8").strip():
            raise OutputValidationError(f"empty output file: {path.name}")
    except (OSError, UnicodeError) as error:
        raise OutputValidationError(f"cannot read output file: {path.name}") from error


def _validate_regular_file(path: Path) -> None:
    try:
        is_regular = stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        is_regular = False
    if not is_regular:
        raise OutputValidationError(f"missing or invalid output file: {path.name}")


def _validate_json(path: Path) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OutputValidationError(f"invalid output file: {path.name}") from error


def _validate_unique_ids(records: Sequence[SequenceRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise OutputValidationError(f"duplicate output record ID: {record.id}")
        seen.add(record.id)


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise OutputValidationError(f"cannot inspect output path: {path.name}") from error
    return True


def _validate_destination(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise OutputValidationError(f"output destination is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise OutputValidationError(f"output destination is not empty: {output_dir}")


def _validate_destination_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as error:
        raise OutputValidationError(f"cannot inspect output destination: {path.name}") from error
    if not stat.S_ISREG(mode):
        raise OutputValidationError(f"unsafe output destination: {path.name}")
