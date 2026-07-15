import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from cluster_msa.errors import OutputValidationError
from cluster_msa.models import SequenceRecord


_publication_locks: dict[str, threading.Lock] = {}
_publication_locks_guard = threading.Lock()


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

    names = [f"{record.id}.a3m" for record in records]
    if af3_json:
        names.extend(f"{record.id}_data.json" for record in records)
    names.extend(name for name in ("run_manifest.json", "run.log") if _path_exists(staging / name))

    transaction = _stage_publication(staging, output_dir, names)
    try:
        with _publication_lock(output_dir):
            _publish_transaction(transaction, output_dir, names, overwrite)
    except BaseException as error:
        if transaction.exists() and "preserved at" not in str(error):
            try:
                shutil.rmtree(transaction, ignore_errors=True)
            except OSError:
                pass
        raise


def _stage_publication(staging: Path, output_dir: Path, names: Sequence[str]) -> Path:
    transaction: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        transaction = Path(
            tempfile.mkdtemp(prefix=".cluster-msa-publish-", dir=output_dir.parent)
        )
        for name in names:
            shutil.copy2(staging / name, transaction / name)
    except OSError as error:
        cleanup_error: OSError | None = None
        if transaction is not None:
            try:
                shutil.rmtree(transaction)
            except OSError as caught:
                cleanup_error = caught
        cleanup_context = f"; transaction cleanup failed: {cleanup_error}" if cleanup_error else ""
        raise OutputValidationError(
            f"cannot stage publication transaction under {output_dir.parent}: "
            f"{error}{cleanup_context}"
        ) from error
    return transaction


@contextmanager
def _publication_lock(output_dir: Path) -> Iterator[None]:
    resolved = str(output_dir.resolve())
    lock_name = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:24]
    lock_path = output_dir.resolve().parent / f".cluster-msa-{lock_name}.lock"
    with _publication_locks_guard:
        thread_lock = _publication_locks.setdefault(resolved, threading.Lock())

    # flock alone does not reliably serialize separate threads in one process.
    with thread_lock:
        try:
            lock_file = lock_path.open("a+b")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError as error:
            try:
                lock_file.close()
            except (OSError, UnboundLocalError):
                pass
            raise OutputValidationError(
                f"cannot acquire publication lock for {output_dir}: {error}"
            ) from error
        body_error: BaseException | None = None
        try:
            yield
        except BaseException as error:
            body_error = error
            raise
        finally:
            unlock_error: OSError | None = None
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError as error:
                unlock_error = error
            finally:
                try:
                    lock_file.close()
                except OSError as error:
                    warnings.warn(
                        f"cannot close publication lock for {output_dir}: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if unlock_error is not None:
                message = f"cannot explicitly unlock publication lock for {output_dir}: {unlock_error}"
                if body_error is not None and hasattr(body_error, "add_note"):
                    body_error.add_note(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)


def _publish_transaction(
    transaction: Path,
    output_dir: Path,
    names: Sequence[str],
    overwrite: bool,
) -> None:
    _validate_destination(output_dir, overwrite)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputValidationError(f"cannot create output directory: {output_dir}: {error}") from error
    for name in names:
        _validate_destination_file(output_dir / name)

    backup = transaction / "publication-backup-recovery"
    try:
        backup.mkdir()
    except OSError as error:
        raise OutputValidationError(f"cannot create publication backup: {backup}: {error}") from error
    backed_up: list[str] = []
    published: list[str] = []
    try:
        for name in names:
            destination = output_dir / name
            if _path_exists(destination):
                os.replace(destination, backup / name)
                backed_up.append(name)
        for name in names:
            os.replace(transaction / name, output_dir / name)
            published.append(name)
    except OSError as error:
        rollback_failures = _rollback_publication(
            transaction, output_dir, backup, published, backed_up
        )
        if rollback_failures:
            raise OutputValidationError(
                f"output publication failed ({error}); backup preserved at {backup}; "
                f"rollback failures: {'; '.join(rollback_failures)}"
            ) from error
        try:
            shutil.rmtree(backup)
        except OSError as cleanup_error:
            raise OutputValidationError(
                f"output publication failed ({error}); backup preserved at {backup}; "
                f"backup cleanup failed: {cleanup_error}"
            ) from error
        try:
            shutil.rmtree(transaction)
        except OSError as cleanup_error:
            raise OutputValidationError(
                f"output publication failed ({error}); transaction preserved at {transaction}; "
                f"transaction cleanup failed: {cleanup_error}"
            ) from error
        raise OutputValidationError(f"output publication failed: {error}") from error
    cleanup_after_publish(transaction, output_dir)


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
