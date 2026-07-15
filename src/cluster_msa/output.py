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
    del output_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="output-", dir=work_dir))
    try:
        yield staging
    except BaseException:
        raise
    else:
        shutil.rmtree(staging)


def validate_outputs(staging: Path, records: Sequence[SequenceRecord], af3_json: bool) -> None:
    for record in records:
        _validate_nonempty_file(staging / f"{record.id}.a3m")
        if af3_json:
            json_path = staging / f"{record.id}_data.json"
            _validate_nonempty_file(json_path)
            try:
                json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise OutputValidationError(f"invalid output file: {json_path.name}") from error


def publish_outputs(
    staging: Path,
    output_dir: Path,
    records: Sequence[SequenceRecord],
    af3_json: bool,
    overwrite: bool,
) -> None:
    validate_outputs(staging, records, af3_json)
    _validate_destination(output_dir, overwrite)

    names = [f"{record.id}.a3m" for record in records]
    if af3_json:
        names.extend(f"{record.id}_data.json" for record in records)
    names.extend(name for name in ("run_manifest.json", "run.log") if (staging / name).is_file())

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        _validate_destination_file(output_dir / name)
    for name in names:
        os.replace(staging / name, output_dir / name)


def _validate_nonempty_file(path: Path) -> None:
    try:
        is_regular = stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        is_regular = False
    if not is_regular:
        raise OutputValidationError(f"missing or invalid output file: {path.name}")
    try:
        if not path.read_text(encoding="utf-8").strip():
            raise OutputValidationError(f"empty output file: {path.name}")
    except (OSError, UnicodeError) as error:
        raise OutputValidationError(f"cannot read output file: {path.name}") from error


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
