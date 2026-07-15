import csv
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from cluster_msa.errors import OutputValidationError
from cluster_msa.manifest import (
    mark_manifest_failed,
    mark_retention_manifests_failed,
    write_manifest,
)
from cluster_msa.models import RunConfig, RunResult, SequenceRecord
from cluster_msa.output import cleanup_after_publish, publish_outputs, staged_output, validate_outputs
from cluster_msa.tools import get_tool_version, run_command


def run_full_database_search(
    records: Sequence[SequenceRecord],
    input_csv: Path,
    destination: Path,
    config: RunConfig,
    log_path: Path,
) -> None:
    _write_canonical_csv(records, input_csv)
    command = [
        str(config.toolchain.colabfold_search),
        str(input_csv),
        str(config.db_path),
        str(destination),
        "--threads",
        str(config.threads),
        "--gpu",
        "1" if config.gpu else "0",
    ]
    if config.af3_json:
        command.append("--af3-json")
    environment = None if config.gpu and not config.gpus else {}
    if config.gpu and config.gpus:
        environment = {"CUDA_VISIBLE_DEVICES": config.gpus}
    if not config.gpu and config.gpus:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"GPU IDs {config.gpus} ignored because GPU mode is disabled\n")
    run_command(
        command,
        stage="colabfold_search",
        log_path=log_path,
        env=environment,
        verbose=config.verbose,
    )


def run_standard(config: RunConfig, records: Sequence[SequenceRecord]) -> RunResult:
    started_at = datetime.now(timezone.utc)
    total_started = time.monotonic()
    if config.mode != "standard":
        raise ValueError("run_standard requires standard mode")
    _preflight_destination(config.output_dir, config.overwrite)
    try:
        config.work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="standard-", dir=config.work_dir))
    except OSError as error:
        raise OutputValidationError(f"cannot create work directory: {config.work_dir}") from error
    with staged_output(config.output_dir, run_dir) as staging:
        input_csv = staging / "canonical-input.csv"
        log_path = staging / "run.log"
        tool_versions = {
            "colabfold_search": get_tool_version(config.toolchain.colabfold_search, log_path)
        }
        search_started = time.monotonic()
        run_full_database_search(records, input_csv, staging, config, log_path)
        search_duration = time.monotonic() - search_started
        validation_started = time.monotonic()
        validate_outputs(staging, records, config.af3_json)
        validation_duration = time.monotonic() - validation_started
        result = RunResult("standard", len(records), len(records))
        finished_at = datetime.now(timezone.utc)
        total_duration = time.monotonic() - total_started
        write_manifest(
            staging / "run_manifest.json",
            config=config,
            result=result,
            tool_versions=tool_versions,
            started_at=started_at,
            finished_at=finished_at,
            stage_durations={
                "full_database_search": search_duration,
                "output_validation": validation_duration,
                "total": total_duration,
            },
        )
        retained_manifest = None
        if config.keep_work:
            retained = run_dir / "retained"
            try:
                shutil.copytree(staging, retained)
            except OSError as caught:
                error = OutputValidationError(f"cannot retain standard work: {caught}")
                mark_retention_manifests_failed(
                    staging / "run_manifest.json",
                    retained / "run_manifest.json",
                    error,
                )
                raise error from caught
            retained_manifest = retained / "run_manifest.json"
        try:
            publish_outputs(
                staging,
                config.output_dir,
                records,
                af3_json=config.af3_json,
                overwrite=config.overwrite,
            )
        except OutputValidationError as error:
            for manifest_path in (staging / "run_manifest.json", retained_manifest):
                if manifest_path is not None:
                    try:
                        mark_manifest_failed(manifest_path, "publication", error)
                    except OutputValidationError as diagnostic_error:
                        error.add_note(str(diagnostic_error))
            raise
    if not config.keep_work:
        cleanup_after_publish(run_dir, config.output_dir)
    return result


def _write_canonical_csv(records: Sequence[SequenceRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("id", "sequence"))
        writer.writerows((record.id, record.sequence) for record in records)


def _preflight_destination(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise OutputValidationError(f"output destination is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()) and not overwrite:
        raise OutputValidationError(f"output destination is not empty: {path}")
