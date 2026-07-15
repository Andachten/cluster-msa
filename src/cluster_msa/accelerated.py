import shutil
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from cluster_msa.af3 import write_af3_json
from cluster_msa.clustering import cluster_sequences
from cluster_msa.compact_db import build_compact_database, search_compact_database
from cluster_msa.errors import ConfigurationError, OutputValidationError
from cluster_msa.manifest import mark_manifest_failed, write_manifest
from cluster_msa.models import RunConfig, RunResult, SequenceRecord
from cluster_msa.output import cleanup_after_publish, publish_outputs, staged_output, validate_outputs
from cluster_msa.standard import _preflight_destination, run_full_database_search
from cluster_msa.tools import get_tool_version


def run_accelerated(config: RunConfig, records: Sequence[SequenceRecord]) -> RunResult:
    started_at = datetime.now(timezone.utc)
    total_started = time.monotonic()
    if config.mode != "accelerated":
        raise ValueError("run_accelerated requires accelerated mode")
    if config.toolchain.mmseqs is None:
        raise ConfigurationError("accelerated mode requires mmseqs")
    if config.work_dir.resolve().is_relative_to(config.output_dir.resolve()):
        raise ConfigurationError("work directory must be outside output directory")
    _preflight_destination(config.output_dir, config.overwrite)
    try:
        config.work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="accelerated-", dir=config.work_dir))
    except OSError as error:
        raise OutputValidationError(f"cannot create work directory: {config.work_dir}") from error
    run_config = replace(config, tmp_dir=run_dir / "tmp")

    with staged_output(config.output_dir, run_dir) as staging:
        log_path = staging / "run.log"
        _log(log_path, f"run directory: {run_dir}")
        tool_versions = {
            "colabfold_search": get_tool_version(
                run_config.toolchain.colabfold_search, log_path
            ),
            "mmseqs": get_tool_version(run_config.toolchain.mmseqs, log_path),
        }
        stage_durations = {}
        _log(log_path, "phase 1: cluster sequences")
        stage_started = time.monotonic()
        clusters = cluster_sequences(
            records,
            mmseqs=run_config.toolchain.mmseqs,
            work_dir=run_dir / "clustering",
            tmp_dir=run_config.tmp_dir,
            min_seq_id=run_config.cluster_identity,
            coverage=run_config.cluster_coverage,
            cluster_mode=run_config.cluster_mode,
            threads=run_config.threads,
            log_path=log_path,
        )
        stage_durations["clustering"] = time.monotonic() - stage_started
        representatives = clusters.representatives
        nonrepresentatives = tuple(record for record, _ in clusters.nonrepresentatives)
        fallback_reason = None

        if not nonrepresentatives:
            fallback_reason = "no_non_representatives"
            _log(log_path, f"fallback_reason: {fallback_reason}")
            _log(log_path, "standard fallback: searching original full validated records")
            stage_started = time.monotonic()
            run_full_database_search(
                records,
                staging / "canonical-input.csv",
                staging,
                run_config,
                log_path,
            )
            stage_durations["standard_search"] = time.monotonic() - stage_started
        else:
            rep_dir = run_dir / "representatives"
            nonrep_dir = run_dir / "nonrepresentatives"
            _log(log_path, "phase 2: full database search for representatives")
            stage_started = time.monotonic()
            run_full_database_search(
                representatives,
                rep_dir / "representatives.csv",
                rep_dir,
                run_config,
                log_path,
            )
            stage_durations["representative_search"] = time.monotonic() - stage_started
            _log(log_path, "phase 3: build compact database")
            stage_started = time.monotonic()
            compact_db = build_compact_database(
                rep_dir, run_dir / "compact", run_config, log_path
            )
            stage_durations["compact_database"] = time.monotonic() - stage_started
            _log(log_path, "phase 4: search nonrepresentatives")
            stage_started = time.monotonic()
            isolated_config = replace(
                run_config, work_dir=run_dir / "nonrepresentative-search"
            )
            search_compact_database(
                nonrepresentatives,
                compact_db,
                nonrep_dir,
                isolated_config,
                log_path,
            )
            stage_durations["nonrepresentative_search"] = time.monotonic() - stage_started
            stage_started = time.monotonic()
            if config.af3_json:
                _log(log_path, "phase 5: write nonrepresentative AlphaFold3 JSON")
                for record in nonrepresentatives:
                    try:
                        write_af3_json(
                            record,
                            nonrep_dir / f"{record.id}.a3m",
                            nonrep_dir / f"{record.id}_data.json",
                        )
                    except (OSError, UnicodeError) as error:
                        raise OutputValidationError(
                            f"cannot write AlphaFold3 JSON for {record.id}: {error}"
                        ) from error
            _merge_expected(rep_dir, representatives, staging, config.af3_json, log_path)
            _merge_expected(nonrep_dir, nonrepresentatives, staging, config.af3_json, log_path)

        validate_outputs(staging, records, config.af3_json)
        result = RunResult(
            mode="accelerated",
            expected_count=len(records),
            generated_count=len(records),
            representative_count=len(representatives),
            nonrepresentative_count=len(nonrepresentatives),
            fallback_reason=fallback_reason,
        )
        if not fallback_reason:
            stage_durations["merge_and_staging"] = time.monotonic() - stage_started
        stage_durations["total"] = time.monotonic() - total_started
        write_manifest(
            staging / "run_manifest.json",
            config=config,
            result=result,
            tool_versions=tool_versions,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            stage_durations=stage_durations,
        )
        retained_manifest = None
        if config.keep_work:
            try:
                retained = run_dir / "retained"
                shutil.copytree(staging, retained)
                retained_manifest = retained / "run_manifest.json"
            except OSError as error:
                raise OutputValidationError(f"cannot retain accelerated work: {error}") from error
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


def _merge_expected(
    source: Path,
    records: Sequence[SequenceRecord],
    staging: Path,
    af3_json: bool,
    log_path: Path,
) -> None:
    names = [f"{record.id}.a3m" for record in records]
    if af3_json:
        names.extend(f"{record.id}_data.json" for record in records)
    for name in names:
        destination = staging / name
        if destination.exists():
            raise OutputValidationError(f"output collision while merging: {name}")
        try:
            shutil.copy2(source / name, destination)
        except OSError as error:
            raise OutputValidationError(f"cannot copy expected output {name}: {error}") from error
        _log(log_path, f"copied: {name}")


def _log(path: Path, message: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log:
            log.write(f"{message}\n")
    except OSError as error:
        raise OutputValidationError(f"cannot write accelerated run log: {path}") from error
