import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from cluster_msa import __version__
from cluster_msa.errors import OutputValidationError
from cluster_msa.models import RunConfig, RunResult


def write_manifest(
    path: Path,
    *,
    config: RunConfig,
    result: RunResult,
    tool_versions: Mapping[str, str],
    started_at: datetime,
    finished_at: datetime,
    stage_durations: Mapping[str, float],
) -> None:
    try:
        _validate_inputs(config, result, tool_versions, started_at, finished_at, stage_durations)
        _write_success_manifest(
            path,
            config,
            result,
            tool_versions,
            started_at,
            finished_at,
            stage_durations,
        )
    except OutputValidationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise OutputValidationError("invalid run manifest data") from error


def _write_success_manifest(
    path: Path,
    config: RunConfig,
    result: RunResult,
    tool_versions: Mapping[str, str],
    started_at: datetime,
    finished_at: datetime,
    stage_durations: Mapping[str, float],
) -> None:
    parameters = {
        "threads": config.threads,
        "gpu": config.gpu,
        "gpus": config.gpus,
        "af3": config.af3_json,
    }
    if config.mode == "accelerated":
        parameters.update(
            cluster_identity=config.cluster_identity,
            cluster_coverage=config.cluster_coverage,
            cluster_mode=config.cluster_mode,
        )

    tools = {
        "colabfold_search": _tool_entry(
            config.toolchain.colabfold_search, tool_versions, "colabfold_search"
        )
    }
    if config.mode == "accelerated" and config.toolchain.mmseqs is not None:
        tools["mmseqs"] = _tool_entry(config.toolchain.mmseqs, tool_versions, "mmseqs")

    result_data = {
        "expected_count": result.expected_count,
        "generated_count": result.generated_count,
    }
    if config.mode == "accelerated":
        result_data.update(
            representative_count=result.representative_count,
            nonrepresentative_count=result.nonrepresentative_count,
            fallback_reason=result.fallback_reason,
        )

    document = {
        "schema_version": 1,
        "package": {"name": "cluster-msa", "version": __version__},
        "status": "success",
        "mode": config.mode,
        "input": {"path": str(config.input_path), "count": result.expected_count},
        "database": {
            "path": config.db_path_supplied or str(config.db_path),
            "resolved_path": str(config.db_path.resolve()),
        },
        "parameters": parameters,
        "tools": tools,
        "timing": {
            "timing_scope": "through_pre_manifest_finalization",
            "started_at": _utc_iso(started_at),
            "finished_at": _utc_iso(finished_at),
            "stage_durations_seconds": dict(stage_durations),
        },
        "result": result_data,
    }
    _write_atomic(
        path,
        json.dumps(
            document, indent=2, ensure_ascii=True, sort_keys=True, allow_nan=False
        ) + "\n",
    )


def mark_manifest_failed(path: Path, stage: str, error: Exception) -> None:
    del error
    messages = {
        "publication": "output publication failed",
        "work_retention": "work retention failed",
    }
    if stage not in messages:
        raise OutputValidationError("cannot mark run manifest failed: invalid failure stage")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(
            status="failed",
            failure_stage=stage,
            error=messages[stage],
        )
        _write_atomic(
            path,
            json.dumps(
                document, indent=2, ensure_ascii=True, sort_keys=True, allow_nan=False
            ) + "\n",
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as caught:
        raise OutputValidationError(f"cannot mark run manifest failed: {path}") from caught


def mark_retention_manifests_failed(
    staging_manifest: Path, retained_manifest: Path, error: OutputValidationError
) -> None:
    try:
        mark_manifest_failed(staging_manifest, "work_retention", error)
    except OutputValidationError as diagnostic_error:
        error.add_note(str(diagnostic_error))

    try:
        mode = retained_manifest.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as caught:
        error.add_note(f"cannot inspect retained manifest: {retained_manifest}: {caught}")
        return
    if not stat.S_ISREG(mode):
        error.add_note(f"retained manifest is not a regular file: {retained_manifest}")
        _quarantine_retained_manifest(retained_manifest, error)
        return
    try:
        mark_manifest_failed(retained_manifest, "work_retention", error)
    except OutputValidationError as diagnostic_error:
        error.add_note(str(diagnostic_error))
        _quarantine_retained_manifest(retained_manifest, error)


def _quarantine_retained_manifest(
    retained_manifest: Path, error: OutputValidationError
) -> None:
    quarantine = retained_manifest.with_name(f"{retained_manifest.name}.failed-unusable")
    try:
        os.replace(retained_manifest, quarantine)
    except OSError as caught:
        error.add_note(
            f"cannot quarantine retained manifest: {retained_manifest}: {caught}"
        )


def _validate_timing(
    started_at: datetime, finished_at: datetime, stage_durations: Mapping[str, float]
) -> None:
    if not isinstance(started_at, datetime) or not isinstance(finished_at, datetime):
        raise OutputValidationError("manifest timestamps must be datetimes")
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise OutputValidationError("manifest timestamps must be timezone-aware")
    if finished_at.tzinfo is None or finished_at.utcoffset() is None:
        raise OutputValidationError("manifest timestamps must be timezone-aware")
    if finished_at < started_at:
        raise OutputValidationError("manifest finish timestamp is before start timestamp")
    for stage, duration in stage_durations.items():
        if not isinstance(stage, str) or not stage.strip():
            raise OutputValidationError("manifest stage names must be nonempty strings")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
        ):
            raise OutputValidationError(f"manifest duration for {stage} must be finite")
        if duration < 0:
            raise OutputValidationError(f"manifest duration for {stage} must be nonnegative")


def _tool_entry(path: Path, versions: Mapping[str, str], key: str) -> dict[str, str]:
    try:
        version = versions[key]
    except KeyError as error:
        raise OutputValidationError(f"manifest is missing version for {key}") from error
    name = path.name
    if not name:
        raise OutputValidationError(f"manifest tool name for {key} must be nonempty")
    return {"path": str(path), "name": name, "version": version}


def _validate_inputs(
    config: RunConfig,
    result: RunResult,
    tool_versions: Mapping[str, str],
    started_at: datetime,
    finished_at: datetime,
    stage_durations: Mapping[str, float],
) -> None:
    if config.mode not in ("standard", "accelerated"):
        raise OutputValidationError("manifest mode must be standard or accelerated")
    if result.mode != config.mode:
        raise OutputValidationError("manifest config and result modes must match")
    _validate_count("expected", result.expected_count)
    _validate_count("generated", result.generated_count)
    if result.expected_count == 0:
        raise OutputValidationError("manifest expected count must be positive")
    if result.generated_count != result.expected_count:
        raise OutputValidationError("manifest generated count must equal expected count")
    if result.fallback_reason is not None and not isinstance(result.fallback_reason, str):
        raise OutputValidationError("manifest fallback reason must be a string or null")

    if config.mode == "standard":
        if (
            result.representative_count is not None
            or result.nonrepresentative_count is not None
            or result.fallback_reason is not None
        ):
            raise OutputValidationError("standard manifest cannot contain accelerated results")
    else:
        if config.toolchain.mmseqs is None:
            raise OutputValidationError("accelerated manifest requires mmseqs")
        _validate_count("representative", result.representative_count)
        _validate_count("nonrepresentative", result.nonrepresentative_count)
        if result.representative_count + result.nonrepresentative_count != result.expected_count:
            raise OutputValidationError("accelerated manifest counts must equal expected count")
        if result.fallback_reason not in (None, "no_non_representatives"):
            raise OutputValidationError("accelerated manifest fallback reason is invalid")
        if result.fallback_reason == "no_non_representatives" and (
            result.nonrepresentative_count != 0
            or result.representative_count != result.expected_count
        ):
            raise OutputValidationError("accelerated fallback counts are inconsistent")
        if result.fallback_reason is None and (
            result.representative_count == 0 or result.nonrepresentative_count == 0
        ):
            raise OutputValidationError("accelerated non-fallback counts must be positive")

    _validate_config(config)

    if not isinstance(tool_versions, Mapping):
        raise OutputValidationError("manifest tool versions must be a mapping")
    for name, version in tool_versions.items():
        if not isinstance(name, str) or not name.strip():
            raise OutputValidationError("manifest tool names must be nonempty strings")
        if not isinstance(version, str) or not version.strip():
            raise OutputValidationError(f"manifest version for {name} must be nonempty")
    required_tools = {"colabfold_search"}
    if config.mode == "accelerated":
        required_tools.add("mmseqs")
    if not required_tools.issubset(tool_versions):
        raise OutputValidationError("manifest is missing a required tool version")

    for key, executable in (
        ("colabfold_search", config.toolchain.colabfold_search),
        ("mmseqs", config.toolchain.mmseqs),
    ):
        if executable is not None and (
            not isinstance(executable, Path) or not executable.name
        ):
            raise OutputValidationError(f"manifest tool name for {key} must be nonempty")
    if not isinstance(stage_durations, Mapping):
        raise OutputValidationError("manifest stage durations must be a mapping")
    _validate_timing(started_at, finished_at, stage_durations)
    _validate_stage_schema(config.mode, result.fallback_reason, stage_durations)


def _validate_config(config: RunConfig) -> None:
    if isinstance(config.threads, bool) or not isinstance(config.threads, int) or config.threads <= 0:
        raise OutputValidationError("manifest threads must be a positive integer")
    if not isinstance(config.gpu, bool) or not isinstance(config.af3_json, bool):
        raise OutputValidationError("manifest GPU and AF3 flags must be booleans")
    if not isinstance(config.gpus, str):
        raise OutputValidationError("manifest GPU IDs must be a string")
    for name in ("input_path", "output_dir", "db_path", "tmp_dir", "work_dir"):
        value = getattr(config, name)
        if not isinstance(value, Path) or not str(value):
            raise OutputValidationError(f"manifest {name} must be a nonempty path")
    if config.db_path_supplied is not None and (
        not isinstance(config.db_path_supplied, str) or not config.db_path_supplied
    ):
        raise OutputValidationError("manifest supplied database path must be a nonempty string")
    if config.mode == "accelerated":
        for name in ("cluster_identity", "cluster_coverage"):
            value = getattr(config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise OutputValidationError(f"manifest {name} must be in (0, 1]")
        if (
            isinstance(config.cluster_mode, bool)
            or not isinstance(config.cluster_mode, int)
            or config.cluster_mode < 0
        ):
            raise OutputValidationError("manifest cluster mode must be a nonnegative integer")


def _validate_count(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OutputValidationError(f"manifest {name} count must be a nonnegative integer")


def _validate_stage_schema(
    mode: str, fallback_reason: str | None, stage_durations: Mapping[str, float]
) -> None:
    if mode == "standard":
        expected = {"full_database_search", "output_validation", "total"}
    elif fallback_reason == "no_non_representatives":
        expected = {"clustering", "standard_search", "output_validation", "total"}
    else:
        expected = {
            "clustering",
            "representative_search",
            "compact_database",
            "nonrepresentative_search",
            "merge_and_staging",
            "output_validation",
            "total",
        }
    if set(stage_durations) != expected:
        raise OutputValidationError("manifest stage durations do not match the workflow")
    total = stage_durations["total"]
    tolerance = max(1.0, total) * 1e-9
    non_total = [duration for name, duration in stage_durations.items() if name != "total"]
    if any(duration > total + tolerance for duration in non_total):
        raise OutputValidationError("manifest stage duration exceeds total")
    if sum(non_total) > total + tolerance:
        raise OutputValidationError("manifest stage durations overlap or exceed total")


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_atomic(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(content)
        os.replace(temporary, path)
    except (OSError, UnicodeError) as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise OutputValidationError(f"cannot write run manifest: {path}: {error}") from error
