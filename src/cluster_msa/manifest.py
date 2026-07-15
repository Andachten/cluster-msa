import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

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
    _validate_timing(started_at, finished_at, stage_durations)
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
            "path": str(config.db_path),
            "resolved_path": str(config.db_path.resolve()),
        },
        "parameters": parameters,
        "tools": tools,
        "timing": {
            "started_at": _utc_iso(started_at),
            "finished_at": _utc_iso(finished_at),
            "stage_durations_seconds": dict(stage_durations),
        },
        "result": result_data,
    }
    _write_atomic(path, json.dumps(document, indent=2, ensure_ascii=True, sort_keys=True) + "\n")


def _validate_timing(
    started_at: datetime, finished_at: datetime, stage_durations: Mapping[str, float]
) -> None:
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise OutputValidationError("manifest timestamps must be timezone-aware")
    if finished_at.tzinfo is None or finished_at.utcoffset() is None:
        raise OutputValidationError("manifest timestamps must be timezone-aware")
    if finished_at < started_at:
        raise OutputValidationError("manifest finish timestamp is before start timestamp")
    for stage, duration in stage_durations.items():
        if not isinstance(duration, (int, float)) or not math.isfinite(duration):
            raise OutputValidationError(f"manifest duration for {stage} must be finite")
        if duration < 0:
            raise OutputValidationError(f"manifest duration for {stage} must be nonnegative")


def _tool_entry(path: Path, versions: Mapping[str, str], key: str) -> dict[str, str]:
    try:
        version = versions[key]
    except KeyError as error:
        raise OutputValidationError(f"manifest is missing version for {key}") from error
    return {"path": str(path), "name": path.name, "version": version}


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
