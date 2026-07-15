import os
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from cluster_msa.errors import ExternalToolError


def run_command(
    command: Sequence[str],
    *,
    stage: str,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    child_env.pop("CUDA_VISIBLE_DEVICES", None)
    if env is not None:
        child_env.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        try:
            result = subprocess.run(
                command, env=child_env, capture_output=True, text=True, check=False
            )
        except OSError as error:
            raise ExternalToolError(f"{stage}: {error}; log: {log_path}") from error
        log.write(result.stdout)
        log.write(result.stderr)
    if verbose:
        print(result.stdout, end="")
        print(result.stderr, end="")
    if result.returncode:
        raise ExternalToolError(
            f"{stage}: failed ({result.returncode}) for {command[0]}; log: {log_path}"
        )
    return result


def get_tool_version(executable: Path, log_path: Path) -> str:
    try:
        result = run_command([str(executable), "--version"], stage="version", log_path=log_path)
    except ExternalToolError as error:
        raise ExternalToolError(f"version check failed for {executable}: {error}") from error
    for line in (result.stdout + result.stderr).splitlines():
        if line.strip():
            return line.strip()
    raise ExternalToolError(f"version check for {executable} returned no output; log: {log_path}")
