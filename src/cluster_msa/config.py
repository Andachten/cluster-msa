import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from cluster_msa.errors import ConfigurationError
from cluster_msa.models import RunConfig, Toolchain


def _executable_path(value: str, source: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise ConfigurationError(f"{source}: path does not exist: {path}")
    if not path.is_file():
        raise ConfigurationError(f"{source}: path is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise ConfigurationError(f"{source}: file is not executable: {path}")
    return path.resolve()


def resolve_executable(explicit: str | None, env_name: str, executable: str) -> Path:
    if explicit:
        return _executable_path(explicit, "explicit executable")
    configured = os.environ.get(env_name)
    if configured:
        return _executable_path(configured, env_name)
    found = shutil.which(executable)
    if found is None:
        raise ConfigurationError(f"cannot find executable {executable!r} in PATH")
    return _executable_path(found, executable)


def resolve_database(explicit: Path | None, environ: Mapping[str, str]) -> Path:
    value = explicit or (Path(environ["CLUSTER_MSA_DB"]) if environ.get("CLUSTER_MSA_DB") else None)
    if value is None:
        raise ConfigurationError("database is required: use --db-path or CLUSTER_MSA_DB")
    path = Path(value).expanduser()
    validate_database(path)
    return path.resolve()


def validate_database(path: Path) -> None:
    if not path.exists():
        raise ConfigurationError(f"database does not exist: {path}")
    if not path.is_dir():
        raise ConfigurationError(f"database is not a directory: {path}")
    if not any(item.is_file() and item.name.startswith("uniref30_") for item in path.iterdir()):
        raise ConfigurationError(f"database is missing a uniref30_ component: {path}")
    if not any(
        item.is_file() and item.name.startswith("colabfold_envdb_") for item in path.iterdir()
    ):
        raise ConfigurationError(f"database is missing a colabfold_envdb_ component: {path}")


def _arg(args: argparse.Namespace, *names: str, default=None):
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)
    return default


def _required_path(value, name: str) -> Path:
    if value is None:
        raise ConfigurationError(f"{name} path is required")
    try:
        return Path(value).expanduser()
    except TypeError as error:
        raise ConfigurationError(f"{name} must be a path") from error


def build_run_config(args: argparse.Namespace, environ: Mapping[str, str]) -> RunConfig:
    mode = _arg(args, "mode")
    if mode not in ("standard", "accelerated"):
        raise ConfigurationError("mode must be standard or accelerated")
    threads = _arg(args, "threads", default=1)
    if not isinstance(threads, int) or threads <= 0:
        raise ConfigurationError("threads must be positive")
    identity = _arg(args, "cluster_identity", default=0.7)
    coverage = _arg(args, "cluster_coverage", default=0.8)
    cluster_mode = _arg(args, "cluster_mode", default=0)
    if not 0 < identity <= 1 or not 0 < coverage <= 1 or cluster_mode < 0:
        raise ConfigurationError("cluster parameters are invalid")

    input_path = _required_path(_arg(args, "input", "input_path"), "input")
    if not input_path.is_file():
        raise ConfigurationError(f"input is not a file: {input_path}")
    output_dir = _required_path(_arg(args, "output_dir", "output"), "output")
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigurationError(f"output is not a directory: {output_dir}")

    search = _resolve_required_with_environment(
        _arg(args, "colabfold_search"), environ, "COLABFOLD_SEARCH", "colabfold_search"
    )
    mmseqs_explicit = _arg(args, "mmseqs")
    mmseqs = _resolve_with_environment(mmseqs_explicit, environ, "MMSEQS", "mmseqs")
    if mode == "accelerated" and mmseqs is None:
        raise ConfigurationError("accelerated mode requires mmseqs")

    gpu = bool(_arg(args, "gpu", default=True))
    gpus = _arg(args, "gpus") or ""
    tmp_dir = Path(_arg(args, "tmp_dir", "tmp") or ".cluster-msa-tmp").expanduser()
    _validate_directory_root(tmp_dir, "tmp")
    work_dir = _resolve_work_dir(args, mode, output_dir, tmp_dir)
    _validate_directory_root(work_dir, "work")
    return RunConfig(
        mode=mode,
        input_path=input_path,
        output_dir=output_dir,
        db_path=resolve_database(_arg(args, "db_path", "db"), environ),
        toolchain=Toolchain(search, mmseqs),
        threads=threads,
        gpu=gpu,
        gpus=gpus,
        af3_json=bool(_arg(args, "af3_json", default=False)),
        tmp_dir=tmp_dir,
        work_dir=work_dir,
        keep_work=bool(_arg(args, "keep_work", default=False)),
        overwrite=bool(_arg(args, "overwrite", default=False)),
        verbose=bool(_arg(args, "verbose", default=False)),
        cluster_identity=identity,
        cluster_coverage=coverage,
        cluster_mode=cluster_mode,
    )


def _resolve_with_environment(explicit, environ, env_name, executable):
    if explicit:
        return _executable_path(explicit, "explicit executable")
    configured = environ.get(env_name)
    if configured:
        return _executable_path(configured, env_name)
    if executable == "mmseqs":
        found = shutil.which(executable)
        return _executable_path(found, executable) if found else None
    return None


def _resolve_work_dir(args, mode: str, output_dir: Path, tmp_dir: Path) -> Path:
    explicit = _arg(args, "work_dir", "work")
    if mode == "standard":
        candidate = (tmp_dir / "cluster-msa-work").expanduser()
        if _inside(candidate, output_dir):
            candidate = Path(tempfile.gettempdir()) / "cluster-msa-work"
        if _inside(candidate, output_dir):
            raise ConfigurationError(
                "cannot place standard work directory outside output directory"
            )
        return candidate

    candidate = Path(explicit or tmp_dir / "cluster-msa-work").expanduser()
    if _inside(candidate, output_dir):
        raise ConfigurationError("work directory must be outside output directory")
    return candidate


def _inside(path: Path, directory: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


def _validate_directory_root(path: Path, name: str) -> None:
    if path.exists() and not path.is_dir():
        raise ConfigurationError(f"{name} directory is not a directory: {path}")
    parent = path
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise ConfigurationError(f"{name} directory parent is not a directory: {parent}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise ConfigurationError(f"{name} directory parent is not writable: {parent}")


def _resolve_required_with_environment(explicit, environ, env_name, executable):
    if explicit:
        return _executable_path(explicit, "explicit executable")
    configured = environ.get(env_name)
    if configured:
        return _executable_path(configured, env_name)
    found = shutil.which(executable)
    if found is None:
        raise ConfigurationError(f"cannot find executable {executable!r} in PATH")
    return _executable_path(found, executable)
