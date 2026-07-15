from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SequenceRecord:
    id: str
    sequence: str


@dataclass(frozen=True)
class Toolchain:
    colabfold_search: Path
    mmseqs: Path | None


@dataclass(frozen=True)
class RunConfig:
    mode: Literal["standard", "accelerated"]
    input_path: Path
    output_dir: Path
    db_path: Path
    toolchain: Toolchain
    threads: int
    gpu: bool
    gpus: str
    af3_json: bool
    tmp_dir: Path
    work_dir: Path
    keep_work: bool
    overwrite: bool
    verbose: bool
    cluster_identity: float = 0.7
    cluster_coverage: float = 0.8
    cluster_mode: int = 0
    db_path_supplied: str | None = None


@dataclass(frozen=True)
class ClusterResult:
    representatives: tuple[SequenceRecord, ...]
    nonrepresentatives: tuple[tuple[SequenceRecord, str], ...]


@dataclass(frozen=True)
class RunResult:
    mode: str
    expected_count: int
    generated_count: int
    representative_count: int | None = None
    nonrepresentative_count: int | None = None
    fallback_reason: str | None = None
