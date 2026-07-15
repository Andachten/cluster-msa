from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import get_type_hints

import pytest

from cluster_msa.errors import (
    ClusterMsaError,
    ConfigurationError,
    ExternalToolError,
    InputValidationError,
    OutputValidationError,
)
from cluster_msa.models import ClusterResult, RunConfig, RunResult, SequenceRecord, Toolchain


def test_errors_share_cluster_msa_base() -> None:
    for error_type in (
        InputValidationError,
        ConfigurationError,
        ExternalToolError,
        OutputValidationError,
    ):
        assert issubclass(error_type, ClusterMsaError)


@pytest.mark.parametrize(
    ("model", "expected_fields"),
    [
        (SequenceRecord, ["id", "sequence"]),
        (Toolchain, ["colabfold_search", "mmseqs"]),
        (
            RunConfig,
            [
                "mode",
                "input_path",
                "output_dir",
                "db_path",
                "toolchain",
                "threads",
                "gpu",
                "gpus",
                "af3_json",
                "tmp_dir",
                "work_dir",
                "keep_work",
                "overwrite",
                "verbose",
                "cluster_identity",
                "cluster_coverage",
                "cluster_mode",
                "db_path_supplied",
            ],
        ),
        (ClusterResult, ["representatives", "nonrepresentatives"]),
        (
            RunResult,
            [
                "mode",
                "expected_count",
                "generated_count",
                "representative_count",
                "nonrepresentative_count",
                "fallback_reason",
            ],
        ),
    ],
)
def test_models_have_exact_fields(model: type, expected_fields: list[str]) -> None:
    assert [field.name for field in fields(model)] == expected_fields


def test_models_are_frozen() -> None:
    record = SequenceRecord(id="example", sequence="ACDE")

    with pytest.raises(FrozenInstanceError):
        record.id = "changed"

    for model in (Toolchain, RunConfig, ClusterResult, RunResult):
        assert model.__dataclass_params__.frozen


def test_run_config_defaults_and_path_annotations() -> None:
    config_fields = {field.name: field for field in fields(RunConfig)}
    hints = get_type_hints(RunConfig)

    assert config_fields["cluster_identity"].default == 0.7
    assert config_fields["cluster_coverage"].default == 0.8
    assert config_fields["cluster_mode"].default == 0
    assert config_fields["db_path_supplied"].default is None
    assert hints["db_path_supplied"] == str | None
    for name in ("input_path", "output_dir", "db_path", "tmp_dir", "work_dir"):
        assert hints[name] is Path


def test_cluster_result_uses_immutable_tuple_types() -> None:
    hints = get_type_hints(ClusterResult)

    assert hints["representatives"] == tuple[SequenceRecord, ...]
    assert hints["nonrepresentatives"] == tuple[tuple[SequenceRecord, str], ...]
