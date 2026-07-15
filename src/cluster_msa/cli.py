import argparse
import os
import sys

from cluster_msa.config import build_run_config
from cluster_msa.errors import ClusterMsaError
from cluster_msa.input import load_sequences
from cluster_msa.standard import run_standard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cluster-msa",
        description="Standard and cluster-accelerated batch MSA generation",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode, help_text in (("standard", "Generate batch MSAs directly"), ("accelerated", "Generate batch MSAs with clustering")):
        command = subparsers.add_parser(mode, help=help_text)
        command.add_argument("--input", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--db-path")
        command.add_argument("--colabfold-search")
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--gpu", action="store_true", default=None)
        command.add_argument("--no-gpu", action="store_true")
        command.add_argument("--gpus", default="")
        command.add_argument("--af3-json", action="store_true")
        command.add_argument("--tmp-dir")
        command.add_argument("--work-dir")
        command.add_argument("--keep-work", action="store_true")
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--verbose", action="store_true")
        if mode == "accelerated":
            command.add_argument("--mmseqs")
            command.add_argument("--cluster-identity", type=float, default=0.7)
            command.add_argument("--cluster-coverage", type=float, default=0.8)
            command.add_argument("--cluster-mode", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        config = build_run_config(args, os.environ)
        records = load_sequences(config.input_path)
        if config.mode == "standard":
            run_standard(config, records)
            return 0
        raise NotImplementedError("accelerated mode is not implemented")
    except (ClusterMsaError, NotImplementedError) as error:
        print(f"cluster-msa: {error}", file=sys.stderr)
        return 1
