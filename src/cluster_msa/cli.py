import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cluster-msa",
        description="Standard and cluster-accelerated batch MSA generation",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("standard", help="Generate batch MSAs directly")
    subparsers.add_parser("accelerated", help="Generate batch MSAs with clustering")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
