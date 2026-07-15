import pytest

from cluster_msa.cli import build_parser


def test_standard_parser_uses_documented_long_options_only():
    args = build_parser().parse_args(
        [
            "standard",
            "--input",
            "input.csv",
            "--output-dir",
            "output",
            "--db-path",
            "db",
            "--tmp-dir",
            "tmp",
            "--mmseqs",
            "mmseqs",
        ]
    )
    assert args.input == "input.csv"
    assert args.output_dir == "output"
    assert args.db_path == "db"
    assert args.tmp_dir == "tmp"
    assert args.mmseqs == "mmseqs"
    assert not hasattr(args, "cluster_identity")
    assert not hasattr(args, "work_dir")


@pytest.mark.parametrize(
    "arguments",
    [
        ["standard", "input.csv", "output"],
        ["standard", "--db", "db"],
        ["standard", "--work-dir", "work"],
    ],
)
def test_standard_parser_rejects_undocumented_positional_or_alias_options(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_accelerated_parser_owns_accelerated_options():
    args = build_parser().parse_args(
        [
            "accelerated",
            "--input",
            "input.csv",
            "--output-dir",
            "output",
            "--db-path",
            "db",
            "--mmseqs",
            "mmseqs",
            "--work-dir",
            "work",
            "--cluster-identity",
            "0.9",
            "--cluster-coverage",
            "0.8",
            "--cluster-mode",
            "1",
        ]
    )
    assert args.mmseqs == "mmseqs"
    assert args.cluster_identity == 0.9
    assert args.cluster_coverage == 0.8
    assert args.cluster_mode == 1
    assert args.work_dir == "work"
