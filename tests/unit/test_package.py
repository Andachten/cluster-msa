from cluster_msa import __version__
from cluster_msa.cli import build_parser


def test_package_version_and_cli_help() -> None:
    assert __version__ == "0.1.0"
    help_text = build_parser().format_help().lower()
    assert "standard" in help_text
    assert "accelerated" in help_text
