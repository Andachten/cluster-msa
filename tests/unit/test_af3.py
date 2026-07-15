import json
import os
from pathlib import Path

import pytest

from cluster_msa.af3 import write_af3_json
from cluster_msa.models import SequenceRecord


def test_write_af3_json_has_exact_schema_and_stable_format(tmp_path: Path) -> None:
    a3m = tmp_path / "example.a3m"
    output = tmp_path / "example_data.json"
    a3m_text = ">query\nACDE\n>hit\nAC-E\n"
    a3m.write_text(a3m_text, encoding="utf-8")

    result = write_af3_json(SequenceRecord(id="example", sequence="ACDE"), a3m, output)

    assert result == output
    assert output.read_text(encoding="utf-8") == (
        "{\n"
        '  "name": "example",\n'
        '  "modelSeeds": [\n'
        "    1\n"
        "  ],\n"
        '  "sequences": [\n'
        "    {\n"
        '      "protein": {\n'
        '        "id": "A",\n'
        '        "sequence": "ACDE",\n'
        '        "unpairedMsa": ">query\\nACDE\\n>hit\\nAC-E\\n",\n'
        '        "pairedMsa": "",\n'
        '        "templates": []\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "dialect": "alphafold3",\n'
        '  "version": 1\n'
        "}\n"
    )
    assert json.loads(output.read_text(encoding="utf-8"))["name"] == "example"


def test_write_af3_json_replaces_atomically_and_cleans_temp_on_failure(tmp_path: Path) -> None:
    a3m = tmp_path / "example.a3m"
    output = tmp_path / "example_data.json"
    a3m.write_text(">query\nACDE\n", encoding="utf-8")
    output.write_text("old", encoding="utf-8")

    write_af3_json(SequenceRecord(id="example", sequence="ACDE"), a3m, output)

    assert output.read_text(encoding="utf-8").endswith("\n")
    assert list(tmp_path.glob(".example_data.json.*.tmp")) == []


def test_write_af3_json_preserves_destination_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a3m = tmp_path / "example.a3m"
    output = tmp_path / "example_data.json"
    a3m.write_text(">query\nACDE\n", encoding="utf-8")
    output.write_text("old", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_af3_json(SequenceRecord(id="example", sequence="ACDE"), a3m, output)

    assert output.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".example_data.json.*.tmp")) == []
