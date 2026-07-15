import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class FakeColabfoldSearch:
    executable: Path
    invocation_path: Path

    def invocation(self) -> dict:
        return self.invocations()[-1]

    def invocations(self) -> list[dict]:
        return json.loads(self.invocation_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FakeMmseqs:
    executable: Path
    invocation_path: Path

    def invocations(self) -> list[list[str]]:
        return json.loads(self.invocation_path.read_text(encoding="utf-8"))

    def count(self, subcommand: str) -> int:
        return sum(command[0] == subcommand for command in self.invocations())


@pytest.fixture
def fake_database(tmp_path: Path) -> Path:
    database = tmp_path / "database"
    database.mkdir()
    (database / "uniref30_component").write_text("fake\n", encoding="utf-8")
    (database / "colabfold_envdb_component").write_text("fake\n", encoding="utf-8")
    return database


@pytest.fixture
def fake_colabfold_search(tmp_path: Path, monkeypatch) -> FakeColabfoldSearch:
    executable = tmp_path / "colabfold_search"
    invocation_path = tmp_path / "colabfold-invocation.json"
    executable.write_text(
        f"""#!{sys.executable}
import csv
import json
import os
import pathlib
import sys

if sys.argv[1:] == ["--version"]:
    print("fake-colabfold-search 1.0")
    raise SystemExit(0)

record_path = pathlib.Path(os.environ["FAKE_COLABFOLD_RECORD"])
history = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else []
history.append({{
    "argv": sys.argv[1:],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "input_csv": pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    if len(sys.argv) > 1 and pathlib.Path(sys.argv[1]).is_file()
    else None,
}})
record_path.write_text(json.dumps(history), encoding="utf-8")
if os.environ.get("FAKE_COLABFOLD_FAIL") == "1":
    print("forced fake failure", file=sys.stderr)
    raise SystemExit(7)

input_path, _database, destination = map(pathlib.Path, sys.argv[1:4])
destination.mkdir(parents=True, exist_ok=True)
with input_path.open(newline="", encoding="utf-8") as source:
    for row in csv.DictReader(source):
        record_id = row["id"]
        if record_id == os.environ.get("FAKE_COLABFOLD_SKIP_ID"):
            continue
        (destination / f"{{record_id}}.a3m").write_text(
            f">{{record_id}}\\n{{row['sequence']}}\\n"
            + (">hit\\nACDE\\n" if os.environ.get("FAKE_COLABFOLD_ADD_HIT") == "1" else ""),
            encoding="utf-8",
        )
        if "--af3-json" in sys.argv:
            (destination / f"{{record_id}}_data.json").write_text(
                json.dumps({{"name": record_id}}), encoding="utf-8"
            )
print("fake search complete")
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("FAKE_COLABFOLD_RECORD", str(invocation_path))
    return FakeColabfoldSearch(executable, invocation_path)


@pytest.fixture
def fake_mmseqs(tmp_path: Path, monkeypatch) -> FakeMmseqs:
    executable = tmp_path / "mmseqs"
    invocation_path = tmp_path / "mmseqs-invocation.json"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

if sys.argv[1:] == ["--version"]:
    print("fake-mmseqs 1.0")
    raise SystemExit(0)

subcommand = sys.argv[1]
record_path = pathlib.Path(os.environ["FAKE_MMSEQS_RECORD"])
history = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else []
history.append(sys.argv[1:])
record_path.write_text(json.dumps(history), encoding="utf-8")
if os.environ.get("FAKE_MMSEQS_FAIL") == subcommand:
    print(f"forced {{subcommand}} failure", file=sys.stderr)
    raise SystemExit(8)

def artifact_family(prefix):
    prefix = pathlib.Path(prefix)
    for suffix in ("", ".dbtype", ".index"):
        pathlib.Path(str(prefix) + suffix).write_text("artifact\\n", encoding="utf-8")

if subcommand == "easy-cluster":
    fasta = pathlib.Path(sys.argv[2])
    prefix = pathlib.Path(sys.argv[3])
    ids = [line[1:].strip() for line in fasta.read_text(encoding="utf-8").splitlines() if line.startswith(">")]
    configured = os.environ.get("FAKE_MMSEQS_CLUSTER_TSV")
    content = configured if configured is not None else "".join(f"{{ids[0]}}\\t{{item}}\\n" for item in ids)
    prefix.with_name(prefix.name + "_cluster.tsv").write_text(content, encoding="utf-8")
elif subcommand == "createdb":
    artifact_family(sys.argv[3])
elif subcommand == "createindex":
    prefix = pathlib.Path(sys.argv[2])
    for suffix in (".idx", ".idx.dbtype", ".idx.index"):
        pathlib.Path(str(prefix) + suffix).write_text("artifact\\n", encoding="utf-8")
elif subcommand == "search":
    artifact_family(sys.argv[4])
elif subcommand == "result2msa":
    query_fasta = pathlib.Path(sys.argv[2]).parent / "queries.fasta"
    lines = query_fasta.read_text(encoding="utf-8").splitlines()
    records = [(lines[index][1:], lines[index + 1]) for index in range(0, len(lines), 2)]
    entries = [f">{{record_id}}\\n{{sequence}}\\n>hit\\nACDE\\n".encode() for record_id, sequence in records]
    if os.environ.get("FAKE_MMSEQS_RESULT2MSA_EMPTY") == "1":
        entries.append(b"")
    pathlib.Path(sys.argv[5]).write_bytes(b"\\0".join(entries) + b"\\0")
else:
    print(f"unsupported fake subcommand: {{subcommand}}", file=sys.stderr)
    raise SystemExit(9)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("FAKE_MMSEQS_RECORD", str(invocation_path))
    return FakeMmseqs(executable, invocation_path)
