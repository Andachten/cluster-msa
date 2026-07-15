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
            f">{{record_id}}\\n{{row['sequence']}}\\n", encoding="utf-8"
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
