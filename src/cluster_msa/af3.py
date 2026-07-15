import json
import os
import tempfile
from pathlib import Path

from cluster_msa.models import SequenceRecord


def write_af3_json(record: SequenceRecord, a3m_path: Path, output_path: Path) -> Path:
    a3m = a3m_path.read_text(encoding="utf-8")
    payload = {
        "name": record.id,
        "modelSeeds": [1],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": record.sequence,
                    "unpairedMsa": a3m,
                    "pairedMsa": "",
                    "templates": [],
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 1,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path
