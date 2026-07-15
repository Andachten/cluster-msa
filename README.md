# cluster-msa

`cluster-msa` provides standard and cluster-accelerated batch multiple sequence
alignment generation. The command-line interface is currently scaffolded for
the `standard` and `accelerated` workflows.

## Install

```bash
python -m pip install -e ".[dev]"
```

## CLI

```bash
cluster-msa --help
```

Example input sequences are available at `examples/inputs.csv`.

## Real-tool smoke test

The real-tool smoke test is opt-in because it runs `colabfold_search` against a
real database and can take substantial time. Set `CLUSTER_MSA_SMOKE_DB` to the
database path to enable it:

```bash
CLUSTER_MSA_SMOKE_DB=/path/to/db pytest -m smoke -q
```

To force CPU execution, also set `CLUSTER_MSA_SMOKE_CPU=1`:

```bash
CLUSTER_MSA_SMOKE_DB=/path/to/db CLUSTER_MSA_SMOKE_CPU=1 pytest -m smoke -q
```

The test uses `colabfold_search` from `PATH` by default. Set the existing
`COLABFOLD_SEARCH` environment variable to use a specific executable.
