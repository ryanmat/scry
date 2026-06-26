# Scry

Project configuration for Claude Code and contributors. This is a public repository.

## What Scry is

Scry predicts infrastructure failure states from a stream of metrics. It sorts each resource into one of five operational states (NORMAL, PRE_SCALE, PRE_FAILURE, ACTIVE_DEGRADATION, ANOMALY) and forecasts where metrics are headed. It is data-source agnostic; LogicMonitor is one adapter, not the anchor. No trained weights ship; you train on your own data.

## Architecture

- `src/scry/model/` - the X-DEC model (dual-encoder temporal VAE plus deep embedded clustering), training, drift detection, and the optional Chronos forecasting layer. Pure PyTorch, no cloud.
- `src/scry/data/` - feature engineering, the windowing pipeline, and the data-source seam.
  - `data/sources/base.py` - the `DataSource` ABC and the canonical metric schema. This is the contract everything normalizes to.
  - `data/sources/object_store.py` - reads Parquet/CSV from local files or object storage (S3/MinIO/ADLS/GCS) through DuckDB. The default path.
  - `data/sources/http_ingest.py` - the LogicMonitor/HttpIngest adapter (needs the `logicmonitor` extra).
- `src/scry/api/` - the FastAPI service (`/predict`, `/forecast`, `/drift`, `/anomaly`).
- `config/features.yaml` - domain feature profiles. `config/config.yaml` - model and training config.

## Canonical data schema

Every data source normalizes to one long-format table:

| column | type | required |
|---|---|---|
| resource_id | str | yes |
| metric_name | str | yes |
| timestamp | UTC timestamp | yes |
| value | float | yes |
| host_name | str | no |
| datasource_instance | str | no |
| datasource_name | str | no |

Bring your own metrics in this shape and train from scratch.

## Toolchain

- `uv` for environments and dependencies: `uv sync --all-extras`.
- `ruff` for lint and format; `pytest` for tests (`asyncio_mode=auto`).
- Python 3.10+; CI (`.github/workflows/ci.yml`) runs `ruff check` and `pytest` across 3.10, 3.11, and 3.12.
- Import package is `scry`; the published distribution is `scryml` (the bare name `scry` was taken on PyPI).

## Conventions

- Every code file starts with two `# Description:` comment lines.
- Match the surrounding style; consistency within a file beats external standards.
- Tests are part of the deliverable; write them in the same pass as the code.
- Forecasting (Chronos) lives behind the `forecast` extra so the core stays offline-capable. Optional dependencies degrade gracefully; a missing extra returns a clear error, it does not crash unrelated paths.
- Never commit secrets, `.env`, model weights (`*.pt`), or real telemetry. See `.gitignore`.
- Comments describe the code as it is; no "recently changed" or temporal notes.

## Commands

- Install: `uv sync --all-extras`
- Lint: `ruff check`
- Test: `pytest`
- Train: `python scripts/train_model.py --data <path-or-uri>`
- Serve: `uvicorn scry.api.main:app --host 127.0.0.1 --port 8000`
