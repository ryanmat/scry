# Getting data in

Scry is bring-your-own-data. Produce the canonical table (see [data-contract.md](data-contract.md)) and point Scry at it. The object store is the single read path; how the data gets there is your choice. The schema is the contract, not the route, so use whatever already fits your stack.

## The read path: object storage

Scry reads Parquet or CSV directly through DuckDB. One reader, any backend, selected by the URI scheme:

```bash
export SCRY_DATA_URI="data/metrics/**/*.parquet"            # local
export SCRY_DATA_URI="s3://bucket/metrics/**/*.parquet"     # S3, MinIO, Ceph
export SCRY_DATA_URI="az://container/metrics/**/*.parquet"  # Azure Data Lake
export SCRY_DATA_URI="gs://bucket/metrics/**/*.parquet"     # GCS
```

Cloud credentials come from the standard environment for that cloud (`AWS_*`, the Azure credential chain, and so on). Install the matching extra: `scry[s3]` or `scry[azure]`. Local files need nothing. CSV works too; pass `--format csv` if the extension is ambiguous. Parquet is recommended: it preserves column types (the quality checks need a real `timestamp` type) and supports partition pruning.

The routes below all land the canonical table in this object store. Pick one, mix them, or write your own.

## Bring your own ETL

The common case: land the canonical table as Parquet or CSV with whatever you already run (a pipeline, a notebook, dbt, Spark). Match the [schema](data-contract.md), partition by time if you like, and you are done. Nothing Scry-specific is required.

## LogicMonitor: REST exporter

Install `scry[logicmonitor]`. `scripts/export_logicmonitor.py` (module `scry.data.sources.lm_export`) pulls device metrics straight from the LogicMonitor REST API with LMv1 auth and writes the canonical table:

```bash
uv run python scripts/export_logicmonitor.py \
  --device-id <id> --datasource-filter <name> --days 14 \
  --output data/metrics.parquet
```

Select targets with `--device-id`, `--group-id`, or `--name-filter`. Credentials come from `LM_ACCESS_ID` / `LM_ACCESS_KEY` / `LM_COMPANY` (a gitignored `.env` works). The REST API carries full coverage, including cloud and computed datapoints and historical backfill, but is rate limited to about 500 GET/min per portal.

## LogicMonitor: Data Publisher (OTLP push)

Only relevant if you already ship metrics through LogicMonitor Data Publisher (OTLP). The companion [HttpIngest](https://github.com/ryanmat/HttpIngest) service receives the OTLP payloads over HTTP and writes the canonical Parquet to your object store (for example ADLS); Scry then reads it via `az://`. This is real-time push of normal collector datapoints, with no historical backfill. It is one optional route for OTLP shops, not a requirement; if you do not use Data Publisher, ignore it.

## Feature profiles

`config/features.yaml` defines, per infrastructure domain, which metric names are numerical and which are categorical. Choose one with `--profile` (or `SCRY_PROFILE`); the default is set in the file. Bring metrics whose names match a profile, or add your own profile.
