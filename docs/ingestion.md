# Getting data in

Scry is bring-your-own-data. Produce the canonical table (see [data-contract.md](data-contract.md)) and point Scry at it. There are two ways in.

## Object storage (default)

Read Parquet or CSV directly through DuckDB. One reader, any backend, selected by the URI scheme:

```bash
export SCRY_DATA_URI="data/metrics/**/*.parquet"            # local
export SCRY_DATA_URI="s3://bucket/metrics/**/*.parquet"     # S3, MinIO, Ceph
export SCRY_DATA_URI="az://container/metrics/**/*.parquet"  # Azure Data Lake
export SCRY_DATA_URI="gs://bucket/metrics/**/*.parquet"     # GCS
```

Cloud credentials come from the standard environment for that cloud (`AWS_*`, the Azure credential chain, and so on). Install the matching extra: `scry[s3]` or `scry[azure]`. Local files need nothing. CSV works too; pass `--format csv` if the extension is ambiguous.

## LogicMonitor (REST exporter)

Install `scry[logicmonitor]`. `scripts/export_logicmonitor.py` (module `scry.data.sources.lm_export`) pulls device metrics straight from the LogicMonitor REST API with LMv1 auth and writes the canonical table; then point Scry at the resulting Parquet as object storage.

```bash
uv run python scripts/export_logicmonitor.py \
  --device-id <id> --datasource-filter <name> --days 14 \
  --output data/metrics.parquet
```

Select targets with `--device-id`, `--group-id`, or `--name-filter`. Credentials come from `LM_ACCESS_ID` / `LM_ACCESS_KEY` / `LM_COMPANY` (a gitignored `.env` works). The REST API carries full coverage, including cloud and computed datapoints and historical backfill, but is rate limited to about 500 GET/min per portal. LogicMonitor Data Publisher is a second route (real-time push over Kafka or HTTPS, normal collector datapoints only, no backfill) that lands in the same object store. Either way, the object store stays the single read path.

## Feature profiles

`config/features.yaml` defines, per infrastructure domain, which metric names are numerical and which are categorical. Choose one with `--profile` (or `SCRY_PROFILE`); the default is set in the file. Bring metrics whose names match a profile, or add your own profile.
