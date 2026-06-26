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

## LogicMonitor (HttpIngest adapter)

Install `scry[logicmonitor]` and set `HTTPINGEST_URL`. Scry pulls training data from the HttpIngest ML endpoints documented in [data-contract.md](data-contract.md). This is one adapter, not a requirement.

## Feature profiles

`config/features.yaml` defines, per infrastructure domain, which metric names are numerical and which are categorical. Choose one with `--profile` (or `SCRY_PROFILE`); the default is set in the file. Bring metrics whose names match a profile, or add your own profile.
