# Data contract

Scry works on one long-format metric table. Every data source normalizes to it, and your own data must match it.

## Canonical schema

| column | type | required | notes |
|---|---|---|---|
| `resource_id` | string | yes | the entity being monitored (host, pod, device) |
| `metric_name` | string | yes | the metric identifier |
| `timestamp` | timestamp (UTC) | yes | when the sample was taken |
| `value` | double | yes | the numeric value |
| `host_name` | string | no | display or host name |
| `datasource_instance` | string | no | source instance label |
| `datasource_name` | string | no | source or collector name |

One row per (resource, metric, timestamp). Numerical and categorical metrics both live here as `value`; which is which is decided by the feature profile (`config/features.yaml`), not by the table.

## Storage

Land this table as Parquet or CSV. Partitioning by time (`year=/month=/day=/hour=`) makes range scans cheaper but is not required. Point Scry at it with a URI; the scheme picks the backend:

- `data/metrics/**/*.parquet` (local)
- `s3://bucket/metrics/**/*.parquet` (S3, MinIO, Ceph)
- `gs://bucket/...` (GCS), `az://container/...` (Azure Data Lake)

## LogicMonitor (optional adapter)

The `scry[logicmonitor]` extra ships an exporter (`scripts/export_logicmonitor.py`) that writes this canonical table directly from the LogicMonitor REST API; see [ingestion.md](ingestion.md). No intermediate service is required. A legacy HttpIngest adapter that reads from an external `/api/ml/*` service also exists for backward compatibility and is slated for removal.
