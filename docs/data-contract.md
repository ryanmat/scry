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

## HttpIngest API contract (optional adapter)

The `logicmonitor` extra reads from an HttpIngest-compatible service over the endpoints below. Implement them and any backend can feed Scry the same way:

| endpoint | returns |
|---|---|
| `GET /api/ml/training-data` | `{ "data": [ {resource_hash or resource_id, metric_name, timestamp, value, attributes} ], "meta": {"total": N} }` |
| `GET /api/ml/inventory` | `{ "resources", "metrics", "time_range" }` |
| `GET /api/ml/profile-coverage` | per-profile coverage percentages |
| `GET /api/ml/quality` | freshness and gap metrics |

Records are normalized by `normalize_record`: `resource_hash` maps to `resource_id`, a JSON `attributes` blob supplies `host_name`/`datasource_instance`, and the value resolves across `value` / `value_double` / `value_int`.
