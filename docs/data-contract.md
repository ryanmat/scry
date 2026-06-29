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

The `scry[logicmonitor]` extra ships an exporter (`scripts/export_logicmonitor.py`) that writes this canonical table directly from the LogicMonitor REST API; see [ingestion.md](ingestion.md). No intermediate service is required.

## Feature schema and alignment

A trained model carries its feature schema in the checkpoint: the ordered numerical and categorical feature names, the profile they came from, and the per-feature normalization (numerical mean/std) and encoding (categorical min/max) params. The schema is the model's input contract.

At serve time, incoming metrics are aligned to the model's input columns by name, not by the order they arrive in. A feature the model expects but the request omits is filled with that feature's mean (a neutral input); a metric the model does not know is ignored. This makes coverage differences between ingestion routes a checked contract rather than a silent misalignment. A checkpoint that predates schema persistence has no names to align against and is rejected at load; retrain it to regenerate the schema. `/predict/lookup` groups a resource's metrics into numerical and categorical sets using the model's own training profile, so a mismatched `SCRY_PROFILE` cannot misalign the split.

## Coverage and quality

Before training or serving, check what the object store actually holds. Both are computed directly over the canonical data with DuckDB.

**Coverage** reports, per feature profile, which expected metric names are present: the intersection of the profile's features (from `config/features.yaml`) with the distinct `metric_name` values in the store, as `coverage_percent`, the `available` and `missing` names, and the totals. A profile with no required features is reported as fully covered.

**Quality** is assessed per `(resource_id, metric_name)` series:

- The sample interval is inferred as the median gap between consecutive timestamps, so no fixed cadence is assumed.
- `gap_score` is the mean per-series point density (observed over expected points across the series' span); the sparsest series are listed in `gaps`.
- `freshness_score` is the share of series whose last sample is recent relative to a reference time. The reference defaults to the dataset's own latest timestamp, not wall-clock now, so a static capture reads fresh rather than uniformly stale; `lag_seconds` reports wall-clock staleness separately. Pass a reference time explicitly to anchor freshness to now for a live store.
- Single-point and constant-timestamp series have no inferable interval and are excluded from scoring; an empty or unassessable store fails closed at zero.

Quality needs `timestamp` to arrive as a real timestamp type. Parquet preserves that; CSV relies on type inference, so Parquet is the better choice when you want the quality checks.

Run both with `python scripts/validate_data.py --data <uri> --profile <name>` (or set `SCRY_DATA_URI`).

## /predict/lookup resource keying

`resource_id` is the canonical key and is always present; `host_name` is optional. Lookup matching is case-insensitive and resolves in this order: exact `resource_id`, then exact `host_name`, then a substring fallback on either. If a lookup resolves to more than one distinct `resource_id` (for example, the substring `worker` across several worker nodes), the endpoint returns `409` with the candidate ids instead of pooling their metrics into one prediction. Pass an exact id to disambiguate.
