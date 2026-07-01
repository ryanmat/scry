# Scry - LogicModules

LogicModule components that surface Scry's predictions inside LogicMonitor. The
collector polls the Scry API per device and stores the results as native
datapoints, so a "Scry AnomalyScore" graph appears on the device and alerts
through LM's normal alerting, with no separate UI.

## Contents

| File | Type | Endpoint | Purpose |
|------|------|----------|---------|
| `datasources/Scry_Anomaly.xml` | DataSource (SCRIPT) | `GET /anomaly/reconstruction/lookup` | Per-resource X-DEC reconstruction anomaly: `AnomalyScore` is the reconstruction error as a ratio of the model's healthy threshold. |
| `datasources/Scry_Predictive.xml` | DataSource (SCRIPT) | `GET /predict/lookup` | Five-state operational cluster, confidence, priority. Cluster-derived alerts are disabled for the healthy-trained keeper (degenerate clustering); the reconstruction signal is authoritative. |
| `datasources/Scry_Drift.xml` | DataSource (SCRIPT) | `GET /drift` | Feature/prediction drift status. |
| `datasources/Scry_Accuracy.xml` | DataSource (SCRIPT) | `GET /accuracy` | Forecast accuracy and cluster-stability metrics. |
| `dashboards/Scry_Predictive_Dashboard.json` | Dashboard | - | Prediction overview. |
| `dashboards/Scry_ModelHealth_Dashboard.json` | Dashboard | - | Model-health overview. |
| `propertysources/Scry_Predictive_Props.xml` | PropertySource | - | Discovery: sets Scry device properties. |
| `external_alert_handler.ps1` | External alerting | - | Interim alert handler until a RemediationSource exists. |
| `REMEDIATION_DESIGN.md` | Design document | - | RemediationSource design (not implemented). |

## Device properties

The DataSources apply to devices where `scry.enabled == "true"` and reach the API
through `scry.api.url`:

| Property | Example | Purpose |
|----------|---------|---------|
| `scry.enabled` | `true` | Gates the Scry DataSources (see each `appliesTo`). |
| `scry.api.url` | `https://scry.example.com` | Base URL the collector calls. |

The collector identifies the resource by `system.displayName`, which the lookup
endpoints resolve against the configured data source (exact `resource_id`, then
`host_name`, then a substring fallback).

## Reconstruction anomaly signal

`Scry_Anomaly` polls `GET /anomaly/reconstruction/lookup?resource_id=<displayName>`
and reads `ratio` into the `AnomalyScore` datapoint. The ratio is the per-window
reconstruction error divided by the model's healthy threshold, so it is directly
alertable: `> 1.0 warn`, `> 2.0 error` (unchanged from the datapoint's existing
thresholds). `ratio` is `null` until a healthy threshold is baked into the model
(`scripts/bake_serving_threshold.py`); the collector treats a null ratio as 0 so
it never alarms on an unthresholded model.

## Installation

1. Set `scry.enabled=true` and `scry.api.url=<url>` on the target devices (or a
   device group). The PropertySource can set these during discovery.
2. Import the DataSource XMLs (Settings > LogicModules > Import), then the
   dashboards and PropertySource.
3. Apply to the devices and confirm datapoints flow. Tune the `AnomalyScore`
   alert thresholds against a healthy baseline before relying on them.

## API reference

Base URL: `scry.api.url` (for example `https://scry.example.com`).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health`, `/health/detailed` | Liveness and model/threshold metadata. |
| POST | `/predict` | Cluster prediction from metrics in the body. |
| GET | `/predict/lookup?resource_id=<id>` | Cluster prediction from the data source. |
| POST | `/anomaly/reconstruction` | Reconstruction score from metrics in the body. |
| GET | `/anomaly/reconstruction/lookup?resource_id=<id>` | Reconstruction score from the data source. |
| POST | `/forecast` | Chronos metric forecast (needs the `forecast` extra). |
| GET | `/drift`, `/anomaly`, `/accuracy`, `/clusters` | Drift, forecast-anomaly, accuracy, cluster definitions. |

Reconstruction response:

```json
{
    "resource_id": "node-a",
    "reconstruction_error": 0.134,
    "threshold": 0.121,
    "ratio": 1.11,
    "is_anomaly": true,
    "severity": 2,
    "coverage": 1.0,
    "timestamp": "2026-06-30T00:00:00+00:00"
}
```
