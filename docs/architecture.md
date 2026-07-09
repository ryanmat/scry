# Architecture

Scry reads metrics, learns what healthy looks like, scores each resource's reconstruction error against a threshold calibrated on healthy data, and optionally forecasts where the metrics are headed. Deep embedded clustering additionally assigns each resource a nominal operational state.

## Components

- `src/scry/model/` - X-DEC: two encoders (one for numerical time series, one for categorical) feed a variational autoencoder; deep embedded clustering groups the latent space into five nominal operational states. The reconstruction core (`reconstruction.py`, `checkpoint.py`) derives healthy thresholds and per-window errors for the anomaly signal. Includes drift detection and an optional Chronos forecasting layer. Pure PyTorch, no cloud.
- `src/scry/data/` - feature engineering and the windowing pipeline, plus the data-source seam in `data/sources/`: the `DataSource` interface (`base.py`), the object-store reader (`object_store.py`), and the LogicMonitor REST exporter (`lm_export.py`).
- `src/scry/api/` - a FastAPI service: `/anomaly/reconstruction` (+ `/lookup`), `/predict` (+ `/lookup`), `/forecast`, plus `/health` and `/clusters`. `/drift`, `/anomaly` (forecast-based), and `/accuracy` need operator-wired detectors and serve 503 until configured; `/health/detailed` reports the wiring flags.
- `config/features.yaml` - per-domain feature profiles. `config/config.yaml` - model and training defaults.

## The reconstruction anomaly signal

The validated serving signal. `scripts/bake_serving_threshold.py` persists a healthy-quantile threshold into the checkpoint; `/anomaly/reconstruction` scores the latest full window through the same windowing path the threshold was calibrated on and returns the error as a ratio of that threshold. Windows shorter than `seq_len` or without numerical coverage are reported not-scored rather than padded, so cold starts and collection outages never fabricate a score. `scripts/validate_incident.py` measures detection lead time against a labeled incident capture.

## The five states

`NORMAL`, `PRE_SCALE`, `PRE_FAILURE`, `ACTIVE_DEGRADATION`, `ANOMALY`. Each carries a recommended action and priority; see the `/clusters` endpoint. The states are discovered by unsupervised clustering and the labels are nominal: they are not yet grounded on labeled incidents, and a model trained only on healthy data partitions healthy operating space rather than failure modes. Treat cluster output as exploratory until validated against real incidents; the reconstruction signal above is the alerting path.

## Flow

```
metrics (canonical schema)
  -> feature engineering (pivot to wide, filter to the active profile, window, normalize/encode)
  -> X-DEC training (VAE pretrain, then deep embedded clustering)
  -> a .pt model artifact (carries its feature schema)
  -> served over the API
```

The `.pt` artifact carries its feature schema: the ordered numerical and categorical feature names, the profile, and the per-feature normalization and encoding params. At serve time incoming metrics are aligned to the model's input columns **by name**, not by position. Features the model expects but the request omits are treated as missing (filled with the feature mean); metrics the model does not know are ignored. A checkpoint without a feature schema is refused at load, because it cannot be aligned safely.

`/predict` takes a window of metrics in the request body and is fully self-contained. `/predict/lookup` pulls a resource's recent metrics through the configured data source first, then predicts. Lookup resolves the resource by exact match on `resource_id` (then `host_name`), falling back to a substring match; if a lookup matches more than one resource it returns `409` with the candidate ids rather than blending them into one prediction.
