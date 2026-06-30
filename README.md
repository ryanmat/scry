<div align="center">

<img src="assets/banner.svg" alt="scry" width="400">

[![CI](https://github.com/ryanmat/scry/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanmat/scry/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/scryml.svg)](https://pypi.org/project/scryml/)
[![Python](https://img.shields.io/pypi/pyversions/scryml.svg)](https://pypi.org/project/scryml/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

**Scry predicts infrastructure failure states from a stream of metrics.** It sorts each resource into one of five operational states, recommends an action for each, and forecasts where the metrics are headed. It is data-source agnostic and runs offline. You bring your own metrics, train your own model, and serve predictions over a small HTTP API. No trained weights ship.

LogicMonitor is one supported source, not the anchor. The default path reads Parquet or CSV from local files or object storage. Everything normalizes to one canonical long-format table, so any metric source works once it is in that shape.

<table>
<tr><td><b>The X-DEC model</b></td><td>A dual-encoder temporal VAE plus deep embedded clustering, pure PyTorch, no cloud dependencies. Trains on your own windowed metrics from scratch.</td></tr>
<tr><td><b>Five operational states</b></td><td>Every resource is sorted into NORMAL, PRE_SCALE, PRE_FAILURE, ACTIVE_DEGRADATION, or ANOMALY, each mapped to a recommended action and priority.</td></tr>
<tr><td><b>Forecasting</b></td><td>An optional Chronos layer (<code>scryml[forecast]</code>) projects where each metric is headed across multiple horizons, kept behind an extra so the core stays offline-capable.</td></tr>
<tr><td><b>Data-source agnostic</b></td><td>Read Parquet or CSV from local disk or object storage (S3, GCS, ADLS, MinIO) through DuckDB. A LogicMonitor REST exporter lives behind <code>scryml[logicmonitor]</code>.</td></tr>
<tr><td><b>Coverage and quality checks</b></td><td>Before training or serving, report which profile features are present or missing and whether the data is fresh and gap-free, computed directly over the object store with <code>scripts/validate_data.py</code>.</td></tr>
<tr><td><b>Incident-validation harness</b></td><td>Score a trained model against a labeled incident capture: per-window reconstruction error thresholded on healthy data only, reporting detection lead time (how early the alarm fires before onset) with <code>scripts/validate_incident.py</code>.</td></tr>
<tr><td><b>A small HTTP service</b></td><td>FastAPI endpoints for prediction, forecasting, drift, anomaly, and accuracy: <code>/predict</code>, <code>/predict/lookup</code>, <code>/forecast</code>, <code>/drift</code>, <code>/anomaly</code>, <code>/accuracy</code>.</td></tr>
<tr><td><b>Schema-checked serving</b></td><td>Each trained model carries its feature schema (names, order, profile, normalization). The API aligns incoming metrics by name, so a coverage gap between sources is a checked contract, not a silent misalignment.</td></tr>
<tr><td><b>Bring your own data</b></td><td>One canonical schema: resource, metric, timestamp, value, plus optional host and datasource fields. Drop your metrics into that table and train. Profiles may be purely numerical or mix numerical and categorical features. No real telemetry or weights are included.</td></tr>
</table>

---

## Install

```bash
pip install scryml                  # core: the model and the API
pip install "scryml[forecast]"      # add Chronos forecasting
pip install "scryml[logicmonitor]"  # add the LogicMonitor REST exporter
```

Or from source with every extra:

```bash
git clone https://github.com/ryanmat/scry && cd scry
uv sync --all-extras
```

## Quickstart

End to end on the bundled synthetic sample, no cloud:

```bash
# extract windowed features (the sample is dated 2026-01-01)
python scripts/extract_features.py --data examples/sample_data/metrics.parquet \
  --start 2026-01-01 --end 2026-01-02 --profile kubernetes \
  --output data/training_data.npz

# train a model, then serve it
python scripts/train_model.py --data data/training_data.npz --output models/xdec_model.pt
MODEL_PATH=models/xdec_model.pt uvicorn scry.api.main:app --port 8000
```

```bash
curl localhost:8000/health
curl localhost:8000/clusters
```

Full walkthrough, including a `/predict` call: [examples/quickstart.md](examples/quickstart.md).

## Documentation

- [Architecture](docs/architecture.md): the model, the data seam, and how the pieces fit together.
- [Data contract](docs/data-contract.md): the canonical metric schema.
- [Ingestion](docs/ingestion.md): the object store and the routes that land data in it.
- [Roadmap](docs/roadmap.md): where Scry is headed.
- [Training](docs/training.md): training locally or on your own orchestrator.

Built with `uv`, `ruff`, and `pytest` on Python 3.10 and up.

## License

Apache-2.0.

WAKE UP TO FIND OUT THAT YOU ARE THE EYES OF THE WORLD
