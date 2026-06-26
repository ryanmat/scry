# Quickstart

End to end on synthetic data, no cloud.

```bash
# 1. install (core)
uv sync

# 2. the sample dataset ships in examples/sample_data/metrics.parquet
#    (regenerate or extend it any time)
python examples/generate_sample_data.py

# 3. extract windowed features (the sample is dated 2026-01-01)
python scripts/extract_features.py \
  --data "examples/sample_data/metrics.parquet" \
  --start 2026-01-01 --end 2026-01-02 \
  --profile kubernetes \
  --output data/training_data.npz

# 4. train
python scripts/train_model.py --data data/training_data.npz --output models/xdec_model.pt

# 5. serve
MODEL_PATH=models/xdec_model.pt uvicorn scry.api.main:app --port 8000
```

Then exercise the API:

```bash
curl localhost:8000/health
curl localhost:8000/clusters
```

Forecasting endpoints need the extra: `uv sync --extra forecast`. To bring your own data instead of the sample, see [../docs/data-contract.md](../docs/data-contract.md) and [../docs/ingestion.md](../docs/ingestion.md).
