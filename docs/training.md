# Training

No weights ship. You train on your own data.

## 1. Get data in the canonical schema

See [data-contract.md](data-contract.md) and [ingestion.md](ingestion.md). To try it immediately with synthetic data:

```bash
python examples/generate_sample_data.py   # writes examples/sample_data/metrics.parquet
```

## 2. Extract windowed features

```bash
python scripts/extract_features.py \
  --data "examples/sample_data/metrics.parquet" \
  --start 2026-01-01 --end 2026-01-02 \
  --profile kubernetes \
  --output data/training_data.npz
```

Use a time range that covers your data: ISO timestamps, or relative like `--start 7d --end now`. (The synthetic sample is dated 2026-01-01, hence the explicit range above.)

## 3. Train

```bash
python scripts/train_model.py --data data/training_data.npz --output models/xdec_model.pt
```

Training runs in two stages (VAE pretraining, then deep embedded clustering). The model is small and trains on CPU or a single GPU. Tune with flags (`--help`) or `config/config.yaml`.

## Running on your own infrastructure

`extract_features` and `train_model` are plain scripts that read a data URI and write an artifact. Containerize them (see the `Dockerfile`) and run on whatever you have: a Kubernetes Job, SageMaker, Vertex AI, Azure ML. Scry does not bundle a vendor-specific training pipeline.
