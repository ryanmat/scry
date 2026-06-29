# Roadmap

Scry is bring-your-own-data: land metrics in the [canonical schema](data-contract.md), train on your own data, serve predictions. The object store is the single read path, and every source is an ingestion route that lands canonical data in it.

## Direction

- **Ingestion routes as adapters, not runtime couplings.** Each source (LogicMonitor REST, LogicMonitor Data Publisher, your own ETL) lands canonical Parquet; Scry reads the object store. The LogicMonitor REST exporter ships today; see [ingestion.md](ingestion.md).
- **Coverage and quality on canonical data.** Report which profile features are present or missing, and whether data is fresh and gap-free, before training or serving.
- **Models carry their feature schema.** Persist the trained feature names, order, and profile, and align incoming data by name, so coverage differences between routes are a checked contract rather than a silent failure.

## Planned

- By-name feature alignment and model feature-schema persistence; exact-match resource keying.
- Coverage and quality checks computed over the object store.
- A LogicMonitor Data Publisher (OTLP) normalizer and ingestion-route documentation.
- Optional purely-numerical profiles, so a categorical feature is not required.
- Profile reconciliation to live source metric names.
- GPU/cloud training for production models, including a cluster-count sweep and longer history.
- An MCP server exposing prediction and forecasting as agent tools (the reserved `mcp` extra).
