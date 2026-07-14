# Roadmap

Scry is bring-your-own-data: land metrics in the [canonical schema](data-contract.md), train on your own data, serve predictions. The object store is the single read path, and every source is an ingestion route that lands canonical data in it.

## Direction

- **Ingestion routes as adapters, not runtime couplings.** Each source (LogicMonitor REST, LogicMonitor Data Publisher, your own ETL) lands canonical Parquet; Scry reads the object store. The LogicMonitor REST exporter ships today; see [ingestion.md](ingestion.md).
- **Coverage and quality on canonical data.** Report which profile features are present or missing, and whether data is fresh and gap-free, before training or serving.
- **Models carry their feature schema.** Persist the trained feature names, order, and profile, and align incoming data by name, so coverage differences between routes are a checked contract rather than a silent failure.
- **Profiles can be purely numerical.** A categorical feature is not required; the X-DEC categorical branch is optional, so a numerical-only profile builds, trains, and serves.
- **Validate by detection lead time.** Score a trained model against a labeled incident capture: per-window reconstruction error thresholded on healthy windows only, reporting how early the alarm fires before onset. See [incident-capture-runbook.md](incident-capture-runbook.md) and `scripts/validate_incident.py`.

## Scaling architecture

How the single-fleet design extends, in order of increasing work. The unit of modeling
is a fleet of like resources, never a single resource and never a mixed bag.

1. **One model per fleet of like resources (current shape).** The model trains on the
   fleet jointly and learns the shared manifold of healthy behavior; windows are scored
   per resource against that shared manifold. Per-single-resource models would be
   data-starved; a shared model gets the whole fleet's history.
2. **Per-resource thresholds on the shared model.** A single global healthy quantile is
   set by the noisiest resource's tail, which both pages on that resource and dulls the
   signal on quiet ones (a quiet resource's incident can be 10x its own baseline but
   barely 3x the global cut). The other failure direction is measured too: a naive
   per-resource q99 drifts too tight within days. The design is a margin multiplier over
   each resource's own healthy quantile (`flag = error > m x own_q99`), swept offline
   against a healthy week (false-positive cost) and labeled incident captures (detection
   lead), with the global threshold as fallback for unknown resources. Static-with-margin
   first; rolling / EVT / conformal baselines are the adaptive follow-on.
3. **Multi-fleet, same resource type** (for example many Kubernetes clusters): works with
   the same profile and one model today. Requirements: resource_ids unique across fleets,
   capacity-relative features preferred (ratios transfer across heterogeneous hardware;
   absolute byte/core counts do not), and per-resource thresholds stop being optional as
   the healthy manifold widens. More fleets means strictly more training windows.
4. **Multi-domain** (Kubernetes + VMware + cloud accounts): model-per-domain plus a
   routing layer, not one cross-domain model. The domains have disjoint metric
   vocabularies, so a combined model is separate encoders in a trenchcoat with a shared
   failure mode. The new architecture work is a model registry and serving-time routing:
   resource type -> model -> per-resource threshold. Training per domain reuses the
   existing profile/train/bake/validate pipeline unchanged.

Operational notes at scale: serving currently loads one model per API process, so
multi-domain needs either one API instance per domain (works today, zero code) or the
registry above; the freshness export and the collector poll fan-out both grow linearly
with fleet size.

## Planned

- Per-resource serving thresholds (margin over own healthy quantile, global fallback).
- Profile reconciliation to live source metric names.
- GPU/cloud training for production models, including a cluster-count sweep and longer history.
- An MCP server exposing prediction and forecasting as agent tools (the reserved `mcp` extra).
