# Capturing a labeled incident for validation

A keeper model trained on healthy data is a normal-behavior model: it learns what
normal looks like and flags deviation by reconstruction error. To prove it actually
detects degradation, and to measure how far ahead it fires, you need a capture that
contains a *known* incident with recorded timestamps. This runbook produces one by
inducing a controlled failure, capturing the window, and labeling it.

Do this in a non-production environment you control.

## What you are producing

1. A Parquet capture spanning **healthy lead-in -> incident -> recovery**. The
   lead-in is not optional: it gives the model baseline context and is what lets you
   measure *detection lead time* (how early the anomaly score crosses threshold
   relative to the incident).
2. A small labels sidecar recording the incident's start and end in UTC.

## Principle: move the metrics the profile actually trains on

Induce failures that move the features in *your* profile. The `aro_node` profile
trains on CPU, memory, and filesystem usage, so drive those. Node-condition flags
(`DiskPressure`, `MemoryPressure`, `ConditionReady`) are not in that profile, so do
not rely on them as the signal; they are a bonus if your profile includes them.

## Tiered induction (least to most disruptive)

Target a single node. Keep a rollback ready (delete the pod, remove the fill file,
uncordon). Record the UTC start before you induce and the UTC end after recovery.

### Tier 1 (safe): CPU + memory stress

A `stress-ng` pod pinned to one node moves `cpuUsageNanoCores`, `memoryUsageBytes`,
and `memoryAvailableBytes`. Limits keep the stressor self-bounded.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: scry-stress
spec:
  nodeName: <target-node>          # pin to the node you will capture
  restartPolicy: Never
  containers:
    - name: stress
      image: ghcr.io/colinianking/stress-ng
      args: ["--cpu", "4", "--vm", "2", "--vm-bytes", "2G", "--timeout", "1800s"]
      resources:
        limits: { cpu: "4", memory: "3Gi" }
```

Rollback: `kubectl delete pod scry-stress`.

### Tier 2 (moderate): filesystem fill

Fill a scratch volume on one non-critical node to move `fsUsedBytes` /
`fsAvailableBytes` (and trip `DiskPressure`, a bonus signal). Write to a quota'd or
clearly reclaimable path and delete the file to recover.

```bash
# inside a pod/host with a scratch mount on the target node
fallocate -l 20G /scratch/scry-fill.bin     # or: dd if=/dev/zero of=... bs=1M count=20000
# ... let it sit through a few collection intervals ...
rm /scratch/scry-fill.bin                    # rollback
```

### Tier 3 (disruptive, optional): node-level

`kubectl cordon <node>` then drain, or stop kubelet, on a throwaway node. Highest
blast radius; only on a node you can rebuild.

## Capture the window

`scripts/export_logicmonitor.py` pulls the canonical Parquet over an exact window.
`--start` / `--end` are epoch seconds, so bracket **[lead_in_start, incident_end +
tail]** with hours-to-days of healthy lead-in before the induction. Credentials come
from a gitignored `.env` (`LM_ACCESS_ID`, `LM_ACCESS_KEY`, `LM_COMPANY`).

```bash
# epoch-second bounds, e.g. via: date -u -d '2026-06-30 12:00:00' +%s
uv run python scripts/export_logicmonitor.py \
    --group-id <node-group-id> \
    --datasource-filter Kubernetes_KSM_Nodes \
    --datasource-filter Kubernetes_KSM_NodeSummary \
    --start <leadin_epoch_s> --end <incident_end_plus_tail_epoch_s> \
    --output data/captures/aro_incident_<date>.parquet
```

Use `--device-id <id>` (repeatable) instead of `--group-id` to target specific nodes.

## Label it

Write the incident window(s) to a sidecar next to the capture. Timestamps in UTC
ISO 8601; `resource_id` matches the capture's `resource_id` (the device displayName).

```json
[
  {
    "resource_id": "<node-display-name>",
    "type": "cpu_memory_stress",
    "start": "2026-06-30T12:00:00Z",
    "end": "2026-06-30T12:30:00Z"
  }
]
```

Save as `data/captures/aro_incident_labels.json`.

## Validate (after the keeper model exists)

`scripts/validate_incident.py` runs the capture through the keeper, computes
per-window reconstruction error, overlays the labeled window, and reports
detection lead time and threshold exceedance.

`scripts/capture_incident.py` wraps this whole runbook after the induction:
given `--onset`/`--incident-end` and a device target it exports the bracketed
window, writes the labels sidecar, runs the validation, and prints the lead
time in one command (`--data` re-validates an already-exported capture).

This validates the anomaly / reconstruction head and detection lead time. It does
not validate the five operational-state labels: a single incident cannot ground a
PRE_FAILURE cluster. It does let you inspect which cluster the incident windows fall
into.
