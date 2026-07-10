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

Two sizing rules before you induce:

- **Duration**: the validation's detection rule requires a *sustained* run --
  `sustain` consecutive windows over threshold, and only windows that *end* by the
  labeled incident end count toward it. Windows are `seq_len` samples long and
  slide by `window_step` samples, so the reconstruction error must stay over
  threshold for `sustain x window_step` samples of stride, after up to a full
  window of fill before the first window clears. At the defaults (seq_len 30,
  step 10, sustain 3, 2-minute samples) that is ~60 minutes of sustained
  elevation after up to 60 minutes of fill: plan the over-threshold portion at
  60 minutes or more and the whole induction at 120 or more, back-loaded. A
  30-minute burst can spike individual windows over threshold without ever
  forming a sustained run.
- **Placement**: never stress the node hosting the monitoring collector (for LM
  Container: the argus/collector, collectorset-controller, and kube-state-metrics
  pods). Starving the collector distorts every node's metrics in the capture.
  Pods reschedule, so verify placement the same day you induce:

  ```bash
  kubectl get pods -A -o wide | grep -iE 'argus|collectorset|kube-state-metrics'
  kubectl get pods -A --field-selector spec.nodeName=<target-node>
  ```

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
      args: ["--cpu", "3", "--vm", "1", "--vm-bytes", "1G", "--timeout", "5400s"]
      resources:
        requests: { cpu: "500m", memory: "512Mi" }
        limits: { cpu: "3", memory: "2Gi" }
```

Keep the requests small: a busy cluster reserves most of a node's allocatable
CPU and memory, and the kubelet rejects a pod whose requests do not fit
(`OutOfcpu`/`OutOfmemory`) -- check `kubectl describe node <node>` under
"Allocated resources" first. The limits, not the requests, set the stress
ceiling. Keep the total `--vm` bytes under the memory limit or the stressor
OOM-kills itself.

Rollback: `kubectl delete pod scry-stress`.

#### Gradual-ramp variant (preferred for lead-time measurement)

A step change gives near-zero lead time by construction; real failures creep.
Run the whole ramp as one self-contained pod so a dropped operator session,
expired login, or network blip cannot stall it mid-ramp; the step timestamps
come from the pod log afterwards.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: scry-ramp
spec:
  nodeName: <target-node>
  restartPolicy: Never
  activeDeadlineSeconds: 10800
  containers:
    - name: ramp
      image: ghcr.io/colinianking/stress-ng
      command: ["/bin/sh", "-c"]
      args:
        - |
          echo "RAMP_START $(date -u +%FT%TZ)"
          echo "STEP25 $(date -u +%FT%TZ)";  stress-ng --cpu 0 --cpu-load 25  --vm 1 --vm-bytes 1G --timeout 1200s
          echo "STEP50 $(date -u +%FT%TZ)";  stress-ng --cpu 0 --cpu-load 50  --vm 1 --vm-bytes 1G --timeout 1200s
          echo "STEP80 $(date -u +%FT%TZ)";  stress-ng --cpu 0 --cpu-load 80  --vm 1 --vm-bytes 1G --timeout 2400s
          echo "STEP100 $(date -u +%FT%TZ)"; stress-ng --cpu 0 --cpu-load 100 --vm 1 --vm-bytes 1G --timeout 3600s
          echo "RAMP_END $(date -u +%FT%TZ)"
      resources:
        requests: { cpu: "500m", memory: "512Mi" }
        limits:   { cpu: "4",    memory: "2Gi" }
```

- `--cpu 0` runs one worker per online CPU; the CPU *limit*, not the worker
  count, caps the delivered load. Limits are not admission-checked, so a limit
  equal to the node's CPU count lets the top step saturate the node; a lower
  limit tops out below it (a 3-of-4-CPU limit peaks near 88% node CPU and never
  crosses a 90% alert tier). Size the top step against the node's measured
  baseline and the alert thresholds you want crossed.
- The pod ends itself (`--timeout` per step plus `activeDeadlineSeconds` as the
  dead-man switch). Take the incident end from the final log line, then delete
  the pod and confirm the node returns to baseline before exporting.
- If the target node cannot pull from the public registry (node-local DNS
  failures surface as `ImagePullBackOff` / "server misbehaving"), import the
  image into the cluster's internal registry off-node and reference it there:

  ```bash
  oc import-image stress-ng --from=ghcr.io/colinianking/stress-ng --confirm \
      --reference-policy=local -n <namespace>
  # then set: image: image-registry.openshift-image-registry.svc:5000/<namespace>/stress-ng:latest
  ```

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

Record more than one onset candidate while the incident runs: the stressor start
(pod log), the first sample at or over the monitoring system's alert threshold
(read from the capture afterwards), and the monitoring system's alert timestamp
(which lags the actual crossing by its poll interval times its trigger count).
Re-running the validation per onset via `--data` is free, and the spread between
those lead times is part of the result, not noise to hide.

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
