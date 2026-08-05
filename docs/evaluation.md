# Evaluation

Scry's detector is judged by a harness, not by inspection. You declare a suite of captures, a rubric of pass criteria, and a threshold policy; the harness scores every capture, resolves a threshold per resource, computes one metric bundle per case, evaluates the rubric, and writes a schema-validated JSON report. The report carries its own provenance, so a number can always be traced back to the model, the data, and the criteria that produced it.

No numbers ship in this repository. The harness runs on captures you supply.

## Running it

```
python scripts/run_eval.py --suite <suite.yaml> --output report.json
```

`--suite` is required. `--case NAME` is repeatable and runs only the named cases; an unknown name is an error rather than a silent no-op.

Output routing depends on `--output`. With it, the report JSON goes to the file and a human summary goes to stdout. Without it, the report JSON goes to stdout so it can be piped, and the summary goes to stderr so it cannot corrupt that JSON. The summary is one line per case per required gate, plus a final verdict line, in plain `PASS` / `FAIL` words.

## Exit codes

| code | meaning |
|---|---|
| 0 | Every required gate of every case passed. |
| 1 | At least one required gate failed. The report is still written and complete. |
| 2 | The suite or rubric is invalid, or a named case does not exist. |

Exit 2 emits no scoring output at all. A malformed suite can never be mistaken for a verdict, and a failed run always leaves a complete report behind to read.

## The suite

A suite is YAML. It names the candidate detector, the threshold policy, the rubric, and the cases. Every path resolves relative to the suite file's own directory, so a suite plus its data is portable.

```yaml
version: 1
suite: my_fleet
candidate:
  type: reconstruction
  model: keeper.pt
  profile: aro_node
threshold_policy:
  type: per_resource_margin
  calibration: data/healthy.parquet
  quantile: 0.99
  margin: 2.0
rubric: rubric.yaml
cases:
  - name: cpu_ramp
    kind: incident_capture
    capture: data/incident.parquet
    labels: data/labels_v2.json
  - name: quiet_week
    kind: healthy_reference
    capture: data/healthy_week.parquet
```

Cases come in two kinds. An `incident_capture` requires labels; a `healthy_reference` takes none. Validation collects every problem into a single error rather than failing on the first, so one run tells you everything wrong with the suite.

Labels are a v2 sidecar: each resource carries a role (`incident`, `negative_control`, or `excluded`) and, for incidents, named onsets (`T0`, `T1`, `T2`, `T2b`), which onset is primary, and an end. Onsets are timezone-aware UTC and are meant to be measured and transcribed from the incident record, not estimated.

Threshold policies are `global_override`, `reference_quantile`, `healthy_split`, `per_resource_margin`, and `serving_block`. The last reads the baked serving block out of a checkpoint and resolves exactly the way the serving path does, per-resource map first and the global as fallback, which is how you ask "what would the deployed configuration have done on this incident?"

## The rubric

A rubric is data, versioned separately from the code that reads it. It declares the detection semantics (run-forming mode, onset anchor, sustain, lead-in window, maximum credited lead time), the scoring grids, which grid is the headline, and the gates.

Seven gates: `no_pre_onset_bridging`, `detection_lead`, `lead_in_fpr`, `alarm_fatigue`, `negative_controls_clean`, `coverage_integrity`, and `sanity`. Each is `required` or not, may bind itself to a named grid, and carries its own parameters. A required gate that fails sets the verdict.

Gates are kind-scoped. The detection gates have nothing to evaluate on a healthy reference, and `alarm_fatigue` has nothing to evaluate on an incident, because its population is the resources carrying a time-in-alarm fraction. A required gate whose population is empty is an error unless it declares `allow_absent: true`, in which case it passes explicitly with the detail `no applicable cases`. Kind-scoped gates therefore need that flag or the rubric cannot evaluate a mixed suite at all. Gates that apply to every case kind, such as coverage and sanity, deliberately do not carry it: they must never pass vacuously.

## Grids and the reading convention

Every case is scored on every grid the rubric declares, and each gate result records which grid produced it. A rubric names one `headline_grid`, and that is the reading convention: unless a gate binds itself elsewhere, its verdict is the headline grid's verdict, and the threshold calibration is fit on the headline grid.

Two grids are worth declaring together. A coarse offline grid steps several samples at a time and is cheap to sweep. A serving grid reproduces the cadence the deployed service actually scores at. They do not always agree, and the disagreement is informative rather than embarrassing: a ramp that forms a sustained run at serving cadence may produce too few exceeding windows on a coarse grid to satisfy the same sustain, so the coarse grid reports no detection. Declaring both, and reading the headline, makes that comparison a checkable artifact instead of an argument.

## Both accountings, always

Offline gates conventionally reason at sustain 3, while a deployed service that scores a single latest window with no sustain is effectively sustain 1. Quoting one number invites comparing it to the other.

So every sustain-bearing field in the report carries both. False-positive rates, time-in-alarm fractions, raise and run rates, and run counts are all objects keyed `"3"` and `"1"`, and the schema validator rejects a report with either accounting stripped. A field that does not apply to a resource's role serializes as null under both keys rather than vanishing.

Time-in-alarm is duration-weighted rather than window-counted, and an alarm's duration includes one trailing grid step so a single-window alarm has non-zero length. A consequence worth knowing before it surprises you: a resource in alarm wall-to-wall reads slightly above 1.0, by exactly one grid step as a fraction of the observed span.

## The report

One JSON object: `provenance`, `suite`, `cases`, `verdict`, and `exit_code`. Each case entry carries its name, kind, the per-grid metrics, the rubric result, and a reported block for context that is recorded but never gates.

Provenance is what makes a number defensible later: the model path and its hash, the profile and sequence length, every grid definition, the sustain accountings, the detection mode, the resolved threshold policy including its per-resource thresholds, the rubric path and version, the package version, and per-case data identity. Two runs on the same inputs produce identical reports apart from the generation timestamp.

The report is validated against its schema before it is returned, so a report that exists is a report that conforms.

## Known divergences

A rubric carries a `known_divergences` list, and it is part of the deliverable rather than an apology. Anywhere the harness and the deployed service differ, or anywhere a measurement convention could be misread, is written down next to the criteria rather than discovered later.

The divergences worth carrying in any deployment of this shape: serving scoring a single window with no sustain while offline gates use sustain 3; any threshold-resolution difference between endpoints; an environment override that disables per-resource thresholds entirely; a serving window that refreshes on a different cadence than the collector polls, so alternate polls score identical windows; and any place a calibration utility and the harness split their data differently. Record yours in the rubric.

## What the harness does not do

It does not train, and it does not tune. It reads a checkpoint and a rubric and reports. Deciding that a gate's budget is right for your fleet is an operator judgment, and the honest way to make it is to run the harness against a healthy reference from your own environment and read what the gate observes before choosing the number.
