# Description: Tests for scry.eval.vus: the import contract, the scoring path, the degradation.
# Description: Pins what scry asks the backend for, the hand-checked VUS-PR value, a backend-free suite run.

"""Tests for the VUS-PR module's import contract, scoring path and degradation.

``scry.eval.vus`` consumes TheDatumOrg/VUS as an optional dependency, so every
test here forces the backend present or absent explicitly through
``sys.modules`` rather than depending on what the environment happens to have
installed: ``None`` under a key makes that import raise ImportError, a stub
module makes it succeed. Pinned: the module imports under both arms (a
module-scope ``import vus`` would fail the absent arm, a module-scope ``from
vus import ...`` the present one); the package imports it eagerly, without
torch and without reaching for the backend; the three states of the backend
(absent, installed but its scoring module unimportable, present) each answer
with their own object; what ``score_vus_pr`` asks the backend for and which key
it reads back; and a real suite run on a host with no backend produces a report
whose every grid carries the backend-unavailable reason under vus_pr.

The metrics wiring has its own class: the series ``compute_case_metrics`` asks
the backend to score and the windows it calls anomalous, the field degrading
with no backend, a failure INSIDE the scoring call becoming the reason rather
than ending the run, and no gate reading the field (spec 8.1).

The metric's own arithmetic cannot be checked against a stub, so it is pinned
in ``TestVusPrAgainstTheBackend``, which carries the ``vus_backend`` marker and
is DESELECTED by the default run (pyproject's ``addopts``): the backend is kept
out of uv.lock and an operator overlays it for that one test with
``uv run --all-extras --with vus python -m pytest -m vus_backend tests/test_eval_vus.py``
(``python -m``: the overlay is on the interpreter uv launches, not on the
project venv's ``pytest`` script). Everything else in this file runs in the
default suite with no backend present.
"""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from synth import write_run_suite_tree

import scry.eval
from scry.eval import vus
from scry.eval.candidate import ScoreSet
from scry.eval.labels import LabelCase, LabelSet
from scry.eval.metrics import CaseMetrics, compute_case_metrics
from scry.eval.rubric import GATE_NAMES, evaluate_rubric
from scry.eval.scoring import ScoringGrid
from scry.eval.suite import load_suite, run_suite

# A four-window series and its ground truth, fixed on 2026-10-27 as the
# argument every test here calls the entry point with. One false positive (0.8,
# at an unlabelled window) outranks one true positive (0.3), which is what
# keeps VUS-PR off 1.0 at every tolerance.
_SCORES = (0.1, 0.9, 0.3, 0.8)
_LABELS = (0, 1, 1, 0)

# VUS-PR over that fixture across tolerances w = 0..4, read on 2026-10-27 from
# upstream 0.0.6's own ``metricor().RangeAUC_volume_opt``.
_VUS_PR_PIN = 0.9080401067190556

# What the recording stub answers with: not the metric, just a number no other
# constant here could be mistaken for.
_STUB_VUS_PR = 0.4242

# A declared negative control of the incident's capture: two quiet windows,
# reported like any labelled resource. node-x is the same capture's UNLABELLED
# resource -- role "excluded", absent from per_resource -- and its windows
# would outrank every other if they leaked into the scored series.
_CONTROL_SCORES = (0.05, 0.07)
_UNLABELLED_SCORES = (9.0, 9.5)

_TEN_MIN_GRID = ScoringGrid(label="serving", step_samples=1, cadence=pd.Timedelta(minutes=10))
_WINDOW_ENDS = pd.date_range("2026-01-01T10:00:00Z", periods=4, freq="10min")


def _import_vus(monkeypatch: pytest.MonkeyPatch, backend: ModuleType | None) -> ModuleType:
    """Import ``scry.eval.vus`` fresh with the backend forced present or absent.

    CPython halts an import whose ``sys.modules`` entry is None, so that entry
    is the absent arm; a bare ``ModuleType`` is the present arm and carries no
    attributes, so a module-scope ``from vus import ...`` would fail on it.
    Both ``sys.modules`` entries are restored by monkeypatch at teardown, and
    so is the ``vus`` attribute on ``scry.eval``, which a fresh import rebinds
    and the ``sys.modules`` undo does not reach.
    """
    monkeypatch.setitem(sys.modules, "vus", backend)
    monkeypatch.setattr(scry.eval, "vus", vus)
    monkeypatch.delitem(sys.modules, "scry.eval.vus", raising=False)
    return importlib.import_module("scry.eval.vus")


def _install_backend(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Force a recording stub of the backend's ``vus.metrics`` into sys.modules.

    The stub's surface is upstream 0.0.6's, read from its source on 2026-10-27:
    ``get_metrics(score, labels, metric=..., slidingWindow=..., thre=...)``
    answering a dict of every metric it computed, keyed ``VUS_PR`` among
    others. It returns canned numbers, as numpy scalars the way upstream does,
    so a test over it pins what scry asks the backend for and never the
    metric, which only the real backend settles.

    Returns the call log: one dict per call, each argument under its name.
    """
    calls: list[dict[str, Any]] = []

    def get_metrics(score: Any, labels: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"score": score, "labels": labels, **kwargs})
        return {"VUS_PR": np.float64(_STUB_VUS_PR), "VUS_ROC": 0.1111, "AUC_PR": 0.2222}

    _put_backend(monkeypatch, get_metrics)
    return calls


def _put_backend(monkeypatch: pytest.MonkeyPatch, get_metrics: Any) -> None:
    """Force a two-module backend answering ``get_metrics`` into sys.modules."""
    backend = ModuleType("vus")
    metrics = ModuleType("vus.metrics")
    metrics.get_metrics = get_metrics
    backend.metrics = metrics
    monkeypatch.setitem(sys.modules, "vus", backend)
    monkeypatch.setitem(sys.modules, "vus.metrics", metrics)


def _case_metrics(incident: bool = True) -> CaseMetrics:
    """One case over the module fixture, with two more resources.

    node-a carries ``_SCORES`` on a ten-minute grid and, with ``incident``,
    an incident whose span (primary onset T1 at 10:10, end 10:20) is exactly
    the two windows ``_LABELS`` marks anomalous; T0 at 10:00 is the anchor
    onset and not the span's start, so the span follows the primary. Its rows
    reach the ScoreSet OUT of end order, so the scored order is the wiring's
    doing, not the input's. node-ctl and node-x carry two windows each over
    the first two ends. Without ``incident`` the control is the only labelled
    resource, the case is a control-only one, and every resource of the
    capture is reported (a control case has no excluded role), so each needs
    a threshold.
    """
    scrambled = [2, 0, 3, 1]
    errors = [_SCORES[i] for i in scrambled] + [*_CONTROL_SCORES, *_UNLABELLED_SCORES]
    scores = ScoreSet(
        errors=np.asarray(errors, dtype=float),
        end_times=_WINDOW_ENDS[scrambled].append(_WINDOW_ENDS[:2]).append(_WINDOW_ENDS[:2]),
        resource_ids=np.asarray(["node-a"] * 4 + ["node-ctl"] * 2 + ["node-x"] * 2),
        grid=_TEN_MIN_GRID,
        meta={},
    )
    labels = LabelSet(
        version=2,
        capture=None,
        cases=[
            LabelCase(
                resource_id="node-a",
                role="incident",
                type="cpu",
                onsets={"T0": _WINDOW_ENDS[0], "T1": _WINDOW_ENDS[1]},
                primary_onset="T1",
                end=_WINDOW_ENDS[2],
                notes=None,
            ),
            LabelCase(
                resource_id="node-ctl",
                role="negative_control",
                type=None,
                onsets={},
                primary_onset=None,
                end=None,
                notes=None,
            ),
        ][0 if incident else 1 :],
    )
    reported = ("node-a", "node-ctl") if incident else ("node-a", "node-ctl", "node-x")
    return compute_case_metrics(
        "vus_wiring",
        {"serving": scores},
        labels,
        {rid: (0.5, "per_resource") for rid in reported},
        mode="no_bridging",
        onset_anchor="T0",
        sustain=3,
        lead_in=pd.Timedelta(hours=1),
        max_leadtime=pd.Timedelta(hours=2),
    )


class TestImportContract:
    def test_the_module_imports_with_the_backend_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _import_vus(monkeypatch, None)

        assert callable(module.score_vus_pr)

    def test_the_module_imports_with_the_backend_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _import_vus(monkeypatch, ModuleType("vus"))

        assert callable(module.score_vus_pr)

    def test_the_package_exports_the_entry_point_eagerly_and_stays_torch_free(self) -> None:
        """The eval package root imports the module without torch or the backend.

        The subprocess runs in this environment, where the optional backend is
        not installed: a module-scope ``import vus`` would make importing
        ``scry.eval`` itself fail, which is the failure spec section 12 forbids.
        """
        assert "score_vus_pr" in scry.eval.__all__
        assert scry.eval.score_vus_pr is vus.score_vus_pr

        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scry.eval, sys; assert 'scry.eval.vus' in sys.modules; "
                "assert 'torch' not in sys.modules; assert 'vus' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_a_fresh_import_leaves_the_package_attribute_and_sys_modules_agreeing(self) -> None:
        """A re-import's teardown leaves the package attribute and sys.modules agreeing.

        A fresh import rebinds ``vus`` on the parent package as a side effect
        and monkeypatch's ``sys.modules`` undo does not reach it, so without
        the ``setattr`` in ``_import_vus`` the attribute and the ``sys.modules``
        entry would be left as two module objects with distinct constants --
        an order-dependent failure for whatever next compares them. The
        re-import runs inside its own monkeypatch context so the test stands
        alone under ``-k`` and in any order.
        """
        with pytest.MonkeyPatch.context() as mp:
            fresh = _import_vus(mp, None)
            assert fresh is not vus

        assert scry.eval.vus is sys.modules["scry.eval.vus"]
        assert scry.eval.vus is vus


class TestDegradation:
    """The two states in which there is no score, each with its own reason.

    The third state, a backend that answers, is ``TestTheScoringPath`` below.
    """

    def test_the_entry_point_degrades_when_the_backend_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "vus", None)

        result = vus.score_vus_pr(_SCORES, _LABELS)

        assert result == {"value": None, "reason": vus.REASON_BACKEND_UNAVAILABLE}
        assert "vus" in result["reason"]
        assert "not installed" in result["reason"]

    def test_the_entry_point_degrades_when_the_backend_cannot_import_its_scoring_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An installed-but-broken backend is not a missing one.

        Upstream's ``vus/__init__.py`` is empty, so ``import vus`` succeeds on
        a host that installed the backend without the dependencies its scoring
        module imports (statsmodels and arch among them). Telling that operator
        the package is not installed sends them to reinstall what they already
        have, so the reason names the failed import and carries its error text.
        """
        monkeypatch.setitem(sys.modules, "vus", ModuleType("vus"))
        monkeypatch.setitem(sys.modules, "vus.metrics", None)
        with pytest.raises(ImportError) as raised:
            importlib.import_module("vus.metrics")

        result = vus.score_vus_pr(_SCORES, _LABELS)

        assert result["value"] is None
        assert result["reason"].startswith(vus.REASON_BACKEND_UNIMPORTABLE)
        assert "vus.metrics" in result["reason"]
        assert str(raised.value) in result["reason"]
        assert "not installed" not in result["reason"]
        assert result["reason"] != vus.REASON_BACKEND_UNAVAILABLE


class TestTheScoringPath:
    def test_the_backend_is_asked_for_vus_over_the_declared_tolerance_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What scry asks the backend for, and what it reads back.

        The stub answers canned numbers, so this pins the plumbing and not the
        metric: the arguments reaching ``get_metrics`` (the sequences as numpy
        arrays, since upstream negates and sorts them), ``VUS_PR`` as the key
        read out of the several returned, and the object built from it, the
        backend's numpy scalar coerced to a plain float. ``slidingWindow`` is
        upstream's name for the top of the tolerance range; it is asserted as
        the literal 4 because the declared range w = 0..4 is part of the
        metric's definition, and moving it moves the number
        ``TestVusPrAgainstTheBackend`` pins.
        """
        calls = _install_backend(monkeypatch)

        result = vus.score_vus_pr(_SCORES, _LABELS)

        assert len(calls) == 1
        call = calls[0]
        assert isinstance(call["score"], np.ndarray)
        assert call["score"].dtype == float
        assert isinstance(call["labels"], np.ndarray)
        assert call["labels"].dtype == int
        assert call["score"].tolist() == list(_SCORES)
        assert call["labels"].tolist() == list(_LABELS)
        assert {k: v for k, v in call.items() if k not in ("score", "labels")} == {
            "metric": "vus",
            "slidingWindow": 4,
        }
        assert result == {"value": _STUB_VUS_PR, "reason": None}
        assert type(result["value"]) is float


def _raise_value_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise ValueError("upstream refused the window geometry")


def _every_gate_rubric() -> dict[str, Any]:
    """Every gate of the vocabulary, required, each evaluable on the fixture."""
    params = {
        "lead_in_fpr": {"max_fraction": 1.0, "min_eval_windows": 0},
        "alarm_fatigue": {
            "max_time_in_alarm_fraction_per_resource": 1.0,
            "max_fleet_raises_per_week": 1000.0,
        },
    }
    return {
        "version": 1,
        "headline_grid": "serving",
        "gates": {
            name: {"required": True, "allow_absent": True, **params.get(name, {})}
            for name in GATE_NAMES
        },
        "reported": {"vus_pr": {}},
    }


class TestTheMetricsWiring:
    def test_the_grid_carries_the_score_over_its_resources_in_id_and_time_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One call per grid, over the reported windows, labelled by the span.

        The stub answers a canned number, so what this pins is the series the
        wiring builds: the resources the grid reports on in sorted id order
        (unlabelled node-x is not one), each one's windows in end-time order
        (the ScoreSet hands them scrambled), and ground truth 1 exactly inside
        the incident's [primary onset, end] span -- the control is negatives.
        """
        calls = _install_backend(monkeypatch)

        grid = _case_metrics().grids["serving"]

        assert grid.vus_pr == {"value": _STUB_VUS_PR, "reason": None}
        assert len(calls) == 1
        assert calls[0]["score"].tolist() == [*_SCORES, *_CONTROL_SCORES]
        assert calls[0]["labels"].tolist() == [*_LABELS, 0, 0]

    def test_the_grid_degrades_with_the_backend_absent_and_the_placeholder_is_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No backend: the field carries the backend's reason, not "not implemented"."""
        monkeypatch.setitem(sys.modules, "vus", None)

        grid = _case_metrics().grids["serving"]

        assert grid.vus_pr == {"value": None, "reason": vus.REASON_BACKEND_UNAVAILABLE}
        assert "not implemented" not in grid.vus_pr["reason"]

    @pytest.mark.parametrize(
        ("answer", "raised"),
        [
            (_raise_value_error, "ValueError: upstream refused the window geometry"),
            (lambda *a, **k: {"VUS_ROC": 0.5}, "KeyError: 'VUS_PR'"),
        ],
        ids=["the backend raises", "the backend answers without VUS_PR"],
    )
    def test_a_failure_inside_the_scoring_call_becomes_the_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        answer: Any,
        raised: str,
    ) -> None:
        """A backend that is present and broken degrades the field, not the run.

        ``score_vus_pr`` catches ImportError and nothing else, so a backend
        that raises on the input it was handed, or answers without the key
        scry reads, arrives here. Reported and never gating (spec 8.1): the
        exception becomes the reason, its traceback goes to the log, and the
        rest of the bundle is computed.
        """
        _put_backend(monkeypatch, answer)

        with caplog.at_level(logging.WARNING, logger="scry.eval.metrics"):
            grid = _case_metrics().grids["serving"]

        assert grid.vus_pr == {"value": None, "reason": f"{vus.REASON_SCORING_FAILED}{raised}"}
        assert grid.n_eval_windows == 6
        assert set(grid.per_resource) == {"node-a", "node-ctl"}
        records = [r for r in caplog.records if r.name == "scry.eval.metrics"]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert records[0].exc_info is not None
        assert raised.split(":")[0] == records[0].exc_info[0].__name__

    def test_a_series_with_no_positive_window_is_undefined_and_never_reaches_the_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A control-only case has no positive, so the metric is undefined.

        Upstream raises on an all-negative label array, so the wiring answers
        with the undefined reason before the call: the backend records no
        call, and the reason is not a scoring failure.
        """
        calls = _install_backend(monkeypatch)

        grid = _case_metrics(incident=False).grids["serving"]

        assert grid.vus_pr == {"value": None, "reason": vus.REASON_NO_POSITIVE_WINDOWS}
        assert calls == []
        assert set(grid.per_resource) == {"node-a", "node-ctl", "node-x"}


class TestTheRubricReportsAndNeverGates:
    def test_no_gate_reads_the_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every gate returns the same verdict whether or not vus_pr has a value.

        Spec 8.1 makes VUS-PR reported: it belongs to the rubric's ``reported``
        block and no gate may consult it, so two bundles differing in nothing
        but that field must be indistinguishable to every gate.
        """
        with pytest.MonkeyPatch.context() as present:
            _install_backend(present)
            scored = _case_metrics()
        monkeypatch.setitem(sys.modules, "vus", None)
        degraded = _case_metrics()
        assert scored.grids["serving"].vus_pr != degraded.grids["serving"].vus_pr

        rubric = _every_gate_rubric()
        scored_result = evaluate_rubric(rubric, scored)
        degraded_result = evaluate_rubric(rubric, degraded)

        assert [gate.name for gate in scored_result.gates] == list(GATE_NAMES)
        assert scored_result.gates == degraded_result.gates
        assert scored_result.passed == degraded_result.passed
        evaluated = repr(scored_result.gates)
        assert "vus_pr" not in evaluated
        assert str(_STUB_VUS_PR) not in evaluated
        assert "vus_pr" in scored_result.reported


@pytest.mark.vus_backend
class TestVusPrAgainstTheBackend:
    """The metric itself, against the real backend. DESELECTED by default.

    The default run deselects this (pyproject's ``addopts``) rather than
    skipping it: the backend is not in uv.lock and installing it would put 31
    packages, TensorFlow among them, into every sandbox and CI job for one
    reported, non-gating metric. Selected without the overlay that supplies it,
    the test fails on the backend import, which is the honest answer: the pin
    was not checked.
    """

    def test_vus_pr_over_the_four_window_fixture(self) -> None:
        """VUS-PR of the module fixture is 0.9080401067190556, checkable by hand.

        Scores (0.1, 0.9, 0.3, 0.8) against labels (0, 1, 1, 0). VUS-PR is the
        mean, over tolerances w = 0..4, of the average precision at each
        tolerance, where a tolerance of w extends each labelled region by
        w // 2 windows on each side with partial credit sqrt(1 - d / w) at
        distance d (so the labels read (0, 1, 1, 0) at w = 0 and 1, and about
        (0.71, 1, 1, 0.71) at w = 2). Those five average precisions are

            w = 0  0.833333
            w = 1  0.833333
            w = 2  0.940185
            w = 3  0.961639
            w = 4  0.971710

        summing to 4.540200, and 4.540200 / 5 = 0.908040. None reaches 1.0
        because the false positive at window 3 (score 0.8) outranks the true
        positive at window 2 (score 0.3), and the two low tolerances agree
        because 1 // 2 is 0: w = 1 extends nothing. The mean moves
        with the range -- 0.833333 over w = 0..1 -- so this pins the range too.
        """
        importlib.import_module("vus.metrics")

        result = vus.score_vus_pr(_SCORES, _LABELS)

        assert result["reason"] is None
        assert result["value"] == pytest.approx(_VUS_PR_PIN, abs=1e-12)


class TestSuiteRunWithoutTheBackend:
    def test_a_suite_run_completes_and_every_grid_carries_null_with_reason(
        self, keeper_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A host without the backend still gets a report.

        vus_pr is reported, never gating, so its absence degrades to a stated
        reason rather than failing the run. The metrics wiring owns the field,
        so the reason on every grid of both cases -- the incident and the
        healthy reference -- is the backend's own, and no grid of the report
        carries "not implemented".
        """
        monkeypatch.setitem(sys.modules, "vus", None)

        report = run_suite(load_suite(write_run_suite_tree(tmp_path, keeper_path)))

        assert [case["name"] for case in report["cases"]] == ["ramp_incident", "pinned_alarm"]
        for case in report["cases"]:
            assert case["grids"]
            for grid in case["grids"].values():
                assert grid["vus_pr"] == {"value": None, "reason": vus.REASON_BACKEND_UNAVAILABLE}
        assert "not implemented" not in json.dumps(report)
