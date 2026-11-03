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
whose every grid carries a null-with-reason vus_pr. The metrics wiring that
replaces that field's placeholder reason lands in a later step -- the suite
test pins the shape, not the wording.

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
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from synth import write_run_suite_tree

import scry.eval
from scry.eval import vus
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

    backend = ModuleType("vus")
    metrics = ModuleType("vus.metrics")
    metrics.get_metrics = get_metrics
    backend.metrics = metrics
    monkeypatch.setitem(sys.modules, "vus", backend)
    monkeypatch.setitem(sys.modules, "vus.metrics", metrics)
    return calls


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
        reason rather than failing the run. The reason's wording belongs to
        whichever step owns the field -- the placeholder's today, the
        backend-unavailable one once the metrics wiring lands -- so this pins
        the null-with-reason shape on every grid of every case and the run
        completing, not the text.
        """
        monkeypatch.setitem(sys.modules, "vus", None)

        report = run_suite(load_suite(write_run_suite_tree(tmp_path, keeper_path)))

        assert [case["name"] for case in report["cases"]] == ["ramp_incident", "pinned_alarm"]
        for case in report["cases"]:
            assert case["grids"]
            for grid in case["grids"].values():
                assert grid["vus_pr"]["value"] is None
                assert isinstance(grid["vus_pr"]["reason"], str)
                assert grid["vus_pr"]["reason"]
