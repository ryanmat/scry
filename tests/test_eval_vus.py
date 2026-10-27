# Description: Tests for scry.eval.vus: the import contract and the backend-absent degradation.
# Description: Pins the deferred backend import, the null-with-reason shapes, and a backend-free suite run.

"""Tests for the VUS-PR module's import contract and its degradation.

``scry.eval.vus`` consumes TheDatumOrg/VUS as an optional dependency, so every
test here forces the backend present or absent explicitly through
``sys.modules`` rather than depending on what the environment happens to have
installed: ``None`` under the key makes ``import vus`` raise ImportError, an
empty stub module makes it succeed while supplying no attributes. Pinned: the
module imports under both arms (a module-scope ``import vus`` would fail the
absent arm, a module-scope ``from vus import ...`` the present one); the
package imports it eagerly, without torch and without reaching for the
backend; ``score_vus_pr`` answers null-with-reason naming the missing backend
rather than raising, and null-with-reason for the unlanded scoring path when
the backend is there; and a real suite run on a host with no backend produces
a report whose every grid carries a null-with-reason vus_pr. The scoring path
itself, and the metrics wiring that replaces that field's placeholder reason,
land in later steps -- the suite test pins the shape, not the wording.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from synth import write_run_suite_tree

from scry.eval import vus
from scry.eval.suite import load_suite, run_suite

# A four-window series and its ground truth: enough to call the entry point
# with real arguments. The values it scores them into are Step 60's fixture.
_SCORES = (0.1, 0.9, 0.8, 0.2)
_LABELS = (0, 1, 1, 0)


def _import_vus(monkeypatch: pytest.MonkeyPatch, backend: ModuleType | None) -> ModuleType:
    """Import ``scry.eval.vus`` fresh with the backend forced present or absent.

    CPython halts an import whose ``sys.modules`` entry is None, so that entry
    is the absent arm; a bare ``ModuleType`` is the present arm and carries no
    attributes, so a module-scope ``from vus import ...`` would fail on it.
    Both ``sys.modules`` entries are restored by monkeypatch at teardown.
    """
    monkeypatch.setitem(sys.modules, "vus", backend)
    monkeypatch.delitem(sys.modules, "scry.eval.vus", raising=False)
    return importlib.import_module("scry.eval.vus")


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
        import scry.eval

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


class TestDegradation:
    def test_the_entry_point_degrades_when_the_backend_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "vus", None)

        result = vus.score_vus_pr(_SCORES, _LABELS)

        assert result == {"value": None, "reason": vus.REASON_BACKEND_UNAVAILABLE}
        assert "vus" in result["reason"]
        assert "not installed" in result["reason"]

    def test_the_entry_point_degrades_when_the_scoring_path_has_not_landed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An installed backend is not yet a score, and is not an exception either.

        The computation lands in a later step; until then the two states stay
        distinguishable, so an operator who installed the backend is not told
        it is missing.
        """
        monkeypatch.setitem(sys.modules, "vus", ModuleType("vus"))

        result = vus.score_vus_pr(_SCORES, _LABELS)

        assert result == {"value": None, "reason": vus.REASON_SCORING_PATH_PENDING}
        assert result["reason"] != vus.REASON_BACKEND_UNAVAILABLE


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
