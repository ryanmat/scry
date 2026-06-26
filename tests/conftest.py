# Description: Pytest fixtures and path setup for scry tests.
# Description: Adds scripts/ to sys.path so scripts can import sibling modules under pytest.

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
