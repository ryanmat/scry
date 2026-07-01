# Description: Pytest fixtures and path setup for scry tests.
# Description: Makes tests/ and scripts/ importable and provides the shared tiny keeper fixture.

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F  # noqa: N812 -- PyTorch convention

import scry.data.feature_engineering as _fe
from scry.data.feature_engineering import set_active_profile
from scry.data.fetcher import DataFetcher
from scry.data.pipeline import XDECFeaturePipeline
from scry.model.xdec import TemporalXDEC
from scry.utils.config import get_config

# Make tests/ importable (shared `synth` helpers) and scripts/ importable
# (`import validate_incident` / `import bake_serving_threshold`).
_TESTS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _TESTS_DIR.parent / "scripts"
for _p in (str(_TESTS_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bake_serving_threshold as _bake  # noqa: E402
from synth import CAT, PROFILE, SEQ_LEN, SERIES, gen_capture, write_csv  # noqa: E402


@pytest.fixture(scope="session")
def keeper_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train a tiny keeper on synthetic normal data and save its checkpoint.

    The training data is windowed and normalized through the real feature
    pipeline so the checkpoint carries the same stored normalization the harness
    and the serving path re-apply. Session-scoped: trained once for the suite.
    """
    tmp = tmp_path_factory.mktemp("keeper")
    train_df, _ = gen_capture("train-node", 800, seed=1)
    train_csv = write_csv(train_df, tmp / "train.csv")

    # Snapshot the global active profile so this session-scoped fixture does not
    # leak aro_node into tests that expect the default profile.
    prev_profile = _fe._active_config
    set_active_profile(PROFILE)
    fetcher = DataFetcher.from_object_store(train_csv)
    pipeline = XDECFeaturePipeline(fetcher, get_config())
    start = pd.Timestamp("2025-01-01T00:00:00Z").to_pydatetime()
    end = pd.Timestamp("2027-01-01T00:00:00Z").to_pydatetime()
    raw = asyncio.run(pipeline.extract(start, end, profile=PROFILE))
    data = pipeline.transform(raw)

    assert data["num_windows"].shape[0] > 0
    assert data["feature_names"]["numerical"] == list(SERIES)

    torch.manual_seed(0)
    np.random.seed(0)
    model = TemporalXDEC(
        num_numerical=len(SERIES),
        num_categorical=len(CAT),
        seq_len=SEQ_LEN,
        num_hidden=16,
        cat_hidden=8,
        latent_dim=4,
        n_clusters=3,
    )
    x_num = torch.tensor(data["num_windows"], dtype=torch.float32)
    x_cat = torch.tensor(data["cat_windows"], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(500):
        optimizer.zero_grad()
        out = model.xvae(x_num, x_cat)
        loss = F.mse_loss(out["x_num_recon"], x_num) + F.mse_loss(out["x_cat_recon"], x_cat)
        loss.backward()
        optimizer.step()
    model.eval()

    ckpt_path = tmp / "keeper.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_numerical": len(SERIES),
                "num_categorical": len(CAT),
                "seq_len": SEQ_LEN,
                "num_hidden": 16,
                "cat_hidden": 8,
                "latent_dim": 4,
                "n_clusters": 3,
            },
            "normalization": {
                "mean": data["num_norm_params"]["mean"],
                "std": data["num_norm_params"]["std"],
            },
            "categorical_normalization": {
                "min": data["cat_norm_params"]["min"],
                "max": data["cat_norm_params"]["max"],
            },
            "feature_schema": {
                "numerical": data["feature_names"]["numerical"],
                "categorical": data["feature_names"]["categorical"],
                "profile": PROFILE,
            },
        },
        ckpt_path,
    )
    _fe._active_config = prev_profile
    return str(ckpt_path)


@pytest.fixture(scope="session")
def serving_keeper_path(keeper_path: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    """A copy of the keeper with a serving reconstruction threshold baked in.

    Bakes the threshold from a fresh all-healthy capture via the real bake
    utility, so the endpoint tests exercise the same serving block a deployed
    model would carry.
    """
    tmp = tmp_path_factory.mktemp("serving_keeper")
    healthy_df, _ = gen_capture("cal-node", 600, seed=11)
    healthy_csv = write_csv(healthy_df, tmp / "healthy.csv")
    out = str(tmp / "keeper_serving.pt")
    _bake.bake(keeper_path, healthy_csv, profile=PROFILE, output=out)
    return out
