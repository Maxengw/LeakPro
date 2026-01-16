"""Testing the tabular GIA implementation."""

import logging
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent  # .../GIA_base_running_tabular
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from synthetic_datasets.create_synthetic_datasets import create_synthetic_datasets  # noqa: E402
from tabular import get_tabular_loaders  # noqa: E402
from train import train_global_model  # noqa: E402

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

CONFIG_PATH = (BASE_DIR / "config.yaml").resolve()


@pytest.fixture
def synthetic_cfgs(tmp_path: Path) -> dict[str, dict[str, Any]]:
    """Reusable helper for dataloading."""
    template = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return create_synthetic_datasets(template, tmp_path)


def test_dataloader(synthetic_cfgs: dict[str, dict[str, Any]]) -> None:
    """Test to test tabular.py."""
    for cfg in synthetic_cfgs.values():
        loaders, encoder_meta = get_tabular_loaders(cfg)
        train_len = len(loaders["train_loader"].dataset)
        val_len = len(loaders["val_loader"].dataset)
        client_len = len(loaders["client_loader"].dataset)
        test_len = len(loaders["test_loader"].dataset) if loaders.get("test_loader") else 0
        total_len = train_len + val_len + test_len + client_len

        logger.info("splits train=%d val=%d test=%d client=%d", train_len, val_len, test_len, client_len)
        assert train_len > 0
        assert val_len > 0
        assert client_len > 0
        assert total_len == 1000

        xb, yb = next(iter(loaders["train_loader"]))
        logger.info("feature dim=%d num_classes=%d", loaders["n_features"], loaders["num_classes"])
        assert xb.shape[1] == loaders["n_features"] == len(encoder_meta["feature_columns"])

        if loaders["num_classes"] == 2:
            # binary uses BCE-style float targets shaped [N,1]
            assert yb.ndim == 2
            assert yb.shape[1] == 1
            assert yb.dtype.is_floating_point
        elif loaders["num_classes"] > 2:
            assert yb.dtype == yb.long().dtype
        else:
            assert yb.dtype.is_floating_point


def test_train(synthetic_cfgs: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    """Test to test train.py."""
    for cfg in synthetic_cfgs.values():
        cfg = cfg.copy()
        cfg["trainer"] = {"checkpoint_dir": tmp_path}
        ckpt, best = train_global_model(cfg)
        assert ckpt.exists()
        assert ckpt.stat().st_size > 0
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "model_state" in state
        assert state["model_state"]
        meta = state.get("meta", {})
        assert {"feature_dim", "num_classes", "task"}.issubset(meta.keys())
        if state.get("data_mean") is not None:
            assert state["data_mean"].numel() == meta.get("feature_dim")
        if state.get("data_std") is not None:
            assert state["data_std"].numel() == meta.get("feature_dim")
        assert torch.isfinite(torch.tensor(best))


def test_main() -> None:
    """Test to test main.py."""
    pytest.skip("not implemented")
