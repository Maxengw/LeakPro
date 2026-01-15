
import logging
from typing import Any
from pathlib import Path
import sys

import pytest
import yaml
import torch

BASE_DIR = Path(__file__).resolve().parent.parent  # .../GIA_base_running_tabular
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.create_synthetic_datasets import create_synthetic_datasets
from tabular import get_tabular_loaders
from train import train_global_model

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

CONFIG_PATH = (BASE_DIR / "config.yaml").resolve()

@pytest.fixture()
def synthetic_cfgs(tmp_path):
    template = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return create_synthetic_datasets(template, tmp_path)

def test_dataloader(synthetic_cfgs):
    for cfg in synthetic_cfgs.values():
        loaders, encoder_meta = get_tabular_loaders(cfg)
        train_len = len(loaders["train_loader"].dataset)
        val_len = len(loaders["val_loader"].dataset)
        client_len = len(loaders["client_loader"].dataset)
        test_len = len(loaders["test_loader"].dataset) if loaders.get("test_loader") else 0
        total_len = train_len + val_len + test_len + client_len

        logger.info("splits train=%d val=%d test=%d client=%d", train_len, val_len, test_len, client_len)
        assert train_len > 0 and val_len > 0 and client_len > 0
        assert total_len == 1000

        xb, yb = next(iter(loaders["train_loader"]))
        logger.info("feature dim=%d num_classes=%d", loaders["n_features"], loaders["num_classes"])
        assert xb.shape[1] == loaders["n_features"] == len(encoder_meta["feature_columns"])

        if loaders["num_classes"] == 2:
            # binary uses BCE-style float targets shaped [N,1]
            assert yb.ndim == 2 and yb.shape[1] == 1 and yb.dtype.is_floating_point
        elif loaders["num_classes"] > 2:
            assert yb.dtype == yb.long().dtype
        else:
            assert yb.dtype.is_floating_point

def test_train(synthetic_cfgs, tmp_path):
    for cfg in synthetic_cfgs.values():
        cfg = cfg.copy()
        cfg["trainer"] = {"checkpoint_dir": tmp_path}
        ckpt, best = train_global_model(cfg)
        assert ckpt.exists() and ckpt.stat().st_size > 0
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "model_state" in state and state["model_state"]
        meta = state.get("meta", {})
        assert set(["feature_dim", "num_classes", "task"]).issubset(meta.keys())
        if state.get("data_mean") is not None:
            assert state["data_mean"].numel() == meta.get("feature_dim")
        if state.get("data_std") is not None:
            assert state["data_std"].numel() == meta.get("feature_dim")
        assert torch.isfinite(torch.tensor(best))


def test_main():
    pytest.skip("not implemented")

# if __name__ == "__main__":
#     pytest.main([__file__])