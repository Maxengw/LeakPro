"""Small importable function which creates 3 small synthetic datasets containing 2 features and 1 target (3 in total).

1 numeric feature and 1 categorical feature.
The target will be binary classification for the 1st dataset.
The target will be multiclassification for the 2nd dataset.
The target will be regression for the 3rd dataset.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _generate_synthetic_data(task: str, template_cfg: dict[str, Any], tmp_root: Path) -> dict:
    """Helper to generate the synthetic data.

    Features a, b, y, where a is numerical, b is categorical, and y is the target.
    """

    rng = np.random.default_rng(42)
    n = 1000
    tmpdir = tmp_root / f"{task}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    if task == "binary":
        a = rng.normal(loc=0.0, scale=1.0, size=n)
        b = rng.choice(["x", "y"], size=n)
        y = (a + (b == "y").astype(float) + rng.normal(scale=0.2, size=n) > 0.5).astype(int)
    elif task == "multiclass":
        a = rng.uniform(low=-1.0, high=1.0, size=n)
        b = rng.choice(["x", "y", "z"], size=n)
        logits = np.stack(
            [
                a + 0.5 * (b == "x"),
                -a + 0.5 * (b == "y"),
                0.2 * a + 0.5 * (b == "z"),
            ],
            axis=1,
        )
        y = logits.argmax(axis=1)
    elif task == "regression":
        a = rng.normal(loc=2.0, scale=1.5, size=n)
        b = rng.choice(["x", "y"], size=n)
        y = 1.5 * a + 0.7 * (b == "y").astype(float) + rng.normal(scale=0.5, size=n)
    else:
        raise ValueError(f"Unsupported task: {task}")

    df = pd.DataFrame({"a": a, "b": b, "y": y})

    csv_path = tmpdir / f"{task}.csv"
    meta_path = tmpdir / f"{task}.yaml"
    df.to_csv(csv_path, index=False)
    meta = {"target": "y", "no_header": False, "missing_values": ""}
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f)

    cfg = template_cfg.get("dataset", template_cfg).copy() if isinstance(template_cfg, dict) else {}
    cfg["data_path"] = csv_path
    cfg["meta_path"] = meta_path
    return cfg


def create_synthetic_datasets(template_cfg: dict[str, Any], tmp_path: Path) -> dict:
    """Return the paths for the synthetic datasets for test.py."""

    tasks = ["binary", "multiclass", "regression"]
    task_cfg_dicts = {} # the paths to the config.yaml and the meta.yaml
    for task in tasks:
        task_cfg_dicts[task] = _generate_synthetic_data(task, template_cfg, tmp_path)

    return task_cfg_dicts
