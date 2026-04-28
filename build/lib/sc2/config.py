from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def merge_train_and_paths(
    train_cfg: dict[str, Any],
    path_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg = deepcopy(train_cfg)

    cfg["paths"] = {
        "data_root": os.environ.get("SC2_DATA_ROOT", path_cfg["data_root"]),
        "output_root": os.environ.get("SC2_OUTPUT_ROOT", path_cfg["output_root"]),
        "log_root": path_cfg["log_root"],
        "artifact_root": path_cfg["artifact_root"],
    }
    return cfg