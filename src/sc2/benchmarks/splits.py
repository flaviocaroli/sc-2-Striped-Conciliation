from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def assign_group_holdout_split(
    manifest: pd.DataFrame,
    group_col: str,
    id_col: str,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    if group_col not in manifest.columns:
        raise KeyError(f"Missing group_col={group_col}")
    if id_col not in manifest.columns:
        raise KeyError(f"Missing id_col={id_col}")

    rng = np.random.default_rng(seed)
    groups = np.array(sorted(manifest[group_col].dropna().unique()))
    rng.shuffle(groups)

    n = len(groups)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    train_groups = set(groups[n_test + n_val :])

    out = manifest.copy()
    out["split"] = "train"
    out.loc[out[group_col].isin(val_groups), "split"] = "val"
    out.loc[out[group_col].isin(test_groups), "split"] = "test"

    assert set(out.loc[out["split"] == "train", group_col]).isdisjoint(
        set(out.loc[out["split"] == "val", group_col])
    )
    assert set(out.loc[out["split"] == "train", group_col]).isdisjoint(
        set(out.loc[out["split"] == "test", group_col])
    )
    assert set(out.loc[out["split"] == "val", group_col]).isdisjoint(
        set(out.loc[out["split"] == "test", group_col])
    )

    return out


def assign_existing_split(
    manifest: pd.DataFrame,
    split_col: str = "split",
) -> pd.DataFrame:
    if split_col not in manifest.columns:
        raise KeyError(f"Missing split column: {split_col}")

    out = manifest.copy()
    valid = {"train", "val", "test"}
    found = set(out[split_col].dropna().unique())
    bad = found - valid
    if bad:
        raise ValueError(f"Invalid split values: {bad}. Expected only {valid}")

    if split_col != "split":
        out["split"] = out[split_col]

    return out


def summarize_splits(manifest: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    rows = []

    for split, df in manifest.groupby("split"):
        row = {
            "split": split,
            "n_rows": len(df),
        }
        if group_col is not None and group_col in df.columns:
            row[f"n_{group_col}"] = df[group_col].nunique()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("split")