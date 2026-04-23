from __future__ import annotations

import pandas as pd


def summarize_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate per-sample metrics by the specified grouping columns.
    """
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_samples=("mse", "size"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
        )
        .reset_index()
    )
    return grouped


def summarize_overall_by_split(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_by_group(df, ["split"])