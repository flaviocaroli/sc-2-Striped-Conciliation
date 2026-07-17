from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.size < 3:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)

    raise ValueError(f"Unknown method: {method}")


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.mean((x - y) ** 2))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.mean(np.abs(x - y)))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(mse(x, y)))


def masked_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prefix: str = "masked",
) -> dict[str, float]:
    pred_m = pred[mask]
    target_m = target[mask]

    return {
        f"{prefix}_n": float(pred_m.size),
        f"{prefix}_mse": mse(pred_m, target_m),
        f"{prefix}_rmse": rmse(pred_m, target_m),
        f"{prefix}_mae": mae(pred_m, target_m),
        f"{prefix}_pearson": safe_corr(pred_m, target_m, "pearson"),
        f"{prefix}_spearman": safe_corr(pred_m, target_m, "spearman"),
    }


def samplewise_reconstruction_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for i in range(pred.shape[0]):
        if mask is None:
            p = pred[i]
            t = target[i]
            m = np.ones_like(t, dtype=bool)
        else:
            p = pred[i][mask[i]]
            t = target[i][mask[i]]
            m = mask[i][mask[i]]

        if p.size == 0:
            rows.append(
                {
                    "sample_index": i,
                    "n_eval": 0,
                    "mse": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "pearson": np.nan,
                    "spearman": np.nan,
                }
            )
            continue

        rows.append(
            {
                "sample_index": i,
                "n_eval": int(p.size),
                "mse": mse(p, t),
                "rmse": rmse(p, t),
                "mae": mae(p, t),
                "pearson": safe_corr(p, t, "pearson"),
                "spearman": safe_corr(p, t, "spearman"),
            }
        )

    return pd.DataFrame(rows)


def gene_wise_correlation(
    pred: np.ndarray,
    target: np.ndarray,
    gene_names: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for j in range(pred.shape[1]):
        rows.append(
            {
                "gene_index": j,
                "gene": gene_names[j] if gene_names is not None else str(j),
                "pearson": safe_corr(pred[:, j], target[:, j], "pearson"),
                "spearman": safe_corr(pred[:, j], target[:, j], "spearman"),
                "mse": mse(pred[:, j], target[:, j]),
                "mae": mae(pred[:, j], target[:, j]),
            }
        )

    return pd.DataFrame(rows)


def false_positive_zero_fill_rate(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.1,
) -> float:
    true_zero = target <= 0
    if true_zero.sum() == 0:
        return float("nan")
    filled = pred[true_zero] > threshold
    return float(filled.mean())


def pseudobulk_consistency_metrics(
    pred_cells: np.ndarray,
    real_bulk: np.ndarray,
) -> dict[str, float]:
    pred_pb = pred_cells.mean(axis=0)
    return {
        "pb_to_bulk_mse": mse(pred_pb, real_bulk),
        "pb_to_bulk_rmse": rmse(pred_pb, real_bulk),
        "pb_to_bulk_mae": mae(pred_pb, real_bulk),
        "pb_to_bulk_pearson": safe_corr(pred_pb, real_bulk, "pearson"),
        "pb_to_bulk_spearman": safe_corr(pred_pb, real_bulk, "spearman"),
    }