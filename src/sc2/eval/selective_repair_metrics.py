from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from scipy.stats import rankdata


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    recall: float
    precision: float
    true_zero_fill: float
    selected: int


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def samplewise_spearman(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return _finite_mean(_spearman(prediction[row, mask[row]], target[row, mask[row]]) for row in range(prediction.shape[0]))


def genewise_spearman(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return _finite_mean(_spearman(prediction[mask[:, column], column], target[mask[:, column], column]) for column in range(prediction.shape[1]))


def calibration_error(probability: np.ndarray, label: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, probability.size)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (probability >= lower) & (probability < upper if upper < 1.0 else probability <= upper)
        if selected.any():
            error += selected.sum() / total * abs(float(probability[selected].mean()) - float(label[selected].mean()))
    return float(error)


def gate_discrimination(probability: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    eligible = positive | negative
    labels = positive[eligible].astype(np.int8)
    scores = probability[eligible].astype(np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "ece": float("nan"), "prevalence": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "ece": calibration_error(scores, labels),
        "prevalence": float(labels.mean()),
    }


def threshold_sweep(
    probability: np.ndarray,
    positive: np.ndarray,
    true_zero: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    values = np.linspace(0.0, 1.0, 201) if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    rows = []
    for threshold in values:
        selected = probability >= threshold
        tp = int((selected & positive).sum())
        fp = int((selected & true_zero).sum())
        recall = tp / max(1, int(positive.sum()))
        precision = tp / max(1, tp + fp)
        zero_fill = fp / max(1, int(true_zero.sum()))
        rows.append({"threshold": float(threshold), "recall": recall, "precision": precision, "true_zero_fill": zero_fill, "selected": int(selected.sum())})
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame, *, max_true_zero_fill: float, min_precision: float = 0.0) -> ThresholdSelection:
    eligible = sweep[(sweep["true_zero_fill"] <= max_true_zero_fill) & (sweep["precision"] >= min_precision)]
    if eligible.empty:
        row = sweep.sort_values(["true_zero_fill", "threshold"], ascending=[True, False]).iloc[0]
    else:
        row = eligible.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return ThresholdSelection(
        threshold=float(row["threshold"]),
        recall=float(row["recall"]),
        precision=float(row["precision"]),
        true_zero_fill=float(row["true_zero_fill"]),
        selected=int(row["selected"]),
    )


def masked_value_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    error = prediction[mask] - target[mask]
    prediction_sd = float(np.std(prediction[mask]))
    target_sd = float(np.std(target[mask]))
    return {
        "masked_mse": float(np.mean(error**2)),
        "masked_mae": float(np.mean(np.abs(error))),
        "sample_spearman": samplewise_spearman(prediction, target, mask),
        "gene_spearman": genewise_spearman(prediction, target, mask),
        "prediction_sd": prediction_sd,
        "target_sd": target_sd,
        "sd_ratio": prediction_sd / target_sd if target_sd > 0.0 else float("nan"),
        "n_masked": int(mask.sum()),
    }


def risk_coverage_curve(
    expected_repair: np.ndarray,
    target: np.ndarray,
    positive: np.ndarray,
    probability: np.ndarray,
    coverages: np.ndarray | None = None,
) -> pd.DataFrame:
    values = np.linspace(0.01, 1.0, 100) if coverages is None else np.asarray(coverages)
    positive_scores = probability[positive]
    positive_prediction = expected_repair[positive]
    positive_target = target[positive]
    order = np.argsort(-positive_scores)
    rows = []
    for coverage in values:
        count = max(1, int(round(float(coverage) * order.size)))
        selected = order[:count]
        error = positive_prediction[selected] - positive_target[selected]
        rows.append({"coverage": count / max(1, order.size), "risk_mse": float(np.mean(error**2)), "risk_mae": float(np.mean(np.abs(error)))})
    return pd.DataFrame(rows)
