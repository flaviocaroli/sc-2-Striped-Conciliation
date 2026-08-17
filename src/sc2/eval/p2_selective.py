from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)



def score_discrimination(
    score: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
) -> dict[str, float]:
    if (
        score.shape != positive.shape
        or score.shape != negative.shape
    ):
        raise ValueError(
            "score, positive and negative must "
            "have identical shapes"
        )

    if np.any(positive & negative):
        raise ValueError(
            "positive and negative overlap"
        )

    eligible = positive | negative

    labels = np.asarray(
        positive[eligible],
        dtype=np.int8,
    )

    scores = np.asarray(
        score[eligible],
        dtype=np.float64,
    )

    if scores.size == 0:
        raise ValueError(
            "No eligible score positions"
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "Non-finite discrimination score"
        )

    prevalence = float(labels.mean())

    if np.unique(labels).size < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "prevalence": prevalence,
        }

    return {
        "auroc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),
        "auprc": float(
            average_precision_score(
                labels,
                scores,
            )
        ),
        "prevalence": prevalence,
    }


def exact_threshold_frontier(
    score: np.ndarray,
    positive: np.ndarray,
    true_zero: np.ndarray,
) -> pd.DataFrame:
    if (
        score.shape != positive.shape
        or score.shape != true_zero.shape
    ):
        raise ValueError(
            "score, positive and true_zero must "
            "have identical shapes"
        )

    if np.any(positive & true_zero):
        raise ValueError(
            "positive and true_zero overlap"
        )

    evaluation = positive | true_zero

    scores = np.asarray(
        score[evaluation],
        dtype=np.float64,
    )

    labels = np.asarray(
        positive[evaluation],
        dtype=bool,
    )

    if scores.size == 0:
        raise ValueError(
            "No positive or true-zero positions"
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "Non-finite repair score"
        )

    n_positive = int(positive.sum())
    n_true_zero = int(true_zero.sum())

    if n_positive <= 0:
        raise ValueError(
            "No masked-positive positions"
        )

    if n_true_zero <= 0:
        raise ValueError(
            "No true-zero positions"
        )

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_score = scores[order]
    sorted_positive = labels[order]

    cumulative_tp = np.cumsum(
        sorted_positive,
        dtype=np.int64,
    )

    cumulative_selected = np.arange(
        1,
        sorted_score.size + 1,
        dtype=np.int64,
    )

    cumulative_fp = (
        cumulative_selected
        - cumulative_tp
    )

    group_end = np.r_[
        sorted_score[:-1]
        != sorted_score[1:],
        True,
    ]

    index = np.flatnonzero(group_end)

    threshold = sorted_score[index]
    tp = cumulative_tp[index]
    fp = cumulative_fp[index]
    selected = cumulative_selected[index]

    recall = (
        tp.astype(np.float64)
        / float(n_positive)
    )

    precision = np.divide(
        tp.astype(np.float64),
        selected.astype(np.float64),
        out=np.zeros(
            selected.size,
            dtype=np.float64,
        ),
        where=selected > 0,
    )

    fill = (
        fp.astype(np.float64)
        / float(n_true_zero)
    )

    sentinel = np.nextafter(
        float(sorted_score.max()),
        np.inf,
    )

    return pd.DataFrame(
        {
            "threshold": np.r_[
                sentinel,
                threshold,
            ],
            "tp": np.r_[0, tp].astype(
                np.int64
            ),
            "fp": np.r_[0, fp].astype(
                np.int64
            ),
            "selected": np.r_[
                0,
                selected,
            ].astype(np.int64),
            "recall": np.r_[
                0.0,
                recall,
            ],
            "precision": np.r_[
                0.0,
                precision,
            ],
            "true_zero_fill": np.r_[
                0.0,
                fill,
            ],
        }
    )


def choose_exact_threshold(
    frontier: pd.DataFrame,
    *,
    max_true_zero_fill: float,
) -> pd.Series:
    feasible = frontier.loc[
        frontier["true_zero_fill"]
        <= float(max_true_zero_fill)
    ].copy()

    if feasible.empty:
        raise RuntimeError(
            "No feasible threshold found"
        )

    feasible = feasible.sort_values(
        by=[
            "recall",
            "precision",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        kind="mergesort",
    )

    return feasible.iloc[0]


def presentation_frontier(
    frontier: pd.DataFrame,
    selected_threshold: float,
    *,
    max_rows: int = 201,
) -> pd.DataFrame:
    if len(frontier) <= max_rows:
        out = frontier.copy()
    else:
        indices = np.linspace(
            0,
            len(frontier) - 1,
            max_rows,
        )

        indices = np.unique(
            np.rint(indices).astype(int)
        )

        out = frontier.iloc[
            indices
        ].copy()

    nearest = int(
        np.argmin(
            np.abs(
                frontier[
                    "threshold"
                ].to_numpy()
                - selected_threshold
            )
        )
    )

    selected = frontier.iloc[
        [nearest]
    ]

    out = pd.concat(
        [out, selected],
        ignore_index=True,
    )

    out = (
        out.drop_duplicates(
            subset=["threshold"]
        )
        .sort_values(
            "threshold",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    out["operating_point"] = np.isclose(
        out["threshold"].to_numpy(),
        selected_threshold,
        rtol=0.0,
        atol=0.0,
    )

    return out
