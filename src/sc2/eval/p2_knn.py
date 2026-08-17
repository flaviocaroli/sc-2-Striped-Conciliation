from __future__ import annotations

import numpy as np

from sklearn.neighbors import NearestNeighbors


def neighbor_indices_excluding_self(
    x: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(
            "x must be a 2D matrix"
        )

    n_samples = int(x.shape[0])

    if k <= 0:
        raise ValueError(
            "k must be positive"
        )

    if k >= n_samples:
        raise ValueError(
            "k must be smaller than n_samples"
        )

    model = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
    )

    model.fit(x)

    #
    # X=None asks for neighbors of each fitted
    # sample while excluding the sample itself.
    #
    distances, indices = model.kneighbors(
        X=None,
        n_neighbors=k,
        return_distance=True,
    )

    if indices.shape != (
        n_samples,
        k,
    ):
        raise RuntimeError(
            f"Unexpected neighbor shape "
            f"{indices.shape}"
        )

    if distances.shape != indices.shape:
        raise RuntimeError(
            "Distance/index shape mismatch"
        )

    row_index = np.arange(
        n_samples,
        dtype=np.int64,
    )[:, None]

    if np.any(indices == row_index):
        raise RuntimeError(
            "Self neighbor was not excluded"
        )

    if not np.all(
        np.isfinite(distances)
    ):
        raise RuntimeError(
            "Non-finite neighbor distance"
        )

    return (
        np.asarray(
            indices,
            dtype=np.int64,
        ),
        np.asarray(
            distances,
            dtype=np.float64,
        ),
    )


def knn_value_and_score(
    x: np.ndarray,
    indices: np.ndarray,
    *,
    zero_threshold: float,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(
            "x must be 2D"
        )

    if indices.ndim != 2:
        raise ValueError(
            "indices must be 2D"
        )

    if indices.shape[0] != x.shape[0]:
        raise ValueError(
            "Neighbor rows do not match x"
        )

    n_samples, n_genes = x.shape

    prediction = np.empty(
        (n_samples, n_genes),
        dtype=np.float32,
    )

    score = np.empty(
        (n_samples, n_genes),
        dtype=np.float32,
    )

    for start in range(
        0,
        n_samples,
        chunk_size,
    ):
        stop = min(
            n_samples,
            start + chunk_size,
        )

        neighbor_values = x[
            indices[start:stop]
        ]

        prediction[start:stop] = (
            np.mean(
                neighbor_values,
                axis=1,
                dtype=np.float32,
            )
        )

        score[start:stop] = np.mean(
            neighbor_values
            > float(zero_threshold),
            axis=1,
            dtype=np.float32,
        )

    if not np.all(
        np.isfinite(prediction)
    ):
        raise RuntimeError(
            "Non-finite kNN prediction"
        )

    if not np.all(
        np.isfinite(score)
    ):
        raise RuntimeError(
            "Non-finite kNN score"
        )

    if np.any(score < 0.0) or np.any(
        score > 1.0
    ):
        raise RuntimeError(
            "kNN score outside [0,1]"
        )

    return prediction, score
