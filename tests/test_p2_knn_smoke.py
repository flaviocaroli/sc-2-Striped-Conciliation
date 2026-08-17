import numpy as np

from sc2.eval.p2_knn import (
    knn_value_and_score,
    neighbor_indices_excluding_self,
)


def test_knn_self_exclusion():
    x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    indices, distances = (
        neighbor_indices_excluding_self(
            x,
            k=2,
        )
    )

    assert indices.shape == (5, 2)
    assert distances.shape == (5, 2)

    rows = np.arange(5)[:, None]

    assert not np.any(
        indices == rows
    )


def test_knn_value_and_score():
    x = np.array(
        [
            [1.0, 0.0],
            [3.0, 2.0],
            [5.0, 0.0],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            [1, 2],
            [0, 2],
            [0, 1],
        ],
        dtype=np.int64,
    )

    prediction, score = (
        knn_value_and_score(
            x,
            indices,
            zero_threshold=1.0e-8,
            chunk_size=2,
        )
    )

    np.testing.assert_allclose(
        prediction[0],
        [4.0, 1.0],
    )

    np.testing.assert_allclose(
        score[0],
        [1.0, 0.5],
    )

    assert np.all(score >= 0.0)
    assert np.all(score <= 1.0)
