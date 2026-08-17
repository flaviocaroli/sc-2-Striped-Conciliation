import numpy as np

from sklearn.utils.extmath import randomized_svd


def test_low_rank_reconstruction():
    rng = np.random.default_rng(
        20260728
    )

    a = rng.normal(
        size=(20, 3)
    )

    b = rng.normal(
        size=(3, 15)
    )

    x = (
        a @ b
    ).astype(np.float32)

    u, s, vt = randomized_svd(
        x,
        n_components=3,
        random_state=20260728,
    )

    reconstructed = (
        u * s[None, :]
    ) @ vt

    assert reconstructed.shape == x.shape

    assert np.mean(
        (
            reconstructed
            - x
        ) ** 2
    ) < 1.0e-8
