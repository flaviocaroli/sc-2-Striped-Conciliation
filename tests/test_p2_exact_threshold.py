import numpy as np

from sc2.eval.p2_selective import (
    choose_exact_threshold,
    exact_threshold_frontier,
)


def test_exact_threshold_zero_fill():
    score = np.array(
        [[0.90, 0.85, 0.80, 0.10]],
        dtype=np.float32,
    )

    positive = np.array(
        [[True, False, True, False]]
    )

    true_zero = np.array(
        [[False, True, False, True]]
    )

    frontier = exact_threshold_frontier(
        score,
        positive,
        true_zero,
    )

    selected = choose_exact_threshold(
        frontier,
        max_true_zero_fill=0.0,
    )

    assert selected["threshold"] == np.float32(0.90)
    assert selected["recall"] == 0.5
    assert selected["precision"] == 1.0
    assert selected["true_zero_fill"] == 0.0


def test_exact_threshold_half_fill():
    score = np.array(
        [[0.90, 0.85, 0.80, 0.10]],
        dtype=np.float32,
    )

    positive = np.array(
        [[True, False, True, False]]
    )

    true_zero = np.array(
        [[False, True, False, True]]
    )

    frontier = exact_threshold_frontier(
        score,
        positive,
        true_zero,
    )

    selected = choose_exact_threshold(
        frontier,
        max_true_zero_fill=0.5,
    )

    assert selected["threshold"] == np.float32(0.80)
    assert selected["recall"] == 1.0
    assert np.isclose(
        selected["precision"],
        2.0 / 3.0,
    )
    assert selected["true_zero_fill"] == 0.5


def test_no_selection_sentinel_exists():
    score = np.array(
        [[0.2, 0.1]],
        dtype=np.float32,
    )

    positive = np.array(
        [[False, False]]
    )

    true_zero = np.array(
        [[True, True]]
    )

    try:
        exact_threshold_frontier(
            score,
            positive,
            true_zero,
        )
    except ValueError as exc:
        assert (
            "No masked-positive positions"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )
