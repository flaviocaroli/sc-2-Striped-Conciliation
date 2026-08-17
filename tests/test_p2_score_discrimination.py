import numpy as np

from sc2.eval.p2_selective import (
    score_discrimination,
)


def test_unbounded_score_discrimination():
    score = np.array(
        [[4.5, 3.0, 2.5, 0.2]],
        dtype=np.float32,
    )

    positive = np.array(
        [[True, False, True, False]]
    )

    negative = np.array(
        [[False, True, False, True]]
    )

    result = score_discrimination(
        score,
        positive,
        negative,
    )

    assert 0.0 <= result["auroc"] <= 1.0
    assert 0.0 <= result["auprc"] <= 1.0
    assert result["prevalence"] == 0.5
