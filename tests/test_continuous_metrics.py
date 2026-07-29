import numpy as np

from sc2.eval.selective_repair_metrics import choose_threshold, gate_discrimination, threshold_sweep


def test_threshold_budget() -> None:
    probability = np.array([0.95, 0.8, 0.2, 0.1])
    positive = np.array([True, False, False, False])
    zero = np.array([False, True, True, True])
    selected = choose_threshold(threshold_sweep(probability, positive, zero), max_true_zero_fill=0.0)
    assert selected.true_zero_fill == 0.0
    assert selected.recall == 1.0


def test_gate_metrics() -> None:
    metrics = gate_discrimination(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([True, True, False, False]),
        np.array([False, False, True, True]),
    )
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
