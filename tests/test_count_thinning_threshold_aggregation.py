from __future__ import annotations

import unittest

import pandas as pd

from scripts.eval.select_count_thinning_validation_thresholds import (
    aggregate_threshold_grid,
    select_common_threshold,
)


class ThresholdAggregationTests(
    unittest.TestCase
):
    def test_every_run_fill_gate(self) -> None:
        rows = []

        for run in ("a", "b", "c"):
            rows.extend(
                [
                    {
                        "run_id": run,
                        "threshold": 0.40,
                        "recall": 0.90,
                        "precision": 0.80,
                        "true_zero_fill": (
                            0.021
                            if run == "c"
                            else 0.010
                        ),
                    },
                    {
                        "run_id": run,
                        "threshold": 0.50,
                        "recall": 0.80,
                        "precision": 0.90,
                        "true_zero_fill": 0.015,
                    },
                ]
            )

        aggregate = aggregate_threshold_grid(
            pd.DataFrame(rows),
            max_true_zero_fill=0.02,
        )

        row40 = aggregate[
            aggregate["threshold"] == 0.40
        ].iloc[0]

        row50 = aggregate[
            aggregate["threshold"] == 0.50
        ].iloc[0]

        self.assertFalse(
            bool(row40["eligible"])
        )

        self.assertTrue(
            bool(row50["eligible"])
        )

        selected = select_common_threshold(
            aggregate
        )

        self.assertAlmostEqual(
            float(selected["threshold"]),
            0.50,
        )

    def test_mean_recall_precedes_precision(
        self,
    ) -> None:
        aggregate = pd.DataFrame(
            [
                {
                    "threshold": 0.40,
                    "n_runs": 15,
                    "mean_recall": 0.80,
                    "mean_precision": 0.99,
                    "mean_true_zero_fill": 0.01,
                    "max_true_zero_fill": 0.02,
                    "eligible": True,
                },
                {
                    "threshold": 0.50,
                    "n_runs": 15,
                    "mean_recall": 0.81,
                    "mean_precision": 0.70,
                    "mean_true_zero_fill": 0.01,
                    "max_true_zero_fill": 0.02,
                    "eligible": True,
                },
            ]
        )

        selected = select_common_threshold(
            aggregate
        )

        self.assertAlmostEqual(
            float(selected["threshold"]),
            0.50,
        )

    def test_precision_breaks_recall_tie(
        self,
    ) -> None:
        aggregate = pd.DataFrame(
            [
                {
                    "threshold": 0.40,
                    "n_runs": 15,
                    "mean_recall": 0.80,
                    "mean_precision": 0.75,
                    "mean_true_zero_fill": 0.01,
                    "max_true_zero_fill": 0.02,
                    "eligible": True,
                },
                {
                    "threshold": 0.50,
                    "n_runs": 15,
                    "mean_recall": 0.80,
                    "mean_precision": 0.76,
                    "mean_true_zero_fill": 0.01,
                    "max_true_zero_fill": 0.02,
                    "eligible": True,
                },
            ]
        )

        selected = select_common_threshold(
            aggregate
        )

        self.assertAlmostEqual(
            float(selected["threshold"]),
            0.50,
        )

    def test_stricter_threshold_breaks_full_tie(
        self,
    ) -> None:
        aggregate = pd.DataFrame(
            [
                {
                    "threshold": 0.50,
                    "n_runs": 15,
                    "mean_recall": 0.80,
                    "mean_precision": 0.75,
                    "mean_true_zero_fill": 0.01,
                    "max_true_zero_fill": 0.02,
                    "eligible": True,
                },
                {
                    "threshold": 0.55,
                    "n_runs": 15,
                    "mean_recall": 0.80,
                    "mean_precision": 0.75,
                    "mean_true_zero_fill": 0.01,
                    "max_true_zero_fill": 0.02,
                    "eligible": True,
                },
            ]
        )

        selected = select_common_threshold(
            aggregate
        )

        self.assertAlmostEqual(
            float(selected["threshold"]),
            0.55,
        )

    def test_no_eligible_threshold_stops(
        self,
    ) -> None:
        aggregate = pd.DataFrame(
            [
                {
                    "threshold": 0.50,
                    "n_runs": 15,
                    "mean_recall": 0.80,
                    "mean_precision": 0.75,
                    "mean_true_zero_fill": 0.03,
                    "max_true_zero_fill": 0.04,
                    "eligible": False,
                }
            ]
        )

        with self.assertRaises(
            RuntimeError
        ):
            select_common_threshold(
                aggregate
            )


if __name__ == "__main__":
    unittest.main()
