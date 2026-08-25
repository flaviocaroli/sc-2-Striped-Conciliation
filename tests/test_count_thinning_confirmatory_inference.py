from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from scripts.eval.run_count_thinning_confirmatory_task import (
    PROTOCOL_ID,
    resolve_task,
    sha256_file,
)


REPO = Path(
    __file__
).resolve().parents[1]

PROTOCOL = (
    REPO
    / "configs"
    / "extension_2026"
    / "count_thinning_confirmatory_inference_protocol_v1.json"
)


class ConfirmatoryInferenceTests(
    unittest.TestCase
):

    def test_protocol_contract(self):

        p = json.loads(
            PROTOCOL.read_text()
        )

        self.assertEqual(
            p["protocol_id"],
            PROTOCOL_ID,
        )

        self.assertEqual(
            p["expected_tasks"],
            45,
        )

        self.assertEqual(
            p["q_per_task"],
            3,
        )

        self.assertEqual(
            p["expected_evaluations"],
            135,
        )

        self.assertEqual(
            p["resources"],
            {
                "partition":
                    "short_gpuh200",
                "gpus":
                    1,
                "cpus_per_task":
                    4,
                "memory":
                    "32G",
                "time":
                    "01:00:00",
            },
        )

        self.assertEqual(
            p["submission_waves"],
            [
                {
                    "array_range":
                        "0-28",
                    "elements":
                        29,
                },
                {
                    "array_range":
                        "29-44",
                    "elements":
                        16,
                },
            ],
        )

        self.assertFalse(
            p[
                "threshold_policy"
            ][
                "automatic_selection_allowed"
            ]
        )

        self.assertFalse(
            p[
                "threshold_policy"
            ][
                "retuning_allowed"
            ]
        )

        self.assertFalse(
            p[
                "threshold_policy"
            ][
                "old_transfer_gpu_rerun"
            ]
        )

    def test_bound_hashes(self):

        p = json.loads(
            PROTOCOL.read_text()
        )

        for block in (
            "scientific_protocol",
            "panel_receipt",
            "threshold_config",
            "evaluator",
            "evaluator_config",
        ):
            path = Path(
                p[block]["path"]
            )

            self.assertEqual(
                sha256_file(path),
                p[block]["sha256"],
            )

    def test_implementation_hashes(self):

        p = json.loads(
            PROTOCOL.read_text()
        )

        mapping = {
            "runner":
                REPO
                / "scripts"
                / "eval"
                / "run_count_thinning_confirmatory_task.py",

            "tests":
                REPO
                / "tests"
                / "test_count_thinning_confirmatory_inference.py",

            "slurm":
                REPO
                / "slurm"
                / "extension_2026"
                / "eval_count_thinning_confirmatory_array.slurm",
        }

        for name, path in mapping.items():
            self.assertEqual(
                sha256_file(path),
                p[
                    "implementation"
                ][name + "_sha256"],
            )

    def test_45_tasks_resolve_to_135_evaluations(
        self,
    ):

        task_keys = set()
        total = 0

        expected_thresholds = {
            0.85: 0.575,
            0.70: 0.560,
            0.50: 0.535,
        }

        for task_id in range(45):

            resolved = resolve_task(
                inference_protocol_path=
                    PROTOCOL,
                task_id=task_id,
                verify_panel_payloads=False,
            )

            key = (
                resolved["model_seed"],
                resolved["dataset"],
                resolved["replicate"],
            )

            self.assertNotIn(
                key,
                task_keys,
            )

            task_keys.add(key)

            evaluations = resolved[
                "evaluations"
            ]

            self.assertEqual(
                len(evaluations),
                3,
            )

            self.assertEqual(
                [
                    round(x["q"], 2)
                    for x in evaluations
                ],
                [0.85, 0.70, 0.50],
            )

            for item in evaluations:

                q = round(
                    item["q"],
                    2,
                )

                self.assertTrue(
                    item[
                        "panel_path"
                    ].is_file()
                )

                self.assertTrue(
                    math.isclose(
                        item["threshold"],
                        expected_thresholds[q],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )

            total += len(evaluations)

        self.assertEqual(
            len(task_keys),
            45,
        )

        self.assertEqual(
            total,
            135,
        )


if __name__ == "__main__":
    unittest.main()
