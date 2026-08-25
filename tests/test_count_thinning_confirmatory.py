from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sc2.data.count_thinning import (
    cp10k_log1p_counts,
)

from scripts.data.materialize_count_thinning_confirmatory import (
    PROTOCOL_ID,
    materialize_dataset,
    sha256_file,
)


REPO = Path(
    __file__
).resolve().parents[1]

PROTOCOL = (
    REPO
    / "configs"
    / "extension_2026"
    / "count_thinning_confirmatory_protocol_v1.json"
)


class ConfirmatoryProtocolTests(
    unittest.TestCase
):

    def test_frozen_protocol_contract(
        self,
    ):
        p = json.loads(
            PROTOCOL.read_text()
        )

        self.assertEqual(
            p["protocol_id"],
            PROTOCOL_ID,
        )

        self.assertEqual(
            p["feature_space"]["n_genes"],
            4096,
        )

        self.assertEqual(
            p["confirmatory_datasets"],
            [
                "internal_test",
                "baron_pancreas",
                "zheng68k",
            ],
        )

        self.assertEqual(
            p["thinning"][
                "depth_retention_fractions"
            ],
            [
                0.85,
                0.70,
                0.50,
            ],
        )

        self.assertEqual(
            p["thinning"][
                "replicates_per_depth"
            ],
            5,
        )

        datasets = p[
            "materialization"
        ][
            "datasets"
        ]

        self.assertEqual(
            set(datasets),
            {
                "internal_test",
                "baron_pancreas",
                "zheng68k",
            },
        )

        all_seeds = []

        for name, spec in datasets.items():

            self.assertEqual(
                spec["n_cells"],
                5000,
            )

            self.assertEqual(
                spec["n_genes"],
                4096,
            )

            self.assertEqual(
                spec[
                    "expected_panel_count"
                ],
                15,
            )

            panels = spec[
                "panel_specs"
            ]

            self.assertEqual(
                len(panels),
                15,
            )

            pairs = {
                (
                    round(
                        float(x["q"]),
                        2,
                    ),
                    int(
                        x["replicate"]
                    ),
                )
                for x in panels
            }

            self.assertEqual(
                pairs,
                {
                    (q, rep)
                    for q in (
                        0.85,
                        0.70,
                        0.50,
                    )
                    for rep in range(
                        1,
                        6,
                    )
                },
            )

            all_seeds.extend(
                int(
                    x["seed"]
                )
                for x in panels
            )

        self.assertEqual(
            len(all_seeds),
            45,
        )

        self.assertEqual(
            len(set(all_seeds)),
            45,
        )

        self.assertEqual(
            p["materialization"][
                "expected_total_panel_count"
            ],
            45,
        )

        thresholds = p[
            "confirmatory_inference"
        ][
            "primary_thresholds"
        ]

        expected = {
            "0.85": 0.575,
            "0.70": 0.560,
            "0.50": 0.535,
        }

        for q, target in expected.items():
            self.assertTrue(
                math.isclose(
                    float(
                        thresholds[q]
                    ),
                    target,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )

        tasks = p[
            "confirmatory_inference"
        ][
            "task_layout"
        ][
            "tasks"
        ]

        self.assertEqual(
            len(tasks),
            45,
        )

        self.assertEqual(
            [
                int(x["task_id"])
                for x in tasks
            ],
            list(
                range(45)
            ),
        )

        self.assertEqual(
            p["confirmatory_inference"][
                "task_layout"
            ][
                "submission_waves"
            ],
            [
                {
                    "array_range": "0-28",
                    "elements": 29,
                },
                {
                    "array_range": "29-44",
                    "elements": 16,
                },
            ],
        )

    def test_implementation_hashes(
        self,
    ):
        p = json.loads(
            PROTOCOL.read_text()
        )

        impl = p[
            "implementation"
        ]

        mapping = {
            "materializer":
                REPO
                / "scripts"
                / "data"
                / "materialize_count_thinning_confirmatory.py",

            "tests":
                REPO
                / "tests"
                / "test_count_thinning_confirmatory.py",

            "slurm":
                REPO
                / "slurm"
                / "extension_2026"
                / "materialize_count_thinning_confirmatory_array.slurm",
        }

        for key, path in mapping.items():

            self.assertEqual(
                sha256_file(path),
                impl[
                    f"{key}_sha256"
                ],
            )


class ConfirmatoryMaterializerTests(
    unittest.TestCase
):

    def test_tiny_end_to_end_bundle(
        self,
    ):
        with tempfile.TemporaryDirectory() as td:

            root = Path(td)

            counts = np.asarray(
                [
                    [20, 10, 5, 0],
                    [15, 12, 7, 0],
                    [18, 11, 9, 0],
                    [13, 14, 6, 0],
                ],
                dtype=np.uint32,
            )

            available = np.asarray(
                [
                    True,
                    True,
                    True,
                    False,
                ],
                dtype=np.bool_,
            )

            counts_path = (
                root
                / "counts_uint32.npy"
            )

            available_path = (
                root
                / "available_gene_mask.npy"
            )

            metadata_path = (
                root
                / "metadata.json"
            )

            reference_path = (
                root
                / "reference.npz"
            )

            np.save(
                counts_path,
                counts,
            )

            np.save(
                available_path,
                available,
            )

            full_target = cp10k_log1p_counts(
                counts,
                available,
                zero_library_rule="error",
            )

            np.savez(
                reference_path,
                y=full_target,
            )

            semantic_counts = (
                "semantic-counts"
            )

            semantic_available = (
                "semantic-available"
            )

            metadata = {
                "dataset":
                    "tiny",

                "shape":
                    [4, 4],

                "counts_dtype":
                    "uint32",

                "counts_file_sha256":
                    sha256_file(
                        counts_path
                    ),

                "counts_uint32_sha256":
                    semantic_counts,

                "available_file_sha256":
                    sha256_file(
                        available_path
                    ),

                "available_gene_mask_sha256":
                    semantic_available,

                "n_available_genes":
                    3,
            }

            metadata_path.write_text(
                json.dumps(
                    metadata,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            output = (
                root
                / "output"
            )

            protocol = {
                "protocol_id":
                    PROTOCOL_ID,

                "feature_space": {
                    "n_genes": 4,
                    "ordered_vocabulary_path":
                        "test",
                    "ordered_vocabulary_sha256":
                        "test-vocab",
                },

                "provenance": {
                    "count_view_receipt_sha256":
                        "test-receipt",

                    "threshold_config_sha256":
                        "test-thresholds",
                },

                "materialization": {
                    "datasets": {
                        "tiny": {
                            "n_cells": 4,
                            "n_genes": 4,
                            "n_available_genes": 3,

                            "split": "test",

                            "counts_path":
                                str(
                                    counts_path
                                ),

                            "counts_file_sha256":
                                sha256_file(
                                    counts_path
                                ),

                            "counts_uint32_sha256":
                                semantic_counts,

                            "available_path":
                                str(
                                    available_path
                                ),

                            "available_file_sha256":
                                sha256_file(
                                    available_path
                                ),

                            "available_gene_mask_sha256":
                                semantic_available,

                            "metadata_path":
                                str(
                                    metadata_path
                                ),

                            "metadata_sha256":
                                sha256_file(
                                    metadata_path
                                ),

                            "reference_target_path":
                                str(
                                    reference_path
                                ),

                            "reference_target_sha256":
                                sha256_file(
                                    reference_path
                                ),

                            "output_dir":
                                str(output),

                            "expected_panel_count":
                                1,

                            "panel_specs": [
                                {
                                    "q": 0.85,
                                    "replicate": 1,
                                    "seed": 12345,
                                }
                            ],
                        }
                    }
                },
            }

            protocol_path = (
                root
                / "protocol.json"
            )

            protocol_path.write_text(
                json.dumps(
                    protocol,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            result = materialize_dataset(
                protocol=protocol,
                protocol_path=
                    protocol_path,
                dataset="tiny",
                output_dir=output,
            )

            self.assertEqual(
                result["panel_count"],
                1,
            )

            self.assertTrue(
                output.is_dir()
            )

            panels = list(
                output.glob(
                    "tiny_q085_"
                    "rep01_seed12345.npz"
                )
            )

            self.assertEqual(
                len(panels),
                1,
            )

            with np.load(
                panels[0],
                allow_pickle=False,
            ) as d:

                self.assertEqual(
                    d["x"].shape,
                    (4, 4),
                )

                self.assertEqual(
                    d["y"].shape,
                    (4, 4),
                )

                self.assertTrue(
                    np.array_equal(
                        d[
                            "synthetic_mask"
                        ],
                        d[
                            "lost_positive_mask"
                        ],
                    )
                )

                thin = np.asarray(
                    d[
                        "thinned_counts"
                    ]
                )

                self.assertTrue(
                    np.all(
                        thin <= counts
                    )
                )

                self.assertTrue(
                    np.all(
                        thin[
                            :,
                            ~available
                        ]
                        == 0
                    )
                )

            manifest = (
                output
                / "SHA256SUMS.txt"
            )

            self.assertTrue(
                manifest.is_file()
            )

            for line in (
                manifest
                .read_text()
                .splitlines()
            ):

                digest, name = line.split(
                    None,
                    1,
                )

                self.assertEqual(
                    sha256_file(
                        output
                        / name.strip()
                    ),
                    digest,
                )

            qc = json.loads(
                (
                    output
                    / "materialization_qc_summary.json"
                ).read_text()
            )

            self.assertEqual(
                qc[
                    "full_depth_zero_library_cells"
                ],
                0,
            )

            self.assertEqual(
                qc[
                    "materialized_panel_count"
                ],
                1,
            )

            self.assertTrue(
                qc[
                    "full_target_exactly_matches_reference_y"
                ]
            )


if __name__ == "__main__":
    unittest.main()
