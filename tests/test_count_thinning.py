from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from sc2.data.count_thinning import (
    binomial_thin_counts,
    cp10k_log1p_counts,
    identity_sha256,
    thinning_masks,
)

from scripts.data.materialize_census_shards import (
    cp10k_log1p as canonical_cp10k_log1p,
)


class CountThinningTests(unittest.TestCase):
    def test_repeated_seed_is_identical(self) -> None:
        counts = np.asarray(
            [
                [0, 1, 2, 5],
                [4, 0, 7, 2],
            ],
            dtype=np.uint32,
        )

        first = binomial_thin_counts(
            counts,
            q=0.70,
            seed=12345,
        )
        second = binomial_thin_counts(
            counts,
            q=0.70,
            seed=12345,
        )

        self.assertTrue(
            np.array_equal(first, second)
        )

    def test_counts_never_increase_and_remain_integer(self) -> None:
        counts = np.arange(
            1,
            201,
            dtype=np.uint32,
        ).reshape(20, 10)

        thinned = binomial_thin_counts(
            counts,
            q=0.50,
            seed=20260825,
        )

        self.assertTrue(
            np.issubdtype(thinned.dtype, np.integer)
        )
        self.assertTrue(
            np.all(thinned <= counts)
        )
        self.assertTrue(
            np.all(thinned >= 0)
        )

    def test_masks_are_exact(self) -> None:
        full = np.asarray(
            [
                [2, 0, 1, 0],
                [0, 3, 4, 0],
            ],
            dtype=np.uint32,
        )
        thin = np.asarray(
            [
                [0, 0, 1, 0],
                [0, 2, 0, 0],
            ],
            dtype=np.uint32,
        )

        lost, zero = thinning_masks(
            full,
            thin,
        )

        expected_lost = np.asarray(
            [
                [True, False, False, False],
                [False, False, True, False],
            ]
        )
        expected_zero = full == 0

        self.assertTrue(
            np.array_equal(lost, expected_lost)
        )
        self.assertTrue(
            np.array_equal(zero, expected_zero)
        )

    def test_unavailable_genes_do_not_enter_denominator(self) -> None:
        counts = np.asarray(
            [[1, 9]],
            dtype=np.uint32,
        )
        available = np.asarray(
            [True, False],
            dtype=bool,
        )

        normalized = cp10k_log1p_counts(
            counts,
            available,
        )

        self.assertAlmostEqual(
            float(normalized[0, 0]),
            float(np.log1p(10000.0)),
            places=6,
        )
        self.assertEqual(
            float(normalized[0, 1]),
            0.0,
        )

    def test_normalization_matches_frozen_census_function(self) -> None:
        counts = np.asarray(
            [
                [2, 0, 8],
                [0, 5, 5],
                [1, 2, 3],
            ],
            dtype=np.uint32,
        )

        ours = cp10k_log1p_counts(counts)

        canonical = np.asarray(
            canonical_cp10k_log1p(
                sparse.csr_matrix(counts)
            ).toarray(),
            dtype=np.float32,
        )

        self.assertTrue(
            np.array_equal(ours, canonical)
        )

    def test_zero_library_rule_is_stop(self) -> None:
        counts = np.asarray(
            [
                [1, 2],
                [0, 0],
            ],
            dtype=np.uint32,
        )

        with self.assertRaises(ValueError):
            cp10k_log1p_counts(
                counts,
                zero_library_rule="error",
            )

    def test_identity_hash_is_deterministic(self) -> None:
        shard = np.asarray(
            ["a", "a", "b"]
        )
        row = np.asarray(
            [1, 2, 3],
            dtype=np.int64,
        )

        self.assertEqual(
            identity_sha256(shard, row),
            identity_sha256(shard, row),
        )


if __name__ == "__main__":
    unittest.main()
