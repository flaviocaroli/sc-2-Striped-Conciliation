import pandas as pd

from sc2.data.census_pipeline import (
    cap_violations,
    largest_remainder_quotas,
    registry_exclusions,
    stable_group_split,
    validate_registry_payload,
)


def test_largest_remainder_exact_total() -> None:
    quotas = largest_remainder_quotas(7, {"train": 0.8, "validation": 0.1, "test": 0.1})
    assert sum(quotas.values()) == 7
    assert quotas == {"train": 5, "validation": 1, "test": 1}


def test_group_split_is_deterministic() -> None:
    fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert stable_group_split("d::x", seed=4, fractions=fractions) == stable_group_split(
        "d::x", seed=4, fractions=fractions
    )


def test_registry_requires_frozen_resolution() -> None:
    unresolved = {"benchmarks": [{"name": "x", "status": "resolve_exact_dataset_ids"}]}
    assert validate_registry_payload(unresolved, expected_release="2025-01-30")
    frozen = {
        "benchmarks": [
            {
                "name": "x",
                "status": "frozen",
                "dataset_ids": ["abc"],
                "resolution": {
                    "disposition": "exact_ids_frozen",
                    "census_release": "2025-01-30",
                    "evidence": "manual review",
                },
            }
        ]
    }
    assert validate_registry_payload(frozen, expected_release="2025-01-30") == []
    exclusions = registry_exclusions(frozen)
    assert "abc" in exclusions.dataset_ids


def test_cap_violation_detected() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["d", "d"],
            "split_group": ["d::x", "d::x"],
            "tissue": ["lung", "lung"],
            "cell_type": ["T", "T"],
        }
    )
    errors = cap_violations(
        frame,
        {"per_dataset": 1, "per_donor": 1, "per_tissue": 1, "per_tissue_cell_type": 1},
    )
    assert len(errors) == 4
