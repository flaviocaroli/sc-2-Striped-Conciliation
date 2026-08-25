#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (
    20260728,
    20260729,
    20260730,
)

MASKS = (
    15,
    30,
    50,
)

SELECTED_N_LATENT = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def verify_bundle(root: Path) -> None:
    sums = root / "SHA256SUMS.txt"

    if not sums.is_file():
        raise RuntimeError(
            f"Missing SHA256SUMS.txt: {root}"
        )

    for raw in sums.read_text().splitlines():
        if not raw.strip():
            continue

        digest, name = raw.split(
            None,
            1,
        )

        name = name.strip()

        observed = sha256_file(
            root / name
        )

        if observed != digest:
            raise RuntimeError(
                f"SHA mismatch: {root / name}"
            )


def get_run_dir(
    tuning_root: Path,
    confirmatory_root: Path,
    *,
    seed: int,
    mask: int,
) -> Path:

    if seed == 20260728:
        return (
            tuning_root
            / "latent10"
            / f"mask{mask}"
        )

    return (
        confirmatory_root
        / f"seed{seed}"
        / f"mask{mask}"
    )


def metrics_at_candidates(
    frontier: pd.DataFrame,
    candidates: np.ndarray,
) -> dict[str, np.ndarray]:

    thresholds = np.asarray(
        frontier["threshold"],
        dtype=np.float64,
    )

    if thresholds.ndim != 1:
        raise RuntimeError(
            "Invalid threshold vector"
        )

    if not np.all(
        thresholds[:-1]
        >= thresholds[1:]
    ):
        raise RuntimeError(
            "Threshold frontier is not "
            "descending"
        )

    ascending = thresholds[::-1]

    #
    # For score >= candidate:
    # choose the smallest exact stored
    # threshold >= candidate.
    #
    idx = np.searchsorted(
        ascending,
        candidates,
        side="left",
    )

    valid = (
        idx < ascending.size
    )

    out: dict[str, np.ndarray] = {}

    for field in (
        "recall",
        "precision",
        "true_zero_fill",
        "tp",
        "fp",
        "selected",
    ):
        values = np.asarray(
            frontier[field],
            dtype=np.float64,
        )[::-1]

        result = np.zeros(
            candidates.size,
            dtype=np.float64,
        )

        result[valid] = (
            values[
                idx[valid]
            ]
        )

        out[field] = result

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--tuning-root",
        required=True,
    )

    p.add_argument(
        "--confirmatory-root",
        required=True,
    )

    p.add_argument(
        "--protocol",
        required=True,
    )

    p.add_argument(
        "--nlatent-receipt",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    tuning_root = Path(
        args.tuning_root
    ).resolve()

    confirmatory_root = Path(
        args.confirmatory_root
    ).resolve()

    protocol_path = Path(
        args.protocol
    ).resolve()

    receipt_path = Path(
        args.nlatent_receipt
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if output_dir.exists():
        raise RuntimeError(
            f"Output exists: {output_dir}"
        )

    protocol = json.loads(
        protocol_path.read_text()
    )

    receipt = json.loads(
        receipt_path.read_text()
    )

    scvi_cfg = (
        protocol[
            "comparators"
        ]["scvi"]
    )

    frozen_seeds = tuple(
        int(x)
        for x
        in scvi_cfg[
            "confirmatory_seeds"
        ]
    )

    if frozen_seeds != SEEDS:
        raise RuntimeError(
            "Confirmatory seed mismatch"
        )

    if int(
        receipt[
            "selected_n_latent"
        ]
    ) != SELECTED_N_LATENT:
        raise RuntimeError(
            "Selected n_latent mismatch"
        )

    max_fill = float(
        protocol[
            "selective_repair_policy"
        ][
            "max_true_zero_fill"
        ]
    )

    if max_fill != 0.02:
        raise RuntimeError(
            "Unexpected true-zero fill limit"
        )

    selected_by_mask = {}
    operating_rows = []

    for mask in MASKS:
        frontiers = {}
        candidate_parts = []

        for seed in SEEDS:
            root = get_run_dir(
                tuning_root,
                confirmatory_root,
                seed=seed,
                mask=mask,
            )

            verify_bundle(root)

            summary = json.loads(
                (
                    root
                    / "summary.json"
                ).read_text()
            )

            if summary["method"] != "scvi":
                raise RuntimeError(
                    "Method mismatch"
                )

            if int(
                summary["random_seed"]
            ) != seed:
                raise RuntimeError(
                    "Seed mismatch"
                )

            if int(
                summary["n_latent"]
            ) != 10:
                raise RuntimeError(
                    "n_latent mismatch"
                )

            if int(
                summary["mask_percent"]
            ) != mask:
                raise RuntimeError(
                    "Mask mismatch"
                )

            if (
                summary[
                    "target_used_for_fit"
                ]
                is not False
            ):
                raise RuntimeError(
                    "Target-access violation"
                )

            frontier = pd.read_csv(
                root
                / "exact_threshold_frontier.csv"
            )

            required = {
                "threshold",
                "tp",
                "fp",
                "selected",
                "recall",
                "precision",
                "true_zero_fill",
            }

            if not required.issubset(
                set(frontier.columns)
            ):
                raise RuntimeError(
                    "Frontier schema mismatch"
                )

            frontiers[seed] = frontier

            candidate_parts.append(
                np.asarray(
                    frontier["threshold"],
                    dtype=np.float64,
                )
            )

        candidates = np.unique(
            np.concatenate(
                candidate_parts
            )
        )

        mean_recall = np.zeros(
            candidates.size,
            dtype=np.float64,
        )

        mean_precision = np.zeros(
            candidates.size,
            dtype=np.float64,
        )

        feasible = np.ones(
            candidates.size,
            dtype=bool,
        )

        metrics_by_seed = {}

        for seed in SEEDS:
            metrics = (
                metrics_at_candidates(
                    frontiers[seed],
                    candidates,
                )
            )

            metrics_by_seed[
                seed
            ] = metrics

            mean_recall += (
                metrics["recall"]
                / 3.0
            )

            mean_precision += (
                metrics["precision"]
                / 3.0
            )

            feasible &= (
                metrics[
                    "true_zero_fill"
                ]
                <= max_fill
                + 1.0e-15
            )

        feasible_idx = np.flatnonzero(
            feasible
        )

        if feasible_idx.size == 0:
            raise RuntimeError(
                f"No feasible threshold "
                f"for mask {mask}"
            )

        #
        # Frozen rule:
        # 1. highest mean recall
        # 2. highest mean precision
        # 3. higher/stricter threshold
        #
        best = max(
            feasible_idx.tolist(),
            key=lambda i: (
                float(
                    mean_recall[i]
                ),
                float(
                    mean_precision[i]
                ),
                float(
                    candidates[i]
                ),
            ),
        )

        threshold = float(
            candidates[best]
        )

        rows = []

        for seed in SEEDS:
            metrics = (
                metrics_by_seed[
                    seed
                ]
            )

            row = {
                "mask_percent":
                    mask,

                "seed":
                    seed,

                "threshold":
                    threshold,

                "recall":
                    float(
                        metrics[
                            "recall"
                        ][best]
                    ),

                "precision":
                    float(
                        metrics[
                            "precision"
                        ][best]
                    ),

                "true_zero_fill":
                    float(
                        metrics[
                            "true_zero_fill"
                        ][best]
                    ),

                "tp":
                    int(
                        metrics[
                            "tp"
                        ][best]
                    ),

                "fp":
                    int(
                        metrics[
                            "fp"
                        ][best]
                    ),

                "selected":
                    int(
                        metrics[
                            "selected"
                        ][best]
                    ),
            }

            if (
                row[
                    "true_zero_fill"
                ]
                > max_fill
                + 1.0e-15
            ):
                raise RuntimeError(
                    "Selected threshold violates "
                    "fill constraint"
                )

            rows.append(row)
            operating_rows.append(row)

        selected_by_mask[
            str(mask)
        ] = {
            "threshold":
                threshold,

            "mean_recall":
                float(
                    mean_recall[best]
                ),

            "mean_precision":
                float(
                    mean_precision[best]
                ),

            "max_seed_true_zero_fill":
                float(
                    max(
                        x[
                            "true_zero_fill"
                        ]
                        for x in rows
                    )
                ),

            "n_candidate_thresholds":
                int(
                    candidates.size
                ),

            "per_seed":
                {
                    str(x["seed"]): x
                    for x in rows
                },
        }

    result = {
        "schema_version":
            "sc2-p2-scvi-common-thresholds-v1",

        "classification":
            "confirmatory_validation_selection_only",

        "selected_n_latent":
            10,

        "confirmatory_seeds":
            list(SEEDS),

        "mask_rates":
            list(MASKS),

        "max_true_zero_fill":
            max_fill,

        "selection_rule":
            (
                "For each mask, maximize "
                "mean recall over seeds "
                "20260728/20260729/20260730 "
                "subject to true_zero_fill "
                "<=0.02 in every seed; ties "
                "maximize mean precision, "
                "then choose the higher "
                "threshold."
            ),

        "protocol_sha256":
            sha256_file(
                protocol_path
            ),

        "nlatent_receipt_sha256":
            sha256_file(
                receipt_path
            ),

        "thresholds":
            selected_by_mask,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        output_dir
        / "common_thresholds.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    pd.DataFrame(
        [
            {
                "mask_percent":
                    int(mask),

                "threshold":
                    selected_by_mask[
                        str(mask)
                    ]["threshold"],

                "mean_recall":
                    selected_by_mask[
                        str(mask)
                    ]["mean_recall"],

                "mean_precision":
                    selected_by_mask[
                        str(mask)
                    ]["mean_precision"],

                "max_seed_true_zero_fill":
                    selected_by_mask[
                        str(mask)
                    ][
                        "max_seed_true_zero_fill"
                    ],
            }
            for mask in MASKS
        ]
    ).to_csv(
        output_dir
        / "common_thresholds.csv",
        index=False,
    )

    pd.DataFrame(
        operating_rows
    ).to_csv(
        output_dir
        / "per_seed_operating_points.csv",
        index=False,
    )

    files = (
        "common_thresholds.json",
        "common_thresholds.csv",
        "per_seed_operating_points.csv",
    )

    with (
        output_dir
        / "SHA256SUMS.txt"
    ).open(
        "w"
    ) as handle:
        for name in files:
            handle.write(
                f"{sha256_file(output_dir / name)}"
                f"  {name}\n"
            )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    print(
        "P2_SCVI_COMMON_THRESHOLDS=PASS"
    )


if __name__ == "__main__":
    main()
