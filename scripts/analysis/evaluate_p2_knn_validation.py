#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sc2.eval.p2_knn import (
    knn_value_and_score,
    neighbor_indices_excluding_self,
)

from sc2.eval.p2_selective import (
    choose_exact_threshold,
    exact_threshold_frontier,
    presentation_frontier,
)

from sc2.eval.selective_repair_metrics import (
    gate_discrimination,
    masked_value_metrics,
    risk_coverage_curve,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def prefixed(
    values: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in values.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only transductive "
            "kNN comparator for p2."
        )
    )

    parser.add_argument(
        "--benchmark",
        required=True,
    )

    parser.add_argument(
        "--benchmark-sha256",
        required=True,
    )

    parser.add_argument(
        "--k",
        required=True,
        type=int,
        choices=(5, 10, 20, 50),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--max-true-zero-fill",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1.0e-8,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    benchmark_path = Path(
        args.benchmark
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if output_dir.exists():
        raise FileExistsError(
            output_dir
        )

    observed_hash = sha256(
        benchmark_path
    )

    if observed_hash != args.benchmark_sha256:
        raise RuntimeError(
            "Benchmark SHA256 mismatch"
        )

    panel = np.load(
        benchmark_path,
        allow_pickle=False,
    )

    x = np.asarray(
        panel["x"],
        dtype=np.float32,
    )

    y = np.asarray(
        panel["y"],
        dtype=np.float32,
    )

    positive = np.asarray(
        panel["synthetic_mask"],
        dtype=bool,
    )

    if (
        x.shape != y.shape
        or x.shape != positive.shape
    ):
        raise ValueError(
            "Panel shape mismatch"
        )

    if x.shape != (2500, 4096):
        raise ValueError(
            f"Unexpected validation shape: "
            f"{x.shape}"
        )

    zero_threshold = float(
        args.zero_threshold
    )

    if np.any(
        positive
        & (
            np.abs(x)
            > zero_threshold
        )
    ):
        raise ValueError(
            "Masked positive is nonzero in x"
        )

    true_zero = (
        y <= zero_threshold
    )

    if np.any(
        positive & true_zero
    ):
        raise ValueError(
            "positive and true_zero overlap"
        )

    eligible_zero = (
        np.abs(x)
        <= zero_threshold
    )

    #
    # Fit/query uses corrupted x only.
    #
    indices, distances = (
        neighbor_indices_excluding_self(
            x,
            k=int(args.k),
        )
    )

    raw_prediction, score = (
        knn_value_and_score(
            x,
            indices,
            zero_threshold=zero_threshold,
            chunk_size=int(
                args.chunk_size
            ),
        )
    )

    #
    # y is first used here, after neighbor
    # selection and predictions exist.
    #
    raw_metrics = masked_value_metrics(
        raw_prediction,
        y,
        positive,
    )

    target_variance = float(
        np.var(
            np.asarray(
                y[positive],
                dtype=np.float64,
            ),
            ddof=0,
        )
    )

    if target_variance <= 0.0:
        raise RuntimeError(
            "Masked target variance is zero"
        )

    recovery_r = (
        1.0
        - float(
            raw_metrics["masked_mse"]
        )
        / target_variance
    )

    frontier = exact_threshold_frontier(
        score,
        positive,
        true_zero,
    )

    selected = choose_exact_threshold(
        frontier,
        max_true_zero_fill=float(
            args.max_true_zero_fill
        ),
    )

    threshold = float(
        selected["threshold"]
    )

    selected_repair = (
        eligible_zero
        & (
            score >= threshold
        )
    )

    reconstruction = x.copy()

    reconstruction[selected_repair] = (
        raw_prediction[selected_repair]
    )

    observed_nonzero = (
        np.abs(x) > zero_threshold
    )

    preservation_error = (
        reconstruction[observed_nonzero]
        - x[observed_nonzero]
    )

    preservation = {
        "observed_nonzero_mse":
            float(
                np.mean(
                    preservation_error ** 2
                )
            ),
        "observed_nonzero_mae":
            float(
                np.mean(
                    np.abs(
                        preservation_error
                    )
                )
            ),
        "observed_nonzero_max_abs_error":
            float(
                np.max(
                    np.abs(
                        preservation_error
                    )
                )
            ),
        "observed_nonzero_changed_fraction":
            float(
                np.mean(
                    preservation_error
                    != 0.0
                )
            ),
        "n_observed_nonzero":
            int(
                preservation_error.size
            ),
    }

    if (
        preservation[
            "observed_nonzero_max_abs_error"
        ]
        != 0.0
    ):
        raise RuntimeError(
            "Observed nonzero preservation failed"
        )

    selective_metrics = (
        masked_value_metrics(
            reconstruction,
            y,
            positive,
        )
    )

    #
    # kNN expression frequency is naturally
    # bounded in [0,1].
    #
    gate_metrics = gate_discrimination(
        score,
        positive,
        true_zero,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    frontier.to_csv(
        output_dir
        / "exact_threshold_frontier.csv",
        index=False,
    )

    presentation_frontier(
        frontier,
        threshold,
    ).to_csv(
        output_dir
        / "threshold_curve.csv",
        index=False,
    )

    risk_coverage_curve(
        raw_prediction,
        y,
        positive,
        score,
    ).to_csv(
        output_dir
        / "risk_coverage.csv",
        index=False,
    )

    summary: dict[str, Any] = {
        "method":
            "transductive_knn",
        "classification":
            "confirmatory_validation_only",
        "transductive":
            True,
        "fit_input":
            "corrupted validation x only",
        "target_used_for_fit":
            False,
        "k":
            int(args.k),
        "metric":
            "cosine",
        "algorithm":
            "brute",
        "self_neighbor_excluded":
            True,
        "value_prediction":
            "arithmetic mean of k neighbor corrupted expression values",
        "repair_score":
            "fraction of k neighbors with expression > 1e-8",
        "score_scale":
            "bounded_0_1_neighbor_fraction",
        "threshold":
            threshold,
        "threshold_recall":
            float(
                selected["recall"]
            ),
        "threshold_precision":
            float(
                selected["precision"]
            ),
        "threshold_true_zero_fill":
            float(
                selected[
                    "true_zero_fill"
                ]
            ),
        "threshold_tp":
            int(selected["tp"]),
        "threshold_fp":
            int(selected["fp"]),
        "threshold_selected":
            int(
                selected["selected"]
            ),
        "target_variance":
            target_variance,
        "recovery_r":
            recovery_r,
        "neighbor_distance_min":
            float(
                np.min(distances)
            ),
        "neighbor_distance_max":
            float(
                np.max(distances)
            ),
        "benchmark":
            str(benchmark_path),
        "benchmark_sha256":
            observed_hash,
        **raw_metrics,
        **prefixed(
            selective_metrics,
            "selective_",
        ),
        **preservation,
        **prefixed(
            gate_metrics,
            "gate_",
        ),
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    files = (
        "summary.json",
        "summary.csv",
        "exact_threshold_frontier.csv",
        "threshold_curve.csv",
        "risk_coverage.csv",
    )

    with (
        output_dir
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for name in files:
            path = output_dir / name

            handle.write(
                f"{sha256(path)}  {name}\n"
            )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "P2_KNN_VALIDATION=PASS"
    )


if __name__ == "__main__":
    main()
