#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from sc2.eval.p2_selective import (
    choose_exact_threshold,
    exact_threshold_frontier,
    presentation_frontier,
    score_discrimination,
)

from sc2.eval.selective_repair_metrics import (
    masked_value_metrics,
    risk_coverage_curve,
)


RANDOM_SEED = 20260728
EXPECTED_SHAPE = (2500, 4096)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sha256_array_float32(
    x: np.ndarray,
) -> str:
    arr = np.asarray(
        x,
        dtype="<f4",
        order="C",
    )

    return hashlib.sha256(
        arr.tobytes(order="C")
    ).hexdigest()


def prefixed(
    values: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in values.items()
    }


def read_metadata(
    path: Path,
) -> dict[str, str]:
    out: dict[str, str] = {}

    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue

        key, value = raw.split(
            "\t",
            1,
        )

        out[key] = value

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--benchmark",
        required=True,
    )

    p.add_argument(
        "--benchmark-sha256",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--alra-r-script",
        required=True,
    )

    p.add_argument(
        "--alra-rlib",
        required=True,
    )

    p.add_argument(
        "--rscript",
        default="Rscript",
    )

    p.add_argument(
        "--max-true-zero-fill",
        type=float,
        default=0.02,
    )

    p.add_argument(
        "--zero-threshold",
        type=float,
        default=1.0e-8,
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    benchmark_path = Path(
        args.benchmark
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    alra_r_script = Path(
        args.alra_r_script
    ).resolve()

    alra_rlib = Path(
        args.alra_rlib
    ).resolve()

    if output_dir.exists():
        raise FileExistsError(
            output_dir
        )

    if not alra_r_script.exists():
        raise FileNotFoundError(
            alra_r_script
        )

    if not alra_rlib.exists():
        raise FileNotFoundError(
            alra_rlib
        )

    observed_hash = sha256_file(
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
        order="C",
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

    if x.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected validation shape: "
            f"{x.shape}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "x contains non-finite values"
        )

    if np.any(x < 0):
        raise ValueError(
            "x contains negative values"
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
            "Masked-positive entry is "
            "nonzero in corrupted x"
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

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # R receives x only.
    # The final output directory is not created
    # unless ALRA succeeds completely.
    #
    with tempfile.TemporaryDirectory(
        prefix="p2_alra_",
        dir=str(output_dir.parent),
    ) as tmp_name:

        tmp = Path(tmp_name)

        input_bin = (
            tmp / "input_x.float32.bin"
        )

        completed_bin = (
            tmp / "completed.float32.bin"
        )

        metadata_tsv = (
            tmp / "alra_metadata.tsv"
        )

        singular_values_csv = (
            tmp
            / "choose_k_singular_values.csv"
        )

        num_of_sds_csv = (
            tmp
            / "choose_k_num_of_sds.csv"
        )

        x.astype(
            "<f4",
            copy=False,
        ).tofile(input_bin)

        env = os.environ.copy()
        env["R_LIBS_USER"] = str(
            alra_rlib
        )

        command = [
            args.rscript,
            "--vanilla",
            str(alra_r_script),
            str(input_bin),
            str(x.shape[0]),
            str(x.shape[1]),
            str(completed_bin),
            str(metadata_tsv),
            str(singular_values_csv),
            str(num_of_sds_csv),
        ]

        proc = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        if proc.returncode != 0:
            print(
                "===== ALRA STDOUT ====="
            )
            print(proc.stdout)

            print(
                "===== ALRA STDERR ====="
            )
            print(proc.stderr)

            raise RuntimeError(
                f"ALRA R process failed "
                f"with exit code "
                f"{proc.returncode}"
            )

        if not completed_bin.exists():
            raise FileNotFoundError(
                completed_bin
            )

        raw = np.fromfile(
            completed_bin,
            dtype="<f4",
        )

        expected = (
            x.shape[0]
            * x.shape[1]
        )

        if raw.size != expected:
            raise RuntimeError(
                f"ALRA output has "
                f"{raw.size} values; "
                f"expected {expected}"
            )

        raw_prediction = (
            raw.reshape(
                x.shape,
                order="C",
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        if not np.all(
            np.isfinite(
                raw_prediction
            )
        ):
            raise RuntimeError(
                "Non-finite ALRA output"
            )

        if np.any(
            raw_prediction < 0
        ):
            raise RuntimeError(
                "Negative ALRA output"
            )

        metadata = read_metadata(
            metadata_tsv
        )

        chosen_k = int(
            metadata["chosen_k"]
        )

        if chosen_k <= 0:
            raise RuntimeError(
                "Invalid ALRA chosen k"
            )

        #
        # y is first used only after the
        # x-only ALRA fit has completed.
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

        if target_variance <= 0:
            raise RuntimeError(
                "Masked target variance "
                "is zero"
            )

        recovery_r = (
            1.0
            - float(
                raw_metrics[
                    "masked_mse"
                ]
            )
            / target_variance
        )

        #
        # Frozen ALRA score:
        # completed normalized expression.
        #
        score = raw_prediction

        score_metrics = (
            score_discrimination(
                score,
                positive,
                true_zero,
            )
        )

        frontier = (
            exact_threshold_frontier(
                score,
                positive,
                true_zero,
            )
        )

        selected = (
            choose_exact_threshold(
                frontier,
                max_true_zero_fill=float(
                    args.max_true_zero_fill
                ),
            )
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

        reconstruction[
            selected_repair
        ] = raw_prediction[
            selected_repair
        ]

        observed_nonzero = (
            np.abs(x)
            > zero_threshold
        )

        preservation_error = (
            reconstruction[
                observed_nonzero
            ]
            - x[
                observed_nonzero
            ]
        )

        if (
            preservation_error.size
            == 0
        ):
            raise RuntimeError(
                "No observed nonzero "
                "entries"
            )

        preservation = {
            "observed_nonzero_mse":
                float(
                    np.mean(
                        preservation_error
                        ** 2
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
                "Observed-nonzero "
                "preservation failed"
            )

        selective_metrics = (
            masked_value_metrics(
                reconstruction,
                y,
                positive,
            )
        )

        summary: dict[str, Any] = {
            "method":
                "alra",

            "classification":
                "confirmatory_validation_only",

            "family":
                "established_transductive_unsupervised",

            "transductive":
                True,

            "fit_input":
                "corrupted normalized log1p(CP10K) x only",

            "target_used_for_fit":
                False,

            "implementation":
                "KlugerLab/ALRA",

            "random_seed":
                RANDOM_SEED,

            "automatic_choose_k":
                True,

            "chosen_k":
                chosen_k,

            "choose_k_K":
                int(
                    metadata[
                        "choose_k_K"
                    ]
                ),

            "choose_k_thresh":
                float(
                    metadata[
                        "choose_k_thresh"
                    ]
                ),

            "choose_k_noise_start":
                int(
                    metadata[
                        "choose_k_noise_start"
                    ]
                ),

            "choose_k_q":
                int(
                    metadata[
                        "choose_k_q"
                    ]
                ),

            "alra_q":
                int(
                    metadata[
                        "alra_q"
                    ]
                ),

            "quantile_prob":
                float(
                    metadata[
                        "quantile_prob"
                    ]
                ),

            "value_prediction":
                "ALRA completed normalized expression",

            "repair_score":
                "ALRA completed normalized expression",

            "score_scale":
                "unbounded_nonnegative_expression",

            "score_calibration_applicable":
                False,

            "threshold":
                threshold,

            "threshold_recall":
                float(
                    selected["recall"]
                ),

            "threshold_precision":
                float(
                    selected[
                        "precision"
                    ]
                ),

            "threshold_true_zero_fill":
                float(
                    selected[
                        "true_zero_fill"
                    ]
                ),

            "threshold_tp":
                int(
                    selected["tp"]
                ),

            "threshold_fp":
                int(
                    selected["fp"]
                ),

            "threshold_selected":
                int(
                    selected[
                        "selected"
                    ]
                ),

            "target_variance":
                target_variance,

            "recovery_r":
                recovery_r,

            "input_nonzero_fraction":
                float(
                    metadata[
                        "input_nonzero_fraction"
                    ]
                ),

            "completed_nonzero_fraction":
                float(
                    metadata[
                        "completed_nonzero_fraction"
                    ]
                ),

            "completed_min":
                float(
                    metadata[
                        "completed_min"
                    ]
                ),

            "completed_max":
                float(
                    metadata[
                        "completed_max"
                    ]
                ),

            "prediction_float32_sha256":
                sha256_array_float32(
                    raw_prediction
                ),

            "benchmark":
                str(
                    benchmark_path
                ),

            "benchmark_sha256":
                observed_hash,

            **raw_metrics,

            **prefixed(
                selective_metrics,
                "selective_",
            ),

            **preservation,

            **prefixed(
                score_metrics,
                "score_",
            ),
        }

        #
        # Validate strict JSON before we
        # materialize confirmatory output.
        #
        summary_json = json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"

        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        (
            output_dir
            / "summary.json"
        ).write_text(
            summary_json,
            encoding="utf-8",
        )

        pd.DataFrame(
            [summary]
        ).to_csv(
            output_dir
            / "summary.csv",
            index=False,
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

        shutil.copy2(
            metadata_tsv,
            output_dir
            / "alra_metadata.tsv",
        )

        shutil.copy2(
            singular_values_csv,
            output_dir
            / "choose_k_singular_values.csv",
        )

        shutil.copy2(
            num_of_sds_csv,
            output_dir
            / "choose_k_num_of_sds.csv",
        )

        (
            output_dir
            / "alra_stdout.txt"
        ).write_text(
            proc.stdout,
            encoding="utf-8",
        )

        (
            output_dir
            / "alra_stderr.txt"
        ).write_text(
            proc.stderr,
            encoding="utf-8",
        )

    files = (
        "summary.json",
        "summary.csv",
        "exact_threshold_frontier.csv",
        "threshold_curve.csv",
        "risk_coverage.csv",
        "alra_metadata.tsv",
        "choose_k_singular_values.csv",
        "choose_k_num_of_sds.csv",
        "alra_stdout.txt",
        "alra_stderr.txt",
    )

    with (
        output_dir
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        for name in files:
            path = (
                output_dir
                / name
            )

            handle.write(
                f"{sha256_file(path)}"
                f"  {name}\n"
            )

    print(
        summary_json,
        end="",
    )

    print(
        "P2_ALRA_VALIDATION=PASS"
    )


if __name__ == "__main__":
    main()
