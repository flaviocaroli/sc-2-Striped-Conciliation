#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scvi
from scipy import sparse
import torch

from sc2.eval.p2_selective import (
    score_discrimination,
)

from sc2.eval.selective_repair_metrics import (
    masked_value_metrics,
    risk_coverage_curve,
)


EXPECTED_PROTOCOL_SHA256 = (
    "3f079aa5e756821dc5c75d88e95b0b377cfdbdf29afe0484e1360cd9e5f7f2b9"
)

EXPECTED_PRETEST_SHA256 = (
    "514b76785abf7fbe16e65a4d4753827e32f9c0469ffd07b3c6bbb94b4bf48452"
)

EXPECTED_SCVI_SELECTION_SHA256 = (
    "a0eb3bcd218b13ce544c2eda9cbc92f2592eb2e3b7d20db2e5417adb1ae13a68"
)

EXPECTED_NLATENT = 10

EXPECTED_SHAPE = (
    5000,
    4096,
)

ZERO_THRESHOLD = 1.0e-8

PANEL_HASHES = {
    "internal_test": {
        15:
            "eb9d3d63f42bc30fb5233f0f39271f792b43535a6be4f46b690cc188ed99c3d7",
        30:
            "ca59d9407830484360c476865c1b339b0ed62b8c4e0a75e8ca5f07acbc1d7ecf",
        50:
            "e1a75d8b27eda99ccf0ba08857fad385708ca0dd5cbdc0166ded9f1607b7a4d2",
    },

    "zheng68k": {
        15:
            "35490be0d1a83b46517d2d5792a2c0a1bc9c5b3f800f29aeb90447d11d8a6ecc",
        30:
            "ba6b8e6d2698048faac52e6b384a795bd429ec3d58bb385b1fc769ad50da7871",
        50:
            "2313651071d4c2d69053a51e21ba477d3f2c3c146172dc187ac809af4590f040",
    },

    "baron_pancreas": {
        15:
            "20dd3ef3454c91fad9ef121518827df4b64ced1adb5d1b405e38ab0f6acc498b",
        30:
            "edc1422a764c1aa40a9028401df6629c771fc7f3809ff5e9a7443be6fabd1e3e",
        50:
            "7654fcc78719327f73fcbe46bf69beaf866e32b9331134f2aec1bc4964b2d792",
    },
}


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sha256_float32(
    x: np.ndarray,
) -> str:
    a = np.asarray(
        x,
        dtype="<f4",
        order="C",
    )

    return hashlib.sha256(
        a.tobytes(
            order="C"
        )
    ).hexdigest()


def sha256_uint32(
    x: np.ndarray,
) -> str:
    a = np.asarray(
        x,
        dtype="<u4",
        order="C",
    )

    return hashlib.sha256(
        a.tobytes(
            order="C"
        )
    ).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def git_file_commit(
    path: str,
) -> str:
    return subprocess.check_output(
        [
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            path,
        ],
        text=True,
    ).strip()


def prefixed(
    values: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}{key}":
            value
        for key, value
        in values.items()
    }


def frozen_operating_point(
    score: np.ndarray,
    positive: np.ndarray,
    true_zero: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:

    if (
        score.shape
        != positive.shape
        or score.shape
        != true_zero.shape
    ):
        raise ValueError(
            "score/positive/true_zero "
            "shape mismatch"
        )

    if np.any(
        positive
        & true_zero
    ):
        raise ValueError(
            "positive and true_zero "
            "overlap"
        )

    eligible = (
        positive
        | true_zero
    )

    selected = (
        eligible
        & (
            score
            >= float(
                threshold
            )
        )
    )

    tp = int(
        np.sum(
            selected
            & positive
        )
    )

    fp = int(
        np.sum(
            selected
            & true_zero
        )
    )

    n_positive = int(
        positive.sum()
    )

    n_true_zero = int(
        true_zero.sum()
    )

    n_selected = int(
        selected.sum()
    )

    if (
        n_positive <= 0
        or n_true_zero <= 0
    ):
        raise ValueError(
            "Operating point requires "
            "positive and true-zero entries"
        )

    recall = (
        tp
        / n_positive
    )

    precision = (
        tp
        / n_selected
        if n_selected > 0
        else 0.0
    )

    fill = (
        fp
        / n_true_zero
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "threshold_tp":
            tp,

        "threshold_fp":
            fp,

        "threshold_selected":
            n_selected,

        "threshold_recall":
            float(
                recall
            ),

        "threshold_precision":
            float(
                precision
            ),

        "threshold_true_zero_fill":
            float(
                fill
            ),
    }


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        required=True,
        choices=(
            "internal_test",
            "zheng68k",
            "baron_pancreas",
        ),
    )

    p.add_argument(
        "--benchmark",
        required=True,
    )

    p.add_argument(
        "--mask-percent",
        required=True,
        type=int,
        choices=(
            15,
            30,
            50,
        ),
    )

    p.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    p.add_argument(
        "--protocol",
        required=True,
    )

    p.add_argument(
        "--pretest-receipt",
        required=True,
    )

    p.add_argument(
        "--scvi-selection-receipt",
        required=True,
    )

    p.add_argument(
        "--count-view-receipt",
        required=True,
    )

    p.add_argument(
        "--count-view-root",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    return p.parse_args()


def main():
    args = parse_args()

    benchmark_path = Path(
        args.benchmark
    ).resolve()

    protocol_path = Path(
        args.protocol
    ).resolve()

    pretest_path = Path(
        args.pretest_receipt
    ).resolve()

    selection_path = Path(
        args.scvi_selection_receipt
    ).resolve()

    count_receipt_path = Path(
        args.count_view_receipt
    ).resolve()

    count_root = Path(
        args.count_view_root
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: "
            f"{output_dir}"
        )

    if (
        sha256_file(
            protocol_path
        )
        != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "Protocol SHA mismatch"
        )

    if (
        sha256_file(
            pretest_path
        )
        != EXPECTED_PRETEST_SHA256
    ):
        raise RuntimeError(
            "Pre-test receipt SHA mismatch"
        )

    if (
        sha256_file(
            selection_path
        )
        != EXPECTED_SCVI_SELECTION_SHA256
    ):
        raise RuntimeError(
            "scVI selection receipt "
            "SHA mismatch"
        )

    benchmark_sha = sha256_file(
        benchmark_path
    )

    if (
        benchmark_sha
        != PANEL_HASHES[
            args.dataset
        ][
            args.mask_percent
        ]
    ):
        raise RuntimeError(
            "Frozen benchmark "
            "SHA mismatch"
        )

    protocol = json.loads(
        protocol_path.read_text()
    )

    selection = json.loads(
        selection_path.read_text()
    )

    count_receipt = json.loads(
        count_receipt_path.read_text()
    )

    cfg = (
        protocol[
            "comparators"
        ][
            "scvi"
        ]
    )

    confirmatory_seeds = tuple(
        int(
            value
        )
        for value
        in cfg[
            "confirmatory_seeds"
        ]
    )

    if (
        args.seed
        not in confirmatory_seeds
    ):
        raise RuntimeError(
            "Seed not in frozen "
            "confirmatory policy"
        )

    if (
        int(
            selection[
                "selected_n_latent"
            ]
        )
        != EXPECTED_NLATENT
    ):
        raise RuntimeError(
            "Selected n_latent mismatch"
        )

    threshold = float(
        selection[
            "common_thresholds"
        ][
            str(
                args.mask_percent
            )
        ][
            "threshold"
        ]
    )

    fixed = cfg[
        "model_fixed_parameters"
    ]

    max_epochs = int(
        cfg[
            "training"
        ][
            "max_epochs"
        ]
    )

    if max_epochs != 400:
        raise RuntimeError(
            "Unexpected max_epochs"
        )

    if (
        str(
            scvi.__version__
        )
        != str(
            cfg[
                "version"
            ]
        )
    ):
        raise RuntimeError(
            "scvi-tools version mismatch: "
            f"{scvi.__version__}"
        )

    receipt_dataset = (
        count_receipt[
            "datasets"
        ][
            args.dataset
        ]
    )

    bundle = (
        count_root
        / args.dataset
    )

    counts_path = (
        bundle
        / "counts_uint32.npy"
    )

    available_path = (
        bundle
        / "available_gene_mask.npy"
    )

    metadata_path = (
        bundle
        / "metadata.json"
    )

    for path in (
        counts_path,
        available_path,
        metadata_path,
        bundle
        / "SHA256SUMS.txt",
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Missing count-view "
                f"artifact: {path}"
            )

    if (
        sha256_file(
            counts_path
        )
        != receipt_dataset[
            "counts_file_sha256"
        ]
    ):
        raise RuntimeError(
            "Count file SHA mismatch"
        )

    if (
        sha256_file(
            available_path
        )
        != receipt_dataset[
            "available_file_sha256"
        ]
    ):
        raise RuntimeError(
            "Availability file SHA mismatch"
        )

    if (
        sha256_file(
            metadata_path
        )
        != receipt_dataset[
            "metadata_sha256"
        ]
    ):
        raise RuntimeError(
            "Count metadata SHA mismatch"
        )

    counts = np.asarray(
        np.load(
            counts_path,
            allow_pickle=False,
        ),
        dtype=np.uint32,
        order="C",
    )

    available_gene_mask = np.asarray(
        np.load(
            available_path,
            allow_pickle=False,
        ),
        dtype=bool,
    )

    metadata = json.loads(
        metadata_path.read_text()
    )

    if counts.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            "Count-view shape mismatch"
        )

    if (
        available_gene_mask.shape
        != (
            EXPECTED_SHAPE[
                1
            ],
        )
    ):
        raise RuntimeError(
            "Availability shape mismatch"
        )

    if (
        sha256_uint32(
            counts
        )
        != metadata[
            "counts_uint32_sha256"
        ]
    ):
        raise RuntimeError(
            "Semantic counts SHA mismatch"
        )

    # --------------------------------------------------------
    # Before fit, access frozen mask + availability only.
    # No benchmark target y is loaded here.
    # --------------------------------------------------------

    with np.load(
        benchmark_path,
        allow_pickle=False,
    ) as panel:

        synthetic_mask = np.asarray(
            panel[
                "synthetic_mask"
            ],
            dtype=bool,
        ).copy()

        if (
            synthetic_mask.shape
            != EXPECTED_SHAPE
        ):
            raise RuntimeError(
                "Synthetic mask "
                "shape mismatch"
            )

        if (
            "available_gene_mask"
            in panel.files
        ):

            panel_available = (
                np.asarray(
                    panel[
                        "available_gene_mask"
                    ],
                    dtype=bool,
                )
            )

            if not np.array_equal(
                panel_available,
                available_gene_mask,
            ):
                raise RuntimeError(
                    "Panel/count-view "
                    "availability mismatch"
                )

        elif not np.all(
            available_gene_mask
        ):
            raise RuntimeError(
                "External availability "
                "missing from panel"
            )

    if np.any(
        synthetic_mask
        & (
            ~available_gene_mask[
                None,
                :
            ]
        )
    ):
        raise RuntimeError(
            "Mask includes unavailable gene"
        )

    if not np.all(
        counts[
            synthetic_mask
        ]
        > 0
    ):
        raise RuntimeError(
            "Mask includes "
            "nonpositive raw count"
        )

    corrupted_counts = (
        counts.copy()
    )

    corrupted_counts[
        synthetic_mask
    ] = 0

    if np.any(
        corrupted_counts[
            synthetic_mask
        ]
        != 0
    ):
        raise RuntimeError(
            "Failed to zero "
            "masked counts"
        )

    if not np.array_equal(
        corrupted_counts[
            ~synthetic_mask
        ],
        counts[
            ~synthetic_mask
        ],
    ):
        raise RuntimeError(
            "Counts changed "
            "outside mask"
        )

    if np.any(
        corrupted_counts.sum(
            axis=1,
            dtype=np.uint64,
        )
        == 0
    ):
        raise RuntimeError(
            "Zero-library cell "
            "after corruption"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    # --------------------------------------------------------
    # Frozen scVI fit.
    # --------------------------------------------------------

    scvi.settings.seed = int(
        args.seed
    )

    adata = ad.AnnData(
        X=sparse.csr_matrix(
            corrupted_counts
        )
    )

    adata.obs_names = [
        f"cell_{index:05d}"
        for index in range(
            EXPECTED_SHAPE[
                0
            ]
        )
    ]

    adata.var_names = [
        f"gene_{index:04d}"
        for index in range(
            EXPECTED_SHAPE[
                1
            ]
        )
    ]

    SCVI = scvi.model.SCVI

    SCVI.setup_anndata(
        adata,
        batch_key=cfg[
            "batch_key"
        ],
    )

    model = SCVI(
        adata,

        n_hidden=int(
            fixed[
                "n_hidden"
            ]
        ),

        n_latent=EXPECTED_NLATENT,

        n_layers=int(
            fixed[
                "n_layers"
            ]
        ),

        dispersion=str(
            fixed[
                "dispersion"
            ]
        ),

        gene_likelihood=str(
            fixed[
                "gene_likelihood"
            ]
        ),

        use_observed_lib_size=bool(
            fixed[
                "use_observed_lib_size"
            ]
        ),
    )

    # Do not specify train_size or other
    # training options: frozen pinned defaults.
    model.train(
        max_epochs=max_epochs
    )

    decoded = np.asarray(
        model.get_normalized_expression(
            library_size=10000,
            return_numpy=True,
        ),
        dtype=np.float32,
        order="C",
    )

    if (
        decoded.shape
        != EXPECTED_SHAPE
        or not np.all(
            np.isfinite(
                decoded
            )
        )
        or np.any(
            decoded < 0
        )
    ):
        raise RuntimeError(
            "Invalid decoded expression"
        )

    prediction = np.asarray(
        np.log1p(
            decoded
        ),
        dtype=np.float32,
        order="C",
    )

    # --------------------------------------------------------
    # Only now access x/y for evaluation.
    # --------------------------------------------------------

    with np.load(
        benchmark_path,
        allow_pickle=False,
    ) as panel:

        x = np.asarray(
            panel["x"],
            dtype=np.float32,
        )

        y = np.asarray(
            panel["y"],
            dtype=np.float32,
        )

    if (
        x.shape
        != EXPECTED_SHAPE
        or y.shape
        != EXPECTED_SHAPE
    ):
        raise RuntimeError(
            "Benchmark x/y "
            "shape mismatch"
        )

    if np.any(
        np.abs(
            x[
                synthetic_mask
            ]
        )
        > ZERO_THRESHOLD
    ):
        raise RuntimeError(
            "Masked x positions "
            "are nonzero"
        )

    if not np.all(
        y[
            synthetic_mask
        ]
        > ZERO_THRESHOLD
    ):
        raise RuntimeError(
            "Masked target "
            "is not positive"
        )

    if not np.array_equal(
        x[
            ~synthetic_mask
        ],
        y[
            ~synthetic_mask
        ],
    ):
        raise RuntimeError(
            "x/y differ outside mask"
        )

    positive = (
        synthetic_mask
        & available_gene_mask[
            None,
            :
        ]
    )

    true_zero = (
        (
            y
            <= ZERO_THRESHOLD
        )
        & available_gene_mask[
            None,
            :
        ]
        & (
            ~positive
        )
    )

    eligible_zero = (
        positive
        | true_zero
    )

    if np.any(
        positive
        & true_zero
    ):
        raise RuntimeError(
            "Positive/zero overlap"
        )

    if not np.all(
        np.abs(
            x[
                eligible_zero
            ]
        )
        <= ZERO_THRESHOLD
    ):
        raise RuntimeError(
            "Eligible repair set "
            "contains observed nonzero"
        )

    raw_metrics = (
        masked_value_metrics(
            prediction,
            y,
            positive,
        )
    )

    target_variance = float(
        np.var(
            np.asarray(
                y[
                    positive
                ],
                dtype=np.float64,
            ),
            ddof=0,
        )
    )

    if target_variance <= 0:
        raise RuntimeError(
            "Target variance "
            "nonpositive"
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

    score_metrics = (
        score_discrimination(
            prediction,
            positive,
            true_zero,
        )
    )

    operating = (
        frozen_operating_point(
            prediction,
            positive,
            true_zero,
            threshold,
        )
    )

    selected_repair = (
        eligible_zero
        & (
            prediction
            >= threshold
        )
    )

    reconstruction = (
        x.copy()
    )

    reconstruction[
        selected_repair
    ] = prediction[
        selected_repair
    ]

    observed_nonzero = (
        (
            np.abs(
                x
            )
            > ZERO_THRESHOLD
        )
        & available_gene_mask[
            None,
            :
        ]
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
            "No observed nonzeros"
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

    risk = risk_coverage_curve(
        prediction,
        y,
        positive,
        prediction,
    )

    summary = {
        "method":
            "scvi",

        "classification":
            "confirmatory_post_pretest_freeze",

        "dataset":
            args.dataset,

        "mask_percent":
            int(
                args.mask_percent
            ),

        "random_seed":
            int(
                args.seed
            ),

        "family":
            cfg[
                "family"
            ],

        "transductive":
            True,

        "target_used_for_fit":
            False,

        "target_loaded_after_fit_and_decode":
            True,

        "implementation":
            "scvi-tools",

        "scvi_version":
            str(
                scvi.__version__
            ),

        "torch_version":
            str(
                torch.__version__
            ),

        "cuda_device":
            str(
                torch.cuda.get_device_name(
                    0
                )
            ),

        "n_latent":
            EXPECTED_NLATENT,

        "n_hidden":
            int(
                fixed[
                    "n_hidden"
                ]
            ),

        "n_layers":
            int(
                fixed[
                    "n_layers"
                ]
            ),

        "dispersion":
            str(
                fixed[
                    "dispersion"
                ]
            ),

        "gene_likelihood":
            str(
                fixed[
                    "gene_likelihood"
                ]
            ),

        "use_observed_lib_size":
            bool(
                fixed[
                    "use_observed_lib_size"
                ]
            ),

        "max_epochs":
            max_epochs,

        "value_prediction":
            (
                "log1p(scVI "
                "get_normalized_expression "
                "library_size=10000)"
            ),

        "repair_score":
            (
                "same log1p decoded "
                "normalized expression"
            ),

        "score_scale":
            (
                "unbounded_nonnegative_"
                "log1p_expression"
            ),

        "score_calibration_applicable":
            False,

        "threshold":
            threshold,

        "target_variance":
            target_variance,

        "recovery_r":
            recovery_r,

        "n_available_genes":
            int(
                available_gene_mask.sum()
            ),

        "input_clean_counts_uint32_sha256":
            metadata[
                "counts_uint32_sha256"
            ],

        "input_corrupted_counts_uint32_sha256":
            sha256_uint32(
                corrupted_counts
            ),

        "prediction_float32_sha256":
            sha256_float32(
                prediction
            ),

        "decoded_min":
            float(
                decoded.min()
            ),

        "decoded_max":
            float(
                decoded.max()
            ),

        "benchmark":
            str(
                benchmark_path
            ),

        "benchmark_sha256":
            benchmark_sha,

        "protocol_sha256":
            sha256_file(
                protocol_path
            ),

        "pretest_receipt_sha256":
            sha256_file(
                pretest_path
            ),

        "scvi_selection_receipt_sha256":
            sha256_file(
                selection_path
            ),

        "count_view_receipt":
            str(
                count_receipt_path
            ),

        "count_view_receipt_sha256":
            sha256_file(
                count_receipt_path
            ),

        "implementation_git_commit":
            git_head(),

        "evaluator_git_commit":
            git_file_commit(
                "scripts/analysis/"
                "evaluate_p2_scvi_final.py"
            ),

        **operating,

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

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    risk.to_csv(
        output_dir
        / "risk_coverage.csv",
        index=False,
    )

    files = (
        "summary.json",
        "summary.csv",
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
            handle.write(
                f"{sha256_file(output_dir / name)}"
                f"  {name}\n"
            )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    print(
        "P2_SCVI_FINAL=PASS"
    )


if __name__ == "__main__":
    main()
