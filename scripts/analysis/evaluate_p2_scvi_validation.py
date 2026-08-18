#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scvi
from scipy import sparse
import torch

from sc2.data.csr_shard import CSRMemmap
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


EXPECTED_PROTOCOL_SHA256 = (
    "3f079aa5e756821dc5c75d88e95b0b377cfdbdf29afe0484e1360cd9e5f7f2b9"
)

EXPECTED_ALIGNMENT_RECEIPT_SHA256 = (
    "ca1548005991533bdfa194a45d9eaed7b1150277d535933b61c3b0dee1519f74"
)

EXPECTED_SHAPE = (
    2500,
    4096,
)

EXPECTED_SHARD_ID = (
    "sc_validation_00000"
)


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


def sha256_dense_uint32(
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


def sha256_array_float32(
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


def prefixed(
    values: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in values.items()
    }


def git_head() -> str:
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def first_validation_counts(
    corpus: Path,
) -> np.ndarray:
    root = (
        corpus
        / "shards"
        / EXPECTED_SHARD_ID
    )

    mm = CSRMemmap.open(
        root,
        "counts",
    )

    if tuple(mm.shape) != (
        25000,
        EXPECTED_SHAPE[1],
    ):
        raise RuntimeError(
            f"Unexpected count shard shape "
            f"{mm.shape}"
        )

    n_rows = EXPECTED_SHAPE[0]

    stop = int(
        mm.indptr[n_rows]
    )

    matrix = sparse.csr_matrix(
        (
            np.array(
                mm.data[:stop],
                copy=True,
            ),
            np.array(
                mm.indices[:stop],
                dtype=np.int32,
                copy=True,
            ),
            np.array(
                mm.indptr[
                    : n_rows + 1
                ],
                dtype=np.int64,
                copy=True,
            ),
        ),
        shape=EXPECTED_SHAPE,
    )

    dense = np.asarray(
        matrix.toarray(),
        dtype=np.uint32,
        order="C",
    )

    if dense.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            "Unexpected dense-count shape"
        )

    return dense


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--corpus",
        required=True,
    )

    p.add_argument(
        "--benchmark",
        required=True,
    )

    p.add_argument(
        "--protocol",
        required=True,
    )

    p.add_argument(
        "--alignment-receipt",
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
        "--n-latent",
        required=True,
        type=int,
    )

    p.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    corpus = Path(
        args.corpus
    ).resolve()

    benchmark_path = Path(
        args.benchmark
    ).resolve()

    protocol_path = Path(
        args.protocol
    ).resolve()

    alignment_receipt_path = Path(
        args.alignment_receipt
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: "
            f"{output_dir}"
        )

    protocol_sha = sha256_file(
        protocol_path
    )

    if (
        protocol_sha
        != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "Frozen comparator protocol SHA "
            "mismatch"
        )

    alignment_sha = sha256_file(
        alignment_receipt_path
    )

    if (
        alignment_sha
        != EXPECTED_ALIGNMENT_RECEIPT_SHA256
    ):
        raise RuntimeError(
            "Frozen alignment receipt SHA "
            "mismatch"
        )

    protocol = json.loads(
        protocol_path.read_text()
    )

    alignment = json.loads(
        alignment_receipt_path.read_text()
    )

    cfg = protocol[
        "comparators"
    ]["scvi"]

    if (
        str(scvi.__version__)
        != str(cfg["version"])
    ):
        raise RuntimeError(
            "scvi-tools version mismatch: "
            f"{scvi.__version__}"
        )

    if args.seed != int(
        cfg[
            "hyperparameter_tuning_seed"
        ]
    ):
        raise RuntimeError(
            "Validation tuning must use only "
            "the frozen tuning seed"
        )

    candidates = tuple(
        int(v)
        for v in cfg[
            "n_latent_candidates"
        ]
    )

    if args.n_latent not in candidates:
        raise RuntimeError(
            f"Invalid n_latent="
            f"{args.n_latent}"
        )

    fixed = cfg[
        "model_fixed_parameters"
    ]

    max_epochs = int(
        cfg["training"][
            "max_epochs"
        ]
    )

    selective_policy = protocol[
        "selective_repair_policy"
    ]

    label_policy = protocol[
        "labels_and_eligibility"
    ]

    max_true_zero_fill = float(
        selective_policy[
            "max_true_zero_fill"
        ]
    )

    zero_threshold = float(
        label_policy[
            "zero_threshold"
        ]
    )

    mask_percent = int(
        args.mask_percent
    )

    panel_receipt = alignment[
        "audit_summary"
    ]["panels"][
        str(mask_percent)
    ]

    observed_benchmark_sha = (
        sha256_file(
            benchmark_path
        )
    )

    if (
        observed_benchmark_sha
        != panel_receipt[
            "panel_sha256"
        ]
    ):
        raise RuntimeError(
            "Benchmark SHA mismatch"
        )

    #
    # Pre-fit access is deliberately restricted:
    # read mask + provenance, not x/y.
    #
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

        row = np.asarray(
            panel["row"],
            dtype=np.int64,
        ).copy()

        shard_id = np.asarray(
            panel["shard_id"]
        ).astype(str)

        split = str(
            panel["split"].item()
        )

        modality = str(
            panel[
                "modality"
            ].item()
        )

        benchmark_seed = int(
            panel["seed"].item()
        )

        mask_rate = float(
            panel[
                "mask_rate"
            ].item()
        )

        manifest_sha = str(
            panel[
                "manifest_sha256"
            ].item()
        )

    if (
        synthetic_mask.shape
        != EXPECTED_SHAPE
    ):
        raise RuntimeError(
            "Synthetic-mask shape mismatch"
        )

    if row.shape != (
        EXPECTED_SHAPE[0],
    ):
        raise RuntimeError(
            "Benchmark row shape mismatch"
        )

    if not np.array_equal(
        row,
        np.arange(
            EXPECTED_SHAPE[0],
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Validation rows are not the "
            "frozen 0..2499 sequence"
        )

    unique_shards = np.unique(
        shard_id
    )

    if (
        unique_shards.size != 1
        or unique_shards[0]
        != EXPECTED_SHARD_ID
    ):
        raise RuntimeError(
            "Unexpected validation shard"
        )

    if split != "validation":
        raise RuntimeError(
            "Benchmark split is not validation"
        )

    if modality != "sc":
        raise RuntimeError(
            "Benchmark modality is not sc"
        )

    if benchmark_seed != int(
        alignment[
            "audit_summary"
        ][
            "benchmark_seed"
        ]
    ):
        raise RuntimeError(
            "Benchmark seed mismatch"
        )

    expected_rate = (
        mask_percent
        / 100.0
    )

    if not np.isclose(
        mask_rate,
        expected_rate,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError(
            "Mask-rate mismatch"
        )

    if manifest_sha != alignment[
        "audit_summary"
    ][
        "manifest_sha256"
    ]:
        raise RuntimeError(
            "Manifest SHA mismatch"
        )

    clean_counts = (
        first_validation_counts(
            corpus
        )
    )

    clean_hash = (
        sha256_dense_uint32(
            clean_counts
        )
    )

    if (
        clean_hash
        != panel_receipt[
            "clean_counts_uint32_sha256"
        ]
    ):
        raise RuntimeError(
            "Clean-count SHA mismatch"
        )

    if not np.all(
        clean_counts[
            synthetic_mask
        ] > 0
    ):
        raise RuntimeError(
            "Mask includes nonpositive "
            "raw-count entries"
        )

    corrupted_counts = (
        clean_counts.copy()
    )

    corrupted_counts[
        synthetic_mask
    ] = 0

    if np.any(
        corrupted_counts[
            synthetic_mask
        ] != 0
    ):
        raise RuntimeError(
            "Failed to zero masked counts"
        )

    if not np.array_equal(
        corrupted_counts[
            ~synthetic_mask
        ],
        clean_counts[
            ~synthetic_mask
        ],
    ):
        raise RuntimeError(
            "Unmasked raw counts changed"
        )

    corrupted_hash = (
        sha256_dense_uint32(
            corrupted_counts
        )
    )

    if (
        corrupted_hash
        != panel_receipt[
            "corrupted_counts_uint32_sha256"
        ]
    ):
        raise RuntimeError(
            "Corrupted-count SHA mismatch"
        )

    library_size = (
        corrupted_counts.sum(
            axis=1,
            dtype=np.uint64,
        )
    )

    if np.any(
        library_size <= 0
    ):
        raise RuntimeError(
            "scVI input contains a "
            "zero-library cell"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable in scVI "
            "validation job"
        )

    #
    # Exact frozen training seed.
    #
    scvi.settings.seed = int(
        args.seed
    )

    adata = ad.AnnData(
        X=sparse.csr_matrix(
            corrupted_counts
        )
    )

    adata.obs_names = [
        f"validation_{i:05d}"
        for i in range(
            EXPECTED_SHAPE[0]
        )
    ]

    adata.var_names = [
        f"gene_{i:04d}"
        for i in range(
            EXPECTED_SHAPE[1]
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
            fixed["n_hidden"]
        ),
        n_latent=int(
            args.n_latent
        ),
        n_layers=int(
            fixed["n_layers"]
        ),
        dispersion=str(
            fixed["dispersion"]
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

    #
    # Frozen protocol says max_epochs=400;
    # all other train() options remain at
    # pinned scvi-tools defaults.
    #
    model.train(
        max_epochs=max_epochs
    )

    decoded = (
        model.get_normalized_expression(
            library_size=10000,
            return_numpy=True,
        )
    )

    decoded = np.asarray(
        decoded,
        dtype=np.float32,
        order="C",
    )

    if decoded.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            "Decoded expression shape "
            "mismatch"
        )

    if not np.all(
        np.isfinite(
            decoded
        )
    ):
        raise RuntimeError(
            "Non-finite decoded expression"
        )

    if np.any(
        decoded < 0.0
    ):
        raise RuntimeError(
            "Negative decoded expression"
        )

    raw_prediction = np.asarray(
        np.log1p(
            decoded
        ),
        dtype=np.float32,
        order="C",
    )

    #
    # y is first accessed only after
    # the x-only/raw-count scVI fit and
    # decode have completed.
    #
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
        x.shape != EXPECTED_SHAPE
        or y.shape != EXPECTED_SHAPE
    ):
        raise RuntimeError(
            "Benchmark x/y shape mismatch"
        )

    positive = synthetic_mask

    true_zero = (
        (~positive)
        & (
            np.abs(y)
            <= zero_threshold
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
            "Positive/true-zero overlap"
        )

    if not np.all(
        np.abs(
            x[eligible_zero]
        )
        <= zero_threshold
    ):
        raise RuntimeError(
            "Eligible repair set contains "
            "observed nonzero input"
        )

    raw_metrics = (
        masked_value_metrics(
            raw_prediction,
            y,
            positive,
        )
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
            raw_metrics[
                "masked_mse"
            ]
        )
        / target_variance
    )

    #
    # Frozen scVI repair score is the
    # same log1p decoded expression.
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
            max_true_zero_fill=(
                max_true_zero_fill
            ),
        )
    )

    threshold = float(
        selected[
            "threshold"
        ]
    )

    selected_repair = (
        eligible_zero
        & (
            score
            >= threshold
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
            "No observed nonzero entries"
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
            "Observed-nonzero preservation "
            "failed"
        )

    selective_metrics = (
        masked_value_metrics(
            reconstruction,
            y,
            positive,
        )
    )

    code_commit = git_head()

    summary: dict[
        str,
        Any,
    ] = {
        "method":
            "scvi",

        "classification":
            "confirmatory_validation_only",

        "family":
            cfg["family"],

        "transductive":
            True,

        "fit_input":
            (
                "aligned genuine raw counts; "
                "frozen synthetic_mask "
                "positions zeroed"
            ),

        "target_used_for_fit":
            False,

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

        "random_seed":
            int(
                args.seed
            ),

        "n_latent":
            int(
                args.n_latent
            ),

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

        "training_options":
            (
                "Only max_epochs explicitly "
                "set; remaining train() "
                "options use pinned defaults"
            ),

        "value_prediction":
            (
                "log1p(scVI decoded "
                "normalized expression, "
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

        "mask_percent":
            mask_percent,

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

        "max_true_zero_fill":
            max_true_zero_fill,

        "target_variance":
            target_variance,

        "recovery_r":
            recovery_r,

        "input_counts_uint32_sha256":
            corrupted_hash,

        "clean_counts_uint32_sha256":
            clean_hash,

        "prediction_float32_sha256":
            sha256_array_float32(
                raw_prediction
            ),

        "decoded_min":
            float(
                decoded.min()
            ),

        "decoded_max":
            float(
                decoded.max()
            ),

        "decoded_nonzero_fraction":
            float(
                np.mean(
                    decoded > 0.0
                )
            ),

        "benchmark":
            str(
                benchmark_path
            ),

        "benchmark_sha256":
            observed_benchmark_sha,

        "protocol":
            str(
                protocol_path
            ),

        "protocol_sha256":
            protocol_sha,

        "alignment_receipt":
            str(
                alignment_receipt_path
            ),

        "alignment_receipt_sha256":
            alignment_sha,

        "implementation_git_commit":
            code_commit,

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

    summary_json = (
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    metadata = {
        "method":
            "scvi",

        "classification":
            "confirmatory_validation_only",

        "model_selection_role":
            (
                "candidate fit for frozen "
                "common-n_latent validation "
                "selection"
            ),

        "target_access_during_fit":
            False,

        "benchmark_target_first_access":
            (
                "after model.train() and "
                "get_normalized_expression()"
            ),

        "corrupted_count_sha256":
            corrupted_hash,

        "alignment_receipt_sha256":
            alignment_sha,

        "protocol_sha256":
            protocol_sha,

        "implementation_git_commit":
            code_commit,
    }

    metadata_json = (
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    #
    # Materialize only after all fit and
    # validation calculations succeeded.
    #
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

    (
        output_dir
        / "scvi_metadata.json"
    ).write_text(
        metadata_json,
        encoding="utf-8",
    )

    files = (
        "summary.json",
        "summary.csv",
        "exact_threshold_frontier.csv",
        "threshold_curve.csv",
        "risk_coverage.csv",
        "scvi_metadata.json",
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
        "P2_SCVI_VALIDATION=PASS"
    )


if __name__ == "__main__":
    main()
