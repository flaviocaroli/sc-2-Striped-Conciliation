#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from sklearn.utils.extmath import randomized_svd

from sc2.eval.p2_knn import (
    neighbor_indices_excluding_self,
)

from sc2.eval.p2_selective import (
    score_discrimination,
)

from sc2.eval.selective_repair_metrics import (
    masked_value_metrics,
    risk_coverage_curve,
)


EXPECTED_SHAPE = (
    5000,
    4096,
)

ZERO_THRESHOLD = 1.0e-8

EXPECTED_PROTOCOL_SHA = (
    "3f079aa5e756821dc5c75d88e95b0b377cfdbdf29afe0484e1360cd9e5f7f2b9"
)

EXPECTED_PRETEST_SHA = (
    "514b76785abf7fbe16e65a4d4753827e32f9c0469ffd07b3c6bbb94b4bf48452"
)

SELECTION_RECEIPT_SHA = {
    "positive_train_mean":
        "4e60ec90d4fdfebd2f0a4af1d68fa97380dd92f6a2ac9c73a2a83ef9a24a8e20",

    "positive_train_median":
        "4e60ec90d4fdfebd2f0a4af1d68fa97380dd92f6a2ac9c73a2a83ef9a24a8e20",

    "truncated_low_rank":
        "ee9c25820cbe390ac3ed448896cf07ed811b4fd167411de6d95b527ac95cd397",

    "knn":
        "4a37cab3864edd2153f0a6ee23f56cffd11cd91437add15153535d48a43658bf",

    "alra":
        "44797fe64604603ab3bd7d0df06ad14ce490d4ef6b43275c56a3c17be502c896",
}

TRAIN_STATS_SHA = (
    "75d713e31ff63269b9885f5076d1d3223bc357a620cea0bc4b1e829181d638c0"
)

ALRA_SOURCE_COMMIT = (
    "f34d46570b1221179047d9f99c235ed880cc3bae"
)

EXPECTED_AVAILABLE = {
    "internal_test": 4096,
    "zheng68k": 4046,
    "baron_pancreas": 3851,
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


def json_safe(
    value: Any,
) -> Any:

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k):
                json_safe(v)
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        value = float(value)

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

    return value


def prefixed(
    d: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}{k}":
            v
        for k, v
        in d.items()
    }


def read_metadata(
    path: Path,
) -> dict[str, str]:

    out: dict[str, str] = {}

    for raw in (
        path.read_text()
        .splitlines()
    ):
        if not raw.strip():
            continue

        parts = raw.split(
            "\t",
            1,
        )

        if len(parts) != 2:
            raise RuntimeError(
                "Invalid ALRA metadata line: "
                f"{raw!r}"
            )

        key, value = parts

        out[
            key.strip()
        ] = value.strip()

    return out


def load_frozen_selection(
    *,
    method: str,
    mask_percent: int,
    receipt_path: Path,
) -> tuple[
    float,
    dict[str, Any],
]:

    expected_sha = (
        SELECTION_RECEIPT_SHA[
            method
        ]
    )

    if (
        sha256_file(
            receipt_path
        )
        != expected_sha
    ):
        raise RuntimeError(
            "Frozen selection receipt "
            "SHA mismatch"
        )

    d = json.loads(
        receipt_path.read_text()
    )

    key = str(
        mask_percent
    )

    if method in (
        "positive_train_mean",
        "positive_train_median",
    ):
        threshold = float(
            d[
                "results"
            ][
                method
            ][
                key
            ][
                "threshold"
            ]
        )

        expected = (
            0.5788050293922424
        )

        if threshold != expected:
            raise RuntimeError(
                "Baseline threshold mismatch"
            )

        details = {
            "selected_hyperparameter":
                None,
        }

    elif method == (
        "truncated_low_rank"
    ):

        rank = int(
            d[
                "common_rank_selection"
            ][
                "selected_rank"
            ]
        )

        if rank != 16:
            raise RuntimeError(
                "Frozen rank mismatch"
            )

        threshold = float(
            d[
                "successful_results"
            ][
                "16"
            ][
                key
            ][
                "threshold"
            ]
        )

        details = {
            "selected_rank":
                rank,
        }

    elif method == "knn":

        k = int(
            d[
                "common_k_selection"
            ][
                "selected_k"
            ]
        )

        if k != 50:
            raise RuntimeError(
                "Frozen k mismatch"
            )

        threshold = float(
            d[
                "successful_results"
            ][
                "50"
            ][
                key
            ][
                "threshold"
            ]
        )

        details = {
            "selected_k":
                k,
        }

    elif method == "alra":

        threshold = float(
            d[
                "successful_results"
            ][
                key
            ][
                "threshold"
            ]
        )

        details = {
            "rank_policy":
                (
                    "automatic choose_k "
                    "separately on this "
                    "corrupted cohort"
                ),
        }

    else:
        raise ValueError(
            method
        )

    return (
        threshold,
        details,
    )


def train_stat_predict(
    x_shape: tuple[int, int],
    *,
    method: str,
    stats_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:

    if (
        sha256_file(
            stats_path
        )
        != TRAIN_STATS_SHA
    ):
        raise RuntimeError(
            "Train statistics SHA mismatch"
        )

    with np.load(
        stats_path,
        allow_pickle=False,
    ) as d:

        prevalence = np.asarray(
            d[
                "positive_prevalence"
            ],
            dtype=np.float32,
        )

        if method == (
            "positive_train_mean"
        ):
            values = np.asarray(
                d[
                    "positive_mean"
                ],
                dtype=np.float32,
            )

        elif method == (
            "positive_train_median"
        ):
            values = np.asarray(
                d[
                    "positive_median"
                ],
                dtype=np.float32,
            )

        else:
            raise ValueError(
                method
            )

        n_train = int(
            d[
                "n_train_cells"
            ]
        )

    if (
        prevalence.shape
        != (
            x_shape[1],
        )
        or values.shape
        != (
            x_shape[1],
        )
    ):
        raise RuntimeError(
            "Train-stat vector shape mismatch"
        )

    if n_train != 200000:
        raise RuntimeError(
            "Train-stat cell-count mismatch"
        )

    prediction = np.tile(
        values[
            None,
            :
        ],
        (
            x_shape[0],
            1,
        ),
    )

    score = np.tile(
        prevalence[
            None,
            :
        ],
        (
            x_shape[0],
            1,
        ),
    )

    return (
        prediction,
        score,
        {
            "n_train_cells":
                n_train,

            "train_stats_sha256":
                TRAIN_STATS_SHA,
        },
    )


def low_rank_predict(
    x: np.ndarray,
    *,
    rank: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:

    if rank != 16:
        raise RuntimeError(
            "Only frozen rank=16 "
            "is permitted"
        )

    u, s, vt = randomized_svd(
        x,
        n_components=rank,
        random_state=20260728,
    )

    prediction = (
        (
            u
            * s[
                None,
                :
            ]
        )
        @ vt
    ).astype(
        np.float32,
        copy=False,
    )

    prediction = np.maximum(
        prediction,
        np.float32(
            0.0
        ),
    )

    return (
        prediction,
        prediction,
        {
            "selected_rank":
                rank,

            "random_state":
                20260728,
        },
    )


def knn_predict(
    x: np.ndarray,
    *,
    k: int,
    chunk_size: int = 64,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:

    if k != 50:
        raise RuntimeError(
            "Only frozen k=50 "
            "is permitted"
        )

    indices, distances = (
        neighbor_indices_excluding_self(
            x,
            k=k,
        )
    )

    if indices.shape != (
        x.shape[0],
        k,
    ):
        raise RuntimeError(
            "kNN index shape mismatch"
        )

    rows = np.arange(
        x.shape[0]
    )[:, None]

    if np.any(
        indices
        == rows
    ):
        raise RuntimeError(
            "kNN self-neighbor detected"
        )

    prediction = np.empty_like(
        x,
        dtype=np.float32,
    )

    score = np.empty_like(
        x,
        dtype=np.float32,
    )

    for start in range(
        0,
        x.shape[0],
        chunk_size,
    ):
        stop = min(
            x.shape[0],
            start + chunk_size,
        )

        values = x[
            indices[
                start:stop
            ]
        ]

        prediction[
            start:stop
        ] = np.asarray(
            values.mean(
                axis=1
            ),
            dtype=np.float32,
        )

        score[
            start:stop
        ] = np.asarray(
            (
                values
                > ZERO_THRESHOLD
            ).mean(
                axis=1
            ),
            dtype=np.float32,
        )

    if (
        np.any(
            score < 0
        )
        or np.any(
            score > 1
        )
    ):
        raise RuntimeError(
            "kNN score outside [0,1]"
        )

    return (
        prediction,
        score,
        {
            "selected_k":
                k,

            "distance_metric":
                "cosine",

            "algorithm":
                "brute",

            "self_excluded":
                True,

            "neighbor_distance_min":
                float(
                    np.min(
                        distances
                    )
                ),

            "neighbor_distance_max":
                float(
                    np.max(
                        distances
                    )
                ),
        },
    )


def alra_predict(
    x: np.ndarray,
    *,
    alra_r_script: Path,
    alra_rlib: Path,
    alra_source: Path,
    rscript: Path,
    work_dir: Path,
) -> tuple[
    np.ndarray,
    dict[str, str],
    subprocess.CompletedProcess[str],
    dict[str, Path],
]:

    source_commit = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(
                    alra_source
                ),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )

    if (
        source_commit
        != ALRA_SOURCE_COMMIT
    ):
        raise RuntimeError(
            "ALRA source commit mismatch"
        )

    source_status = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(
                    alra_source
                ),
                "status",
                "--porcelain",
            ],
            text=True,
        )
    )

    if source_status.strip():
        raise RuntimeError(
            "ALRA source is dirty"
        )

    input_bin = (
        work_dir
        / "input.bin"
    )

    output_bin = (
        work_dir
        / "output.bin"
    )

    metadata_tsv = (
        work_dir
        / "alra_metadata.tsv"
    )

    singular_values_csv = (
        work_dir
        / "choose_k_singular_values.csv"
    )

    num_of_sds_csv = (
        work_dir
        / "choose_k_num_of_sds.csv"
    )

    np.asarray(
        x,
        dtype="<f4",
        order="C",
    ).tofile(
        input_bin
    )

    env = os.environ.copy()

    env[
        "R_LIBS_USER"
    ] = str(
        alra_rlib
    )

    proc = subprocess.run(
        [
            str(
                rscript
            ),
            "--vanilla",
            str(
                alra_r_script
            ),
            str(
                input_bin
            ),
            str(
                x.shape[0]
            ),
            str(
                x.shape[1]
            ),
            str(
                output_bin
            ),
            str(
                metadata_tsv
            ),
            str(
                singular_values_csv
            ),
            str(
                num_of_sds_csv
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "ALRA R wrapper failed:\n"
            + proc.stdout
            + "\n"
            + proc.stderr
        )

    for path in (
        output_bin,
        metadata_tsv,
        singular_values_csv,
        num_of_sds_csv,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Missing ALRA artifact: "
                f"{path}"
            )

    raw = np.fromfile(
        output_bin,
        dtype="<f4",
    )

    if raw.size != x.size:
        raise RuntimeError(
            "ALRA output size mismatch: "
            f"{raw.size} != {x.size}"
        )

    prediction = raw.reshape(
        x.shape,
        order="C",
    )

    if not np.all(
        np.isfinite(
            prediction
        )
    ):
        raise RuntimeError(
            "ALRA output is non-finite"
        )

    if np.any(
        prediction < 0
    ):
        raise RuntimeError(
            "ALRA output contains "
            "negative values"
        )

    metadata = read_metadata(
        metadata_tsv
    )

    chosen_k = int(
        float(
            metadata[
                "chosen_k"
            ]
        )
    )

    if chosen_k <= 0:
        raise RuntimeError(
            "Invalid ALRA chosen_k"
        )

    artifacts = {
        "metadata":
            metadata_tsv,

        "singular_values":
            singular_values_csv,

        "num_of_sds":
            num_of_sds_csv,
    }

    return (
        np.asarray(
            prediction,
            dtype=np.float32,
            order="C",
        ),
        metadata,
        proc,
        artifacts,
    )


def frozen_operating_point(
    score: np.ndarray,
    positive: np.ndarray,
    true_zero: np.ndarray,
    threshold: float,
) -> dict[str, Any]:

    if np.any(
        positive
        & true_zero
    ):
        raise RuntimeError(
            "positive/true_zero overlap"
        )

    eligible = (
        positive
        | true_zero
    )

    selected = (
        eligible
        & (
            score
            >= threshold
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

    n_zero = int(
        true_zero.sum()
    )

    n_selected = int(
        selected.sum()
    )

    if (
        n_positive <= 0
        or n_zero <= 0
    ):
        raise RuntimeError(
            "Invalid operating-point "
            "denominators"
        )

    recall = (
        tp
        / n_positive
    )

    precision = (
        tp
        / n_selected
        if n_selected
        else 0.0
    )

    fill = (
        fp
        / n_zero
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
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

        "threshold_f1":
            float(
                f1
            ),
    }


def presentation_threshold_curve(
    score: np.ndarray,
    positive: np.ndarray,
    true_zero: np.ndarray,
    frozen_threshold: float,
    *,
    max_score_rows: int = 201,
) -> pd.DataFrame:
    """
    Presentation-only curve.

    No threshold is selected here. We sort the
    complete eligible score vector, sample at most
    201 exact observed score boundaries across its
    full range, and append the already-frozen
    operating point exactly.
    """

    eligible = (
        positive
        | true_zero
    )

    scores = np.asarray(
        score[
            eligible
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        positive[
            eligible
        ],
        dtype=bool,
    )

    if scores.size == 0:
        raise RuntimeError(
            "No eligible threshold scores"
        )

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_score = scores[
        order
    ]

    sorted_positive = labels[
        order
    ]

    cumulative_tp = np.cumsum(
        sorted_positive,
        dtype=np.int64,
    )

    negative_sorted_score = (
        -sorted_score
    )

    positions = np.unique(
        np.rint(
            np.linspace(
                0,
                sorted_score.size - 1,
                min(
                    max_score_rows,
                    sorted_score.size,
                ),
            )
        ).astype(
            np.int64
        )
    )

    threshold_values = np.unique(
        sorted_score[
            positions
        ]
    )

    n_positive = int(
        positive.sum()
    )

    n_zero = int(
        true_zero.sum()
    )

    rows: list[
        dict[str, Any]
    ] = []

    sentinel = np.nextafter(
        float(
            sorted_score[
                0
            ]
        ),
        np.inf,
    )

    rows.append({
        "threshold":
            sentinel,

        "tp":
            0,

        "fp":
            0,

        "selected":
            0,

        "recall":
            0.0,

        "precision":
            0.0,

        "true_zero_fill":
            0.0,

        "operating_point":
            False,
    })

    for threshold in (
        threshold_values
    ):
        end = int(
            np.searchsorted(
                negative_sorted_score,
                -float(
                    threshold
                ),
                side="right",
            )
            - 1
        )

        tp = int(
            cumulative_tp[
                end
            ]
        )

        selected = (
            end
            + 1
        )

        fp = (
            selected
            - tp
        )

        rows.append({
            "threshold":
                float(
                    threshold
                ),

            "tp":
                tp,

            "fp":
                fp,

            "selected":
                selected,

            "recall":
                tp
                / n_positive,

            "precision":
                tp
                / selected,

            "true_zero_fill":
                fp
                / n_zero,

            "operating_point":
                False,
        })

    op = frozen_operating_point(
        score,
        positive,
        true_zero,
        frozen_threshold,
    )

    rows.append({
        "threshold":
            op[
                "threshold"
            ],

        "tp":
            op[
                "threshold_tp"
            ],

        "fp":
            op[
                "threshold_fp"
            ],

        "selected":
            op[
                "threshold_selected"
            ],

        "recall":
            op[
                "threshold_recall"
            ],

        "precision":
            op[
                "threshold_precision"
            ],

        "true_zero_fill":
            op[
                "threshold_true_zero_fill"
            ],

        "operating_point":
            True,
    })

    out = pd.DataFrame(
        rows
    )

    out = (
        out.sort_values(
            [
                "threshold",
                "operating_point",
            ],
            ascending=[
                False,
                False,
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "threshold"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return out


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--method",
        required=True,
        choices=(
            "positive_train_mean",
            "positive_train_median",
            "truncated_low_rank",
            "knn",
            "alra",
        ),
    )

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
        "--benchmark",
        required=True,
    )

    p.add_argument(
        "--benchmark-sha256",
        required=True,
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
        "--selection-receipt",
        required=True,
    )

    p.add_argument(
        "--train-stats",
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
        "--alra-source",
        required=True,
    )

    p.add_argument(
        "--rscript",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    return p.parse_args()


def main():
    args = parse_args()

    method = args.method
    dataset = args.dataset
    mask_percent = int(
        args.mask_percent
    )

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
        args.selection_receipt
    ).resolve()

    stats_path = Path(
        args.train_stats
    ).resolve()

    alra_r_script = Path(
        args.alra_r_script
    ).resolve()

    alra_rlib = Path(
        args.alra_rlib
    ).resolve()

    alra_source = Path(
        args.alra_source
    ).resolve()

    rscript = Path(
        args.rscript
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if output_dir.exists():
        raise RuntimeError(
            f"Output exists: "
            f"{output_dir}"
        )

    if (
        sha256_file(
            protocol_path
        )
        != EXPECTED_PROTOCOL_SHA
    ):
        raise RuntimeError(
            "Protocol SHA mismatch"
        )

    if (
        sha256_file(
            pretest_path
        )
        != EXPECTED_PRETEST_SHA
    ):
        raise RuntimeError(
            "Pre-test receipt SHA mismatch"
        )

    observed_benchmark_sha = (
        sha256_file(
            benchmark_path
        )
    )

    if (
        observed_benchmark_sha
        != args.benchmark_sha256
    ):
        raise RuntimeError(
            "Benchmark SHA mismatch"
        )

    protocol = json.loads(
        protocol_path.read_text()
    )

    if (
        method
        not in protocol[
            "comparators"
        ]
    ):
        raise RuntimeError(
            "Method absent from "
            "frozen protocol"
        )

    threshold, selection_details = (
        load_frozen_selection(
            method=method,
            mask_percent=mask_percent,
            receipt_path=selection_path,
        )
    )

    #
    # Before prediction / fitting, load x,
    # corruption locations and availability.
    # Do NOT load target y.
    #
    with np.load(
        benchmark_path,
        allow_pickle=False,
    ) as panel:

        x = np.asarray(
            panel[
                "x"
            ],
            dtype=np.float32,
        ).copy()

        positive = np.asarray(
            panel[
                "synthetic_mask"
            ],
            dtype=bool,
        ).copy()

        if (
            "available_gene_mask"
            in panel.files
        ):
            available_gene_mask = (
                np.asarray(
                    panel[
                        "available_gene_mask"
                    ],
                    dtype=bool,
                ).copy()
            )
        else:
            available_gene_mask = (
                np.ones(
                    x.shape[1],
                    dtype=bool,
                )
            )

    if (
        x.shape
        != EXPECTED_SHAPE
        or positive.shape
        != EXPECTED_SHAPE
    ):
        raise RuntimeError(
            "Final benchmark shape mismatch"
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
        int(
            available_gene_mask.sum()
        )
        != EXPECTED_AVAILABLE[
            dataset
        ]
    ):
        raise RuntimeError(
            "Available-gene count mismatch"
        )

    available = (
        available_gene_mask[
            None,
            :
        ]
    )

    if np.any(
        positive
        & (
            ~available
        )
    ):
        raise RuntimeError(
            "Masked unavailable gene"
        )

    #
    # Produce prediction and repair score
    # WITHOUT target y.
    #
    alra_context = None
    alra_metadata = None
    alra_proc = None
    alra_artifacts = None

    if method in (
        "positive_train_mean",
        "positive_train_median",
    ):

        (
            prediction,
            score,
            method_details,
        ) = train_stat_predict(
            x.shape,
            method=method,
            stats_path=stats_path,
        )

    elif method == (
        "truncated_low_rank"
    ):

        rank = int(
            selection_details[
                "selected_rank"
            ]
        )

        (
            prediction,
            score,
            method_details,
        ) = low_rank_predict(
            x,
            rank=rank,
        )

    elif method == "knn":

        k = int(
            selection_details[
                "selected_k"
            ]
        )

        (
            prediction,
            score,
            method_details,
        ) = knn_predict(
            x,
            k=k,
            chunk_size=64,
        )

    elif method == "alra":

        alra_context = (
            tempfile.TemporaryDirectory(
                prefix=(
                    "p2_alra_final_"
                )
            )
        )

        alra_work = Path(
            alra_context.name
        )

        (
            prediction,
            alra_metadata,
            alra_proc,
            alra_artifacts,
        ) = alra_predict(
            x,
            alra_r_script=alra_r_script,
            alra_rlib=alra_rlib,
            alra_source=alra_source,
            rscript=rscript,
            work_dir=alra_work,
        )

        score = prediction

        method_details = {
            "chosen_k":
                int(
                    float(
                        alra_metadata[
                            "chosen_k"
                        ]
                    )
                ),

            "random_seed":
                20260728,

            "source_commit":
                ALRA_SOURCE_COMMIT,

            "rank_policy":
                (
                    "automatic choose_k "
                    "on corrupted x"
                ),
        }

    else:
        raise ValueError(
            method
        )

    if (
        prediction.shape
        != EXPECTED_SHAPE
        or score.shape
        != EXPECTED_SHAPE
    ):
        raise RuntimeError(
            "Prediction/score shape mismatch"
        )

    if (
        not np.all(
            np.isfinite(
                prediction
            )
        )
        or not np.all(
            np.isfinite(
                score
            )
        )
    ):
        raise RuntimeError(
            "Non-finite prediction/score"
        )

    #
    # Target y is first loaded HERE,
    # after all fitting/prediction.
    #
    with np.load(
        benchmark_path,
        allow_pickle=False,
    ) as panel:

        y = np.asarray(
            panel[
                "y"
            ],
            dtype=np.float32,
        ).copy()

    if y.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            "Target shape mismatch"
        )

    if not np.all(
        y[
            positive
        ]
        > ZERO_THRESHOLD
    ):
        raise RuntimeError(
            "Masked target is not positive"
        )

    if np.any(
        np.abs(
            x[
                positive
            ]
        )
        > ZERO_THRESHOLD
    ):
        raise RuntimeError(
            "Masked x is not zero"
        )

    true_zero = (
        (
            y
            <= ZERO_THRESHOLD
        )
        & available
        & (
            ~positive
        )
    )

    eligible_zero = (
        positive
        | true_zero
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
            "Repair eligibility "
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
            "Masked target variance "
            "is nonpositive"
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
            score,
            positive,
            true_zero,
        )
    )

    operating = (
        frozen_operating_point(
            score,
            positive,
            true_zero,
            threshold,
        )
    )

    selected_repair = (
        eligible_zero
        & (
            score
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
        & available
    )

    preservation_error = (
        reconstruction[
            observed_nonzero
        ]
        - x[
            observed_nonzero
        ]
    )

    if preservation_error.size <= 0:
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

    threshold_curve = (
        presentation_threshold_curve(
            score,
            positive,
            true_zero,
            threshold,
        )
    )

    risk_curve = (
        risk_coverage_curve(
            prediction,
            y,
            positive,
            score,
        )
    )

    summary: dict[
        str,
        Any,
    ] = {
        "method":
            method,

        "dataset":
            dataset,

        "mask_percent":
            mask_percent,

        "classification":
            (
                "confirmatory_post_"
                "pretest_freeze"
            ),

        "target_used_for_fit":
            False,

        "target_loaded_after_prediction":
            True,

        "threshold_selection_performed":
            False,

        "threshold_source":
            (
                "frozen_internal_"
                "validation_receipt"
            ),

        "threshold_selection_receipt":
            str(
                selection_path
            ),

        "threshold_selection_receipt_sha256":
            sha256_file(
                selection_path
            ),

        "n_available_genes":
            int(
                available_gene_mask.sum()
            ),

        "benchmark":
            str(
                benchmark_path
            ),

        "benchmark_sha256":
            observed_benchmark_sha,

        "protocol_sha256":
            sha256_file(
                protocol_path
            ),

        "pretest_receipt_sha256":
            sha256_file(
                pretest_path
            ),

        "implementation_git_commit":
            git_head(),

        "evaluator_git_commit":
            git_file_commit(
                "scripts/analysis/"
                "evaluate_p2_remaining_final.py"
            ),

        "zero_threshold":
            ZERO_THRESHOLD,

        "target_variance":
            target_variance,

        "recovery_r":
            recovery_r,

        "prediction_float32_sha256":
            sha256_float32(
                prediction
            ),

        "score_float32_sha256":
            sha256_float32(
                score
            ),

        "threshold_curve_policy":
            (
                "presentation-only: "
                "at most 201 exact observed "
                "score boundaries spanning "
                "the full score range plus "
                "the exact frozen operating "
                "threshold; no selection"
            ),

        **selection_details,

        **method_details,

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

    safe_summary = (
        json_safe(
            summary
        )
    )

    tmp_output = (
        output_dir.parent
        / (
            "."
            + output_dir.name
            + ".tmp"
        )
    )

    if tmp_output.exists():
        raise RuntimeError(
            f"Temporary output exists: "
            f"{tmp_output}"
        )

    tmp_output.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        (
            tmp_output
            / "summary.json"
        ).write_text(
            json.dumps(
                safe_summary,
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
            tmp_output
            / "summary.csv",
            index=False,
        )

        threshold_curve.to_csv(
            tmp_output
            / "threshold_curve.csv",
            index=False,
        )

        risk_curve.to_csv(
            tmp_output
            / "risk_coverage.csv",
            index=False,
        )

        files = [
            "summary.json",
            "summary.csv",
            "threshold_curve.csv",
            "risk_coverage.csv",
        ]

        if method == "alra":

            assert (
                alra_metadata
                is not None
            )

            assert (
                alra_proc
                is not None
            )

            assert (
                alra_artifacts
                is not None
            )

            shutil.copy2(
                alra_artifacts[
                    "metadata"
                ],
                tmp_output
                / "alra_metadata.tsv",
            )

            shutil.copy2(
                alra_artifacts[
                    "singular_values"
                ],
                tmp_output
                / "choose_k_singular_values.csv",
            )

            shutil.copy2(
                alra_artifacts[
                    "num_of_sds"
                ],
                tmp_output
                / "choose_k_num_of_sds.csv",
            )

            (
                tmp_output
                / "alra_stdout.txt"
            ).write_text(
                alra_proc.stdout,
                encoding="utf-8",
            )

            (
                tmp_output
                / "alra_stderr.txt"
            ).write_text(
                alra_proc.stderr,
                encoding="utf-8",
            )

            files.extend([
                "alra_metadata.tsv",
                "choose_k_singular_values.csv",
                "choose_k_num_of_sds.csv",
                "alra_stdout.txt",
                "alra_stderr.txt",
            ])

        with (
            tmp_output
            / "SHA256SUMS.txt"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:

            for name in files:
                handle.write(
                    f"{sha256_file(tmp_output / name)}"
                    f"  {name}\n"
                )

        tmp_output.rename(
            output_dir
        )

    except Exception:
        shutil.rmtree(
            tmp_output,
            ignore_errors=True,
        )
        raise

    finally:
        if (
            alra_context
            is not None
        ):
            alra_context.cleanup()

    print(
        json.dumps(
            safe_summary,
            sort_keys=True,
            allow_nan=False,
        )
    )

    print(
        "P2_REMAINING_FINAL=PASS"
    )


if __name__ == "__main__":
    main()
