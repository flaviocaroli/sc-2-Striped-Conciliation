#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from scipy import sparse
from scipy.stats import rankdata


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def finite_mean(values) -> float:
    array = np.asarray(
        list(values),
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    return (
        float(array.mean())
        if array.size
        else float("nan")
    )


def spearman(x, y) -> float:
    x = np.asarray(x)
    y = np.asarray(y)

    if (
        x.size < 2
        or np.std(x) == 0.0
        or np.std(y) == 0.0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            rankdata(x),
            rankdata(y),
        )[0, 1]
    )


def masked_metrics(
    prediction,
    target,
    mask,
):

    if not bool(mask.any()):
        raise RuntimeError(
            "empty lost-positive mask"
        )

    error = (
        prediction[mask]
        - target[mask]
    )

    masked_prediction = prediction[
        mask
    ]

    masked_target = target[
        mask
    ]

    prediction_sd = float(
        np.std(
            masked_prediction
        )
    )

    target_sd = float(
        np.std(
            masked_target
        )
    )

    if (
        not np.isfinite(
            prediction_sd
        )
        or not np.isfinite(
            target_sd
        )
        or target_sd <= 0.0
    ):
        raise RuntimeError(
            "invalid masked SD"
        )

    masked_mse = float(
        np.mean(
            error ** 2
        )
    )

    masked_mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    if (
        not np.isfinite(
            masked_mse
        )
        or not np.isfinite(
            masked_mae
        )
    ):
        raise RuntimeError(
            "nonfinite masked error metric"
        )

    sample_values = np.asarray(
        [
            spearman(
                prediction[
                    row,
                    mask[row],
                ],
                target[
                    row,
                    mask[row],
                ],
            )
            for row
            in range(
                prediction.shape[0]
            )
        ],
        dtype=np.float64,
    )

    gene_values = np.asarray(
        [
            spearman(
                prediction[
                    mask[:, column],
                    column,
                ],
                target[
                    mask[:, column],
                    column,
                ],
            )
            for column
            in range(
                prediction.shape[1]
            )
        ],
        dtype=np.float64,
    )

    sample_valid = np.isfinite(
        sample_values
    )

    gene_valid = np.isfinite(
        gene_values
    )

    sample_spearman = (
        float(
            sample_values[
                sample_valid
            ].mean()
        )
        if sample_valid.any()
        else None
    )

    gene_spearman = (
        float(
            gene_values[
                gene_valid
            ].mean()
        )
        if gene_valid.any()
        else None
    )

    sd_ratio = float(
        prediction_sd
        / target_sd
    )

    recovery_index = float(
        1.0
        - masked_mse
        / (
            target_sd ** 2
        )
    )

    if (
        not np.isfinite(
            sd_ratio
        )
        or not np.isfinite(
            recovery_index
        )
    ):
        raise RuntimeError(
            "nonfinite SD/recovery metric"
        )

    return {
        "masked_mse":
            masked_mse,

        "masked_mae":
            masked_mae,

        "sample_spearman":
            sample_spearman,

        "n_sample_spearman_valid":
            int(
                sample_valid.sum()
            ),

        "gene_spearman":
            gene_spearman,

        "n_gene_spearman_valid":
            int(
                gene_valid.sum()
            ),

        "prediction_sd":
            prediction_sd,

        "target_sd":
            target_sd,

        "sd_ratio":
            sd_ratio,

        "sd_ratio_abs_error":
            abs(
                sd_ratio
                - 1.0
            ),

        "recovery_index":
            recovery_index,

        "n_masked":
            int(
                mask.sum()
            ),
    }


def cp10k_log1p(values):
    x = np.asarray(
        values,
        dtype=np.float32,
    ).copy()

    if not np.isfinite(x).all():
        raise RuntimeError(
            "nonfinite count-like matrix"
        )

    if float(
        x.min()
    ) < -1.0e-5:
        raise RuntimeError(
            "negative native prediction: "
            f"min={float(x.min())}"
        )

    x[
        x < 0.0
    ] = 0.0

    total = (
        x.astype(
            np.float64
        )
        .sum(
            axis=1
        )
    )

    if (
        (~np.isfinite(total)).any()
        or (total <= 0.0).any()
    ):
        raise RuntimeError(
            "zero/nonfinite evaluation library"
        )

    x *= (
        10000.0
        / total
    ).astype(
        np.float32
    )[:, None]

    np.log1p(
        x,
        out=x,
    )

    return x


def full_metrics(
    prediction,
    target,
    true_zero,
):

    error = (
        prediction
        - target
    )

    result = {
        "full_available_mse":
            float(
                np.mean(
                    error ** 2
                )
            ),

        "full_available_mae":
            float(
                np.mean(
                    np.abs(error)
                )
            ),
    }

    if bool(
        true_zero.any()
    ):

        zero_values = prediction[
            true_zero
        ]

        result.update(
            {
                "true_zero_mse":
                    float(
                        np.mean(
                            zero_values ** 2
                        )
                    ),

                "true_zero_mae":
                    float(
                        np.mean(
                            np.abs(
                                zero_values
                            )
                        )
                    ),

                "true_zero_positive_fraction":
                    float(
                        np.mean(
                            zero_values
                            > 0.0
                        )
                    ),
            }
        )

    return result


def load_p3_genes(
    path: Path,
):
    frame = pd.read_parquet(
        path
    )

    for column in (
        "ensembl_gene_id",
        "ensembl_id",
        "gene_id",
        "feature_id",
    ):

        if column not in frame.columns:
            continue

        genes = (
            frame[column]
            .astype(str)
            .to_numpy()
        )

        if (
            len(genes)
            == 16384
            and len(
                set(genes)
            )
            == 16384
            and np.mean(
                [
                    gene.startswith(
                        "ENSG"
                    )
                    for gene
                    in genes
                ]
            )
            > 0.95
        ):
            return genes

    genes = (
        frame.index
        .astype(str)
        .to_numpy()
    )

    if (
        len(genes) == 16384
        and len(
            set(genes)
        )
        == 16384
    ):
        return genes

    raise RuntimeError(
        "cannot resolve ordered P3 gene IDs"
    )


def make_adata(
    counts,
    p3_genes,
):

    if counts.shape != (
        5000,
        16384,
    ):
        raise RuntimeError(
            f"bad count shape: "
            f"{counts.shape}"
        )

    obs = pd.DataFrame(
        index=[
            f"cell_{i:05d}"
            for i in range(5000)
        ]
    )

    obs[
        "organism_ontology_term_id"
    ] = "NCBITaxon:9606"

    return ad.AnnData(
        X=sparse.csr_matrix(
            counts
        ),
        obs=obs,
        var=pd.DataFrame(
            index=p3_genes,
        ),
    )


def prepare_model(
    model_name,
    info,
    p3_genes,
    work_root,
):

    source = Path(
        info[
            "source_path"
        ]
    )

    sys.path.insert(
        0,
        str(source),
    )

    checkpoint = Path(
        info[
            "checkpoint"
        ]
    )

    if (
        sha256_file(
            checkpoint
        )
        != info[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "checkpoint SHA mismatch"
        )

    if model_name == "scprint1":

        from scdataloader import (
            Preprocessor,
        )

        from scdataloader.utils import (
            load_genes,
        )

        from scprint import scPrint

        from scprint.tasks.denoise import (
            Denoiser,
        )

        runtime = (
            work_root
            / "runtime_scprint1.ckpt"
        )

        shutil.copy2(
            checkpoint,
            runtime,
        )

        raw = torch.load(
            runtime,
            map_location="cpu",
            weights_only=False,
        )

        if (
            "prenorm"
            in raw[
                "hyper_parameters"
            ]
        ):
            raw[
                "hyper_parameters"
            ].pop(
                "prenorm"
            )

            torch.save(
                raw,
                runtime,
            )

        kwargs = {
            "precpt_gene_emb":
                None,

            "transformer":
                "flash",
        }

        if (
            "label_counts"
            in raw[
                "hyper_parameters"
            ]
        ):
            kwargs[
                "classes"
            ] = raw[
                "hyper_parameters"
            ][
                "label_counts"
            ]

        model = (
            scPrint
            .load_from_checkpoint(
                runtime,
                **kwargs,
            )
        )

        missing = (
            set(
                model.genes
            )
            - set(
                load_genes(
                    model.organisms
                ).index
            )
        )

        if missing:
            model._rm_genes(
                missing
            )

        model_genes = set(
            map(
                str,
                model.genes,
            )
        )

        fixed = [
            str(gene)
            for gene
            in p3_genes
            if str(gene)
            in model_genes
        ]

        expected = int(
            info[
                "qualified_p3_overlap"
            ]
        )

        if len(fixed) != expected:
            raise RuntimeError(
                "scPRINT-1 P3 overlap changed: "
                f"{len(fixed)} != {expected}"
            )

        model = model.to(
            "cuda"
        )

        denoiser = Denoiser(
            batch_size=8,
            num_workers=4,
            max_len=16384,
            how="some",
            predict_depth_mult=1,
            dtype=torch.float16,
            genelist=fixed,
        )

        def infer(
            adata,
        ):
            prep = Preprocessor(
                do_postp=False,
                force_preprocess=True,
                skip_validate=True,
            )

            processed = prep(
                adata
            )

            (
                _,
                random_indices,
                prediction,
            ) = denoiser(
                model=model,
                adata=processed,
            )

            if random_indices is not None:
                raise RuntimeError(
                    "unexpected cell subsampling"
                )

            return prediction

        return (
            model,
            fixed,
            infer,
        )

    if model_name == "scprint2":

        from scdataloader import (
            Preprocessor,
        )

        from scdataloader.utils import (
            load_genes,
        )

        from scprint2 import scPRINT2

        from scprint2.tasks.denoise import (
            Denoiser,
        )

        model = (
            scPRINT2
            .load_from_checkpoint(
                checkpoint,
                precpt_gene_emb=None,
                gene_pos_file=None,
            )
        )

        missing = (
            set(
                model.genes
            )
            - set(
                load_genes(
                    model.organisms
                ).index
            )
        )

        if missing:
            model._rm_genes(
                missing
            )

        model_genes = set(
            map(
                str,
                model.genes,
            )
        )

        fixed = [
            str(gene)
            for gene
            in p3_genes
            if str(gene)
            in model_genes
        ]

        expected = int(
            info[
                "qualified_p3_overlap"
            ]
        )

        if len(fixed) != expected:
            raise RuntimeError(
                "scPRINT-2 P3 overlap changed: "
                f"{len(fixed)} != {expected}"
            )

        model = model.to(
            "cuda"
        )

        denoiser = Denoiser(
            batch_size=8,
            num_workers=4,
            max_len=16384,
            how="some",
            predict_depth_mult=1,
            genelist=fixed,
            pred_embedding=
                model.pred_embedding,
            apply_zero_pred=False,
        )

        def infer(
            adata,
        ):
            prep = Preprocessor(
                do_postp=(
                    model.expr_emb_style
                    == "metacell"
                ),
                force_preprocess=True,
                skip_validate=True,
                use_raw=False,
                is_symbol=False,
            )

            processed = prep(
                adata
            )

            if (
                model.expr_emb_style
                == "metacell"
            ):
                sc.pp.neighbors(
                    processed,
                    use_rep="X_pca",
                )

            (
                _,
                random_indices,
                prediction,
            ) = denoiser(
                model=model,
                adata=processed,
            )

            if random_indices is not None:
                raise RuntimeError(
                    "unexpected cell subsampling"
                )

            return prediction

        return (
            model,
            fixed,
            infer,
        )

    raise RuntimeError(
        "unknown model"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        required=True,
    )

    parser.add_argument(
        "--task-id",
        required=True,
        type=int,
    )

    args = parser.parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    protocol = json.loads(
        protocol_path.read_text()
    )

    if (
        protocol.get(
            "protocol_id"
        )
        !=
        "sc2-p3-scprint-count-thinning-v1"
        or protocol.get(
            "status"
        )
        !=
        "FROZEN_BEFORE_INFERENCE"
    ):
        raise RuntimeError(
            "protocol identity/status mismatch"
        )

    expected_sha = os.environ.get(
        "EXPECTED_PROTOCOL_SHA"
    )

    if (
        expected_sha
        and sha256_file(
            protocol_path
        )
        != expected_sha
    ):
        raise RuntimeError(
            "protocol SHA mismatch"
        )

    tasks = {
        int(
            item[
                "task_id"
            ]
        ):
            item
        for item
        in protocol[
            "task_layout"
        ][
            "tasks"
        ]
    }

    if (
        len(tasks) != 18
        or args.task_id
        not in tasks
    ):
        raise RuntimeError(
            "task layout mismatch"
        )

    task = tasks[
        args.task_id
    ]

    model_name = task[
        "model"
    ]

    dataset = task[
        "dataset"
    ]

    q = float(
        task[
            "q"
        ]
    )

    model_info = protocol[
        "models"
    ][model_name]

    source = Path(
        model_info[
            "source_path"
        ]
    )

    source_head = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )

    if (
        source_head
        != model_info[
            "source_commit"
        ]
    ):
        raise RuntimeError(
            "source commit mismatch"
        )

    panels_path = Path(
        protocol[
            "panels"
        ][
            "receipt"
        ]
    )

    views_path = Path(
        protocol[
            "count_views"
        ][
            "receipt"
        ]
    )

    vocab_path = Path(
        protocol[
            "p3_vocabulary"
        ][
            "path"
        ]
    )

    if (
        sha256_file(
            panels_path
        )
        != protocol[
            "panels"
        ][
            "receipt_sha256"
        ]
    ):
        raise RuntimeError(
            "panel receipt SHA mismatch"
        )

    if (
        sha256_file(
            views_path
        )
        != protocol[
            "count_views"
        ][
            "receipt_sha256"
        ]
    ):
        raise RuntimeError(
            "count-view receipt SHA mismatch"
        )

    if (
        sha256_file(
            vocab_path
        )
        != protocol[
            "p3_vocabulary"
        ][
            "sha256"
        ]
    ):
        raise RuntimeError(
            "vocabulary SHA mismatch"
        )

    panels = json.loads(
        panels_path.read_text()
    )

    views = json.loads(
        views_path.read_text()
    )

    p3_genes = load_p3_genes(
        vocab_path
    )

    p3_index = {
        str(gene):
            index
        for index, gene
        in enumerate(
            p3_genes
        )
    }

    records = [
        item
        for item
        in panels[
            "datasets_detail"
        ][dataset][
            "panels"
        ]
        if np.isclose(
            float(
                item["q"]
            ),
            q,
            atol=1.0e-12,
            rtol=0.0,
        )
    ]

    records = sorted(
        records,
        key=lambda x:
            int(
                x[
                    "replicate"
                ]
            ),
    )

    if [
        int(
            item[
                "replicate"
            ]
        )
        for item
        in records
    ] != [
        1,
        2,
        3,
        4,
        5,
    ]:
        raise RuntimeError(
            "replicate grid mismatch"
        )

    for item in records:
        panel = Path(
            item[
                "path"
            ]
        )

        if not panel.is_file():
            raise RuntimeError(
                f"missing panel: {panel}"
            )

    view = views[
        "datasets"
    ][dataset]

    full_path = Path(
        view[
            "counts_path"
        ]
    )

    available_path = Path(
        view[
            "available_path"
        ]
    )

    if (
        sha256_file(
            full_path
        )
        != view[
            "counts_file_sha256"
        ]
    ):
        raise RuntimeError(
            "full-count file SHA mismatch"
        )

    if (
        sha256_file(
            available_path
        )
        != view[
            "available_file_sha256"
        ]
    ):
        raise RuntimeError(
            "availability file SHA mismatch"
        )

    full_counts = np.load(
        full_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    available = np.asarray(
        np.load(
            available_path,
            allow_pickle=False,
        ),
        dtype=bool,
    )

    if (
        full_counts.shape
        != (
            5000,
            16384,
        )
        or available.shape
        != (
            16384,
        )
    ):
        raise RuntimeError(
            "count-view dimensions mismatch"
        )

    output_root = Path(
        protocol[
            "output_root"
        ]
    )

    final_dir = (
        output_root
        / (
            f"task_{args.task_id:02d}_"
            f"{model_name}_"
            f"{dataset}_"
            f"q{int(round(q * 100)):03d}"
        )
    )

    temp_dir = (
        output_root
        / (
            f".tmp_task_"
            f"{args.task_id:02d}_"
            f"{os.getpid()}"
        )
    )

    work_dir = (
        output_root
        / (
            f".work_task_"
            f"{args.task_id:02d}_"
            f"{os.getpid()}"
        )
    )

    if final_dir.exists():
        raise RuntimeError(
            f"final output exists: "
            f"{final_dir}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir.mkdir()
    work_dir.mkdir()

    old_cwd = Path.cwd()

    try:

        os.chdir(
            work_dir
        )

        (
            model,
            fixed_p3,
            infer,
        ) = prepare_model(
            model_name,
            model_info,
            p3_genes,
            work_dir,
        )

        fixed_idx = np.asarray(
            [
                p3_index[gene]
                for gene
                in fixed_p3
            ],
            dtype=np.int64,
        )

        effective_idx = fixed_idx[
            available[
                fixed_idx
            ]
        ]

        effective_genes = (
            np.asarray(
                p3_genes,
                dtype=str,
            )[
                effective_idx
            ]
        )

        if effective_idx.size < 10000:
            raise RuntimeError(
                "insufficient evaluated genes: "
                f"{effective_idx.size}"
            )

        gene_order_sha = (
            hashlib.sha256(
                "\n".join(
                    effective_genes.tolist()
                ).encode()
            )
            .hexdigest()
        )

        full_effective = np.asarray(
            full_counts[
                :,
                effective_idx,
            ],
            dtype=np.float32,
        )

        target = cp10k_log1p(
            full_effective
        )

        true_zero = (
            full_effective
            == 0
        )

        conditions = []

        for item in records:

            replicate = int(
                item[
                    "replicate"
                ]
            )

            panel_path = Path(
                item[
                    "path"
                ]
            )

            if (
                sha256_file(
                    panel_path
                )
                != item[
                    "sha256"
                ]
            ):
                raise RuntimeError(
                    "panel SHA mismatch: "
                    f"rep={replicate}"
                )

            with np.load(
                panel_path,
                allow_pickle=False,
            ) as panel:

                thinned_counts = np.asarray(
                    panel[
                        "thinned_counts"
                    ],
                    dtype=np.float32,
                )

                lost_positive = np.asarray(
                    panel[
                        "lost_positive_mask"
                    ],
                    dtype=bool,
                )

            if (
                thinned_counts.shape
                != (
                    5000,
                    16384,
                )
                or lost_positive.shape
                != (
                    5000,
                    16384,
                )
            ):
                raise RuntimeError(
                    "panel array dimensions mismatch"
                )

            thinned_effective = (
                thinned_counts[
                    :,
                    effective_idx,
                ]
            )

            positive = (
                (
                    full_effective
                    > 0
                )
                & (
                    thinned_effective
                    == 0
                )
            )

            if not np.array_equal(
                positive,
                lost_positive[
                    :,
                    effective_idx,
                ],
            ):
                raise RuntimeError(
                    "lost-positive mask mismatch"
                )

            corrupted = cp10k_log1p(
                thinned_effective
            )

            input_adata = make_adata(
                thinned_counts,
                p3_genes,
            )

            prediction_adata = infer(
                input_adata
            )

            if (
                "scprint_mu"
                not in
                prediction_adata.layers
            ):
                raise RuntimeError(
                    "scprint_mu missing"
                )

            returned = list(
                map(
                    str,
                    prediction_adata.var_names,
                )
            )

            returned_index = {
                gene:
                    index
                for index, gene
                in enumerate(
                    returned
                )
            }

            if (
                len(returned_index)
                != len(returned)
            ):
                raise RuntimeError(
                    "duplicate native output genes"
                )

            missing = [
                gene
                for gene
                in effective_genes
                if gene
                not in returned_index
            ]

            if missing:
                raise RuntimeError(
                    "native output missing "
                    f"{len(missing)} frozen "
                    "evaluation genes"
                )

            prediction_indices = (
                np.asarray(
                    [
                        returned_index[gene]
                        for gene
                        in effective_genes
                    ],
                    dtype=np.int64,
                )
            )

            mu = prediction_adata.layers[
                "scprint_mu"
            ]

            if sparse.issparse(mu):

                prediction_counts = (
                    mu[
                        :,
                        prediction_indices,
                    ]
                    .toarray()
                    .astype(
                        np.float32,
                    )
                )

            else:

                prediction_counts = (
                    np.asarray(
                        mu[
                            :,
                            prediction_indices,
                        ],
                        dtype=np.float32,
                    )
                )

            prediction = cp10k_log1p(
                prediction_counts
            )

            model_masked = masked_metrics(
                prediction,
                target,
                positive,
            )

            baseline_masked = masked_metrics(
                corrupted,
                target,
                positive,
            )

            model_full = full_metrics(
                prediction,
                target,
                true_zero,
            )

            baseline_full = full_metrics(
                corrupted,
                target,
                true_zero,
            )

            model_metrics = {
                **model_masked,
                **model_full,
            }

            baseline_metrics = {
                **baseline_masked,
                **baseline_full,
            }

            gains = {
                "masked_mse_gain_vs_corrupted":
                    float(
                        baseline_masked[
                            "masked_mse"
                        ]
                        - model_masked[
                            "masked_mse"
                        ]
                    ),

                "masked_mae_gain_vs_corrupted":
                    float(
                        baseline_masked[
                            "masked_mae"
                        ]
                        - model_masked[
                            "masked_mae"
                        ]
                    ),

                "recovery_gain_vs_corrupted":
                    float(
                        model_masked[
                            "recovery_index"
                        ]
                        - baseline_masked[
                            "recovery_index"
                        ]
                    ),

                "full_mse_gain_vs_corrupted":
                    float(
                        baseline_full[
                            "full_available_mse"
                        ]
                        - model_full[
                            "full_available_mse"
                        ]
                    ),
            }

            numeric_values = []

            for section in (
                model_metrics,
                baseline_metrics,
                gains,
            ):
                for value in section.values():

                    if isinstance(
                        value,
                        (
                            int,
                            float,
                            np.integer,
                            np.floating,
                        ),
                    ):
                        numeric_values.append(
                            float(value)
                        )

            if not np.isfinite(
                numeric_values
            ).all():
                raise RuntimeError(
                    "nonfinite metric"
                )

            payload = {
                "schema":
                    "sc2-p3-scprint-count-thinning-condition-v1",

                "status":
                    "PASS",

                "model":
                    model_name,

                "dataset":
                    dataset,

                "q":
                    q,

                "loss_fraction":
                    float(
                        1.0 - q
                    ),

                "replicate":
                    replicate,

                "thinning_seed":
                    int(
                        item[
                            "thinning_seed"
                        ]
                    ),

                "panel":
                    str(
                        panel_path
                    ),

                "panel_sha256":
                    item[
                        "sha256"
                    ],

                "checkpoint":
                    model_info[
                        "checkpoint"
                    ],

                "checkpoint_sha256":
                    model_info[
                        "checkpoint_sha256"
                    ],

                "model_p3_genes":
                    len(
                        fixed_p3
                    ),

                "dataset_available_genes":
                    int(
                        available.sum()
                    ),

                "evaluated_genes":
                    int(
                        effective_idx.size
                    ),

                "evaluated_gene_order_sha256":
                    gene_order_sha,

                "native_returned_genes":
                    int(
                        prediction_adata.n_vars
                    ),

                "native_prediction_scale":
                    "scprint_mu count scale",

                "evaluation_transform":
                    (
                        "independent CP10K+log1p "
                        "on fixed overlap"
                    ),

                "predict_depth_mult":
                    1,

                "apply_zero_pred":
                    False,

                "model_metrics":
                    model_metrics,

                "corrupted_baseline":
                    baseline_metrics,

                "gains":
                    gains,

                "technical_replicate":
                    True,

                "biological_replicate":
                    False,
            }

            (
                temp_dir
                / (
                    f"rep{replicate:02d}_"
                    "metrics.json"
                )
            ).write_text(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )

            conditions.append(
                payload
            )

            del (
                thinned_counts,
                lost_positive,
                thinned_effective,
                corrupted,
                input_adata,
                prediction_adata,
                prediction_counts,
                prediction,
            )

            gc.collect()
            torch.cuda.empty_cache()

            for h5 in work_dir.rglob(
                "*.h5ad"
            ):
                try:
                    h5.unlink()
                except FileNotFoundError:
                    pass

        summary = {
            "schema":
                "sc2-p3-scprint-count-thinning-task-summary-v1",

            "status":
                "PASS",

            "task_id":
                args.task_id,

            "model":
                model_name,

            "dataset":
                dataset,

            "q":
                q,

            "replicate_count":
                5,

            "model_p3_genes":
                len(
                    fixed_p3
                ),

            "dataset_available_genes":
                int(
                    available.sum()
                ),

            "evaluated_genes":
                int(
                    effective_idx.size
                ),

            "evaluated_gene_order_sha256":
                gene_order_sha,

            "conditions":
                conditions,
        }

        (
            temp_dir
            / "task_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

        files = sorted(
            path
            for path
            in temp_dir.rglob("*")
            if path.is_file()
        )

        (
            temp_dir
            / "SHA256SUMS.txt"
        ).write_text(
            "".join(
                f"{sha256_file(path)}  "
                f"{path.relative_to(temp_dir).as_posix()}\n"
                for path
                in files
            )
        )

        os.replace(
            temp_dir,
            final_dir,
        )

        print(
            "SCPRINT_COUNT_THINNING_TASK=PASS"
        )

    finally:

        os.chdir(
            old_cwd
        )

        if temp_dir.exists():
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

        if work_dir.exists():
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()
