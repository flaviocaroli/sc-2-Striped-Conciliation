#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


ZERO_EPS = 1e-8


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def safe_spearman(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if (
        len(a) < 2
        or np.std(a) == 0
        or np.std(b) == 0
    ):
        return np.nan

    r = spearmanr(a, b).statistic

    return (
        float(r)
        if np.isfinite(r)
        else np.nan
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--task-id", required=True, type=int)
    args = ap.parse_args()

    protocol_path = Path(args.protocol).resolve()
    p = json.loads(protocol_path.read_text())

    expected = os.environ.get(
        "SC2_SCFOUNDATION_FINAL_PROTOCOL_SHA"
    )

    if expected and sha(protocol_path) != expected:
        raise RuntimeError("protocol SHA mismatch")

    tasks = {
        int(x["task_id"]): x
        for x in p["tasks"]
    }

    if args.task_id not in tasks:
        raise RuntimeError("unknown task")

    task = tasks[args.task_id]

    panel = Path(task["panel"])

    if sha(panel) != task["panel_sha256"]:
        raise RuntimeError("benchmark SHA mismatch")

    ckpt = Path(p["checkpoint"]["path"])

    if sha(ckpt) != p["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint SHA mismatch")

    src = Path(p["source"]["path"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    with np.load(panel, allow_pickle=False) as d:
        required = {
            "x",
            "y",
            "synthetic_mask",
            "available_gene_mask",
        }

        if not required <= set(d.files):
            raise RuntimeError(
                f"panel keys missing: {required-set(d.files)}"
            )

        x = np.asarray(
            d["x"],
            dtype=np.float32,
        )

        y = np.asarray(
            d["y"],
            dtype=np.float32,
        )

        synthetic = np.asarray(
            d["synthetic_mask"],
            dtype=bool,
        )

        if "available_gene_mask" in d.files:
            available = np.asarray(
                d["available_gene_mask"],
                dtype=bool,
            )
            availability_source = "panel_available_gene_mask"
        elif task["dataset"] == "internal_test":
            # Internal corpus uses the complete frozen 4096-gene
            # vocabulary; no separate availability mask is needed.
            available = np.ones(
                4096,
                dtype=bool,
            )
            availability_source = "implicit_all_4096_internal"
        else:
            raise RuntimeError(
                "external panel lacks available_gene_mask"
            )

    if x.shape != (5000, 4096):
        raise RuntimeError(
            f"unexpected x shape {x.shape}"
        )

    if y.shape != x.shape:
        raise RuntimeError("y shape mismatch")

    if synthetic.shape != x.shape:
        raise RuntimeError("mask shape mismatch")

    if available.shape != (4096,):
        raise RuntimeError("availability shape mismatch")

    if not np.isfinite(x).all():
        raise RuntimeError("nonfinite x")

    if not np.isfinite(y).all():
        raise RuntimeError("nonfinite y")

    # --------------------------------------------------------
    # Frozen SC2 Ensembl -> symbol -> scFoundation map
    # --------------------------------------------------------

    vocab = pd.read_parquet(
        p["mapping"]["sc2_vocabulary"]
    )

    stats = pd.read_parquet(
        p["mapping"]["gene_stats"]
    )

    sf = pd.read_csv(
        p["mapping"]["scfoundation_vocabulary"],
        sep="\t",
    )

    mapping = (
        vocab[
            ["gene_index", "ensembl_id"]
        ]
        .merge(
            stats[
                ["ensembl_id", "feature_name"]
            ],
            on="ensembl_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("gene_index")
    )

    sf_genes = (
        sf["gene_name"]
        .astype(str)
        .tolist()
    )

    sf_index = {
        g: i
        for i, g in enumerate(sf_genes)
    }

    mapping["feature_name"] = (
        mapping["feature_name"].astype(str)
    )

    mapping = mapping[
        mapping["feature_name"].isin(sf_index)
    ].copy()

    # Same deterministic collision rule frozen at smoke.
    mapping = mapping.drop_duplicates(
        subset=["feature_name"],
        keep="first",
    )

    if len(mapping) != 4006:
        raise RuntimeError(
            f"mapping changed: {len(mapping)}"
        )

    sc2_idx = mapping[
        "gene_index"
    ].astype(int).to_numpy()

    sf_idx = np.asarray(
        [
            sf_index[g]
            for g in mapping["feature_name"]
        ],
        dtype=np.int64,
    )

    mapped_mask = np.zeros(
        4096,
        dtype=bool,
    )

    mapped_mask[sc2_idx] = True

    eval_gene_mask = (
        mapped_mask
        & available
    )

    n_eval_genes = int(
        eval_gene_mask.sum()
    )

    if n_eval_genes < 3500:
        raise RuntimeError(
            f"insufficient evaluable genes: {n_eval_genes}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sys.path.insert(
        0,
        str(src / "model"),
    )

    from load import (
        getEncoerDecoderData,
        load_model_frommmf,
    )

    model, config = load_model_frommmf(
        str(ckpt),
        key="gene",
    )

    model.eval()

    # Full 4096 positions, NaN where scFoundation
    # cannot represent the SC2 gene.
    prediction = np.full(
        x.shape,
        np.nan,
        dtype=np.float32,
    )

    batch_size = int(
        p["inference"]["batch_size"]
    )

    def decode_batch(x_sc2):
        b = x_sc2.shape[0]

        sf_input = np.zeros(
            (b, 19264),
            dtype=np.float32,
        )

        sf_input[:, sf_idx] = (
            x_sc2[:, sc2_idx]
        )

        total = (
            sf_input
            .astype(np.float64)
            .sum(axis=1)
        )

        if (
            (~np.isfinite(total)).any()
            or (total <= 0).any()
        ):
            raise RuntimeError(
                "invalid mapped library total"
            )

        extra = np.stack(
            [
                np.full(
                    b,
                    4.0,
                    dtype=np.float32,
                ),
                np.log10(total).astype(
                    np.float32
                ),
            ],
            axis=1,
        )

        full = np.concatenate(
            [sf_input, extra],
            axis=1,
        )

        t = torch.from_numpy(
            full
        ).cuda()

        (
            encoder_data,
            encoder_position_gene_ids,
            encoder_data_padding,
            encoder_labels,
            decoder_data,
            decoder_data_padding,
            new_data_raw,
            data_mask_labels,
            decoder_position_gene_ids,
        ) = getEncoerDecoderData(
            t.float(),
            t.float(),
            config,
        )

        with torch.no_grad():
            out = model.forward(
                x=encoder_data,
                padding_label=
                    encoder_data_padding,
                encoder_position_gene_ids=
                    encoder_position_gene_ids,
                encoder_labels=
                    encoder_labels,
                decoder_data=
                    decoder_data,
                mask_gene_name=False,
                mask_labels=None,
                decoder_position_gene_ids=
                    decoder_position_gene_ids,
                decoder_data_padding_labels=
                    decoder_data_padding,
            )

        out = (
            out[:, :19264]
            .contiguous()
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if out.shape != (b, 19264):
            raise RuntimeError(
                f"decoder shape {out.shape}"
            )

        return out[:, sf_idx]

    n = x.shape[0]

    for start in range(
        0,
        n,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            n,
        )

        decoded = decode_batch(
            x[start:stop]
        )

        if not np.isfinite(decoded).all():
            raise RuntimeError(
                "nonfinite decoder result"
            )

        prediction[
            start:stop,
            sc2_idx,
        ] = decoded

        if (
            stop == n
            or stop % 250 == 0
        ):
            print(
                f"DECODED={stop}/{n}",
                flush=True,
            )

    # --------------------------------------------------------
    # Exact metrics on mapped + dataset-available genes
    # --------------------------------------------------------

    masked = (
        synthetic
        & eval_gene_mask[None, :]
    )

    n_masked = int(
        masked.sum()
    )

    if n_masked == 0:
        raise RuntimeError(
            "no evaluable masked entries"
        )

    target_masked = y[masked]
    pred_masked = prediction[masked]

    if not np.isfinite(
        pred_masked
    ).all():
        raise RuntimeError(
            "masked prediction nonfinite"
        )

    diff = (
        pred_masked
        - target_masked
    )

    masked_mse = float(
        np.mean(diff ** 2)
    )

    masked_mae = float(
        np.mean(np.abs(diff))
    )

    target_sd = float(
        np.std(target_masked)
    )

    prediction_sd = float(
        np.std(pred_masked)
    )

    if target_sd <= 0:
        raise RuntimeError(
            "zero target SD"
        )

    recovery_index = float(
        1.0
        - masked_mse
        / (target_sd ** 2)
    )

    sd_ratio = float(
        prediction_sd
        / target_sd
    )

    # Direct-decoder reconstruction:
    # observed nonzeros copied exactly;
    # zero inputs on mapped genes get official decoder output.
    reconstruction = x.copy()

    zero_candidate = (
        (np.abs(x) <= ZERO_EPS)
        & eval_gene_mask[None, :]
    )

    reconstruction[
        zero_candidate
    ] = prediction[
        zero_candidate
    ]

    observed = (
        (np.abs(x) > ZERO_EPS)
        & eval_gene_mask[None, :]
    )

    if not np.array_equal(
        reconstruction[observed],
        x[observed],
    ):
        raise RuntimeError(
            "observed-value preservation failed"
        )

    true_zero = (
        (np.abs(y) <= ZERO_EPS)
        & eval_gene_mask[None, :]
    )

    tz_pred = prediction[true_zero]

    true_zero_changed_fraction = float(
        np.mean(
            np.abs(tz_pred) > ZERO_EPS
        )
    )

    true_zero_positive_fill_fraction = float(
        np.mean(
            tz_pred > ZERO_EPS
        )
    )

    true_zero_negative_fraction = float(
        np.mean(
            tz_pred < -ZERO_EPS
        )
    )

    # Gene-wise structural fidelity.
    gene_values = np.full(
        4096,
        np.nan,
        dtype=np.float64,
    )

    for j in np.flatnonzero(
        eval_gene_mask
    ):
        gene_values[j] = safe_spearman(
            reconstruction[:, j],
            y[:, j],
        )

    valid_gene = np.isfinite(
        gene_values
    )

    gene_spearman = float(
        np.mean(
            gene_values[valid_gene]
        )
    )

    # Cell/sample-wise rank fidelity.
    idx = np.flatnonzero(
        eval_gene_mask
    )

    sample_values = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    for i in range(n):
        sample_values[i] = safe_spearman(
            reconstruction[i, idx],
            y[i, idx],
        )

    sample_spearman = float(
        np.nanmean(sample_values)
    )

    # --------------------------------------------------------
    # Save atomically
    # --------------------------------------------------------

    final = (
        Path(p["output_root"])
        / (
            f"task_{args.task_id:02d}_"
            f"{task['dataset']}_"
            f"mask{task['mask_percent']}"
        )
    )

    temp = (
        Path(p["output_root"])
        / (
            f".tmp_task_"
            f"{args.task_id:02d}_"
            f"{os.getpid()}"
        )
    )

    if final.exists() or temp.exists():
        raise RuntimeError(
            "output collision"
        )

    temp.mkdir(
        parents=True,
    )

    np.save(
        temp / "prediction_sc2.npy",
        prediction,
        allow_pickle=False,
    )

    np.save(
        temp / "gene_spearman.npy",
        gene_values,
        allow_pickle=False,
    )

    np.save(
        temp / "sample_spearman.npy",
        sample_values,
        allow_pickle=False,
    )

    summary = {
        "status": "PASS",

        "model":
            "scFoundation",

        "source_commit":
            p["source"]["commit"],

        "dataset":
            task["dataset"],

        "mask_percent":
            task["mask_percent"],

        "cells":
            5000,

        "sc2_genes":
            4096,

        "scfoundation_mapped_genes":
            4006,

        "dataset_available_genes":
            int(available.sum()),

        "availability_source":
            availability_source,

        "evaluated_genes":
            n_eval_genes,

        "evaluated_gene_fraction":
            n_eval_genes / 4096.0,

        "n_masked":
            n_masked,

        "masked_mse":
            masked_mse,

        "masked_mae":
            masked_mae,

        "target_sd":
            target_sd,

        "prediction_sd":
            prediction_sd,

        "sd_ratio":
            sd_ratio,

        "recovery_index":
            recovery_index,

        "gene_spearman":
            gene_spearman,

        "n_gene_spearman_valid":
            int(valid_gene.sum()),

        "sample_spearman":
            sample_spearman,

        "true_zero_changed_fraction":
            true_zero_changed_fraction,

        "true_zero_positive_fill_fraction":
            true_zero_positive_fill_fraction,

        "true_zero_negative_fraction":
            true_zero_negative_fraction,

        "observed_nonzero_changed_fraction":
            0.0,

        "repair_policy":
            (
                "official continuous decoder applied "
                "to zero input entries; observed "
                "nonzeros copied exactly"
            ),

        "threshold":
            None,

        "threshold_reason":
            (
                "not applicable: official "
                "scFoundation gene-expression decoder "
                "has no selective gate; no ad-hoc "
                "threshold introduced"
            ),

        "pretraining_overlap_excluded":
            False,

        "panel":
            str(panel),

        "panel_sha256":
            task["panel_sha256"],

        "checkpoint_sha256":
            p["checkpoint"]["sha256"],
    }

    (
        temp / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    )

    files = sorted(
        q for q in temp.iterdir()
        if q.is_file()
    )

    (
        temp / "SHA256SUMS.txt"
    ).write_text(
        "".join(
            f"{sha(q)}  {q.name}\n"
            for q in files
            if q.name != "SHA256SUMS.txt"
        )
    )

    os.replace(
        temp,
        final,
    )

    print(
        json.dumps(
            summary,
            sort_keys=True,
        )
    )

    print(
        "SCFOUNDATION_FINAL_TASK=PASS"
    )


if __name__ == "__main__":
    main()
