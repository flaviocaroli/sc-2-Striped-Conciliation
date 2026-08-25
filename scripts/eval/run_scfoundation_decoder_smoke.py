#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    args = ap.parse_args()

    protocol_path = Path(args.protocol).resolve()
    p = json.loads(protocol_path.read_text())

    expected = os.environ.get(
        "SC2_SCFOUNDATION_SMOKE_PROTOCOL_SHA"
    )

    if expected and sha(protocol_path) != expected:
        raise RuntimeError("protocol SHA mismatch")

    src = Path(p["source"]["path"])
    ckpt = Path(p["checkpoint"]["path"])
    inp = Path(p["input"]["path"])
    outroot = Path(p["output_root"])

    if sha(ckpt) != p["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint SHA mismatch")

    if sha(inp) != p["input"]["sha256"]:
        raise RuntimeError("input SHA mismatch")

    if outroot.exists():
        raise RuntimeError("output collision")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

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

    X = np.load(
        inp,
        allow_pickle=False,
    )

    if X.shape != (4, 19264):
        raise RuntimeError(
            f"unexpected input shape: {X.shape}"
        )

    model, config = load_model_frommmf(
        str(ckpt),
        key="gene",
    )

    model.eval()

    required_config = {
        "pad_token_id",
        "seq_len",
    }

    missing = required_config - set(config)
    if missing:
        raise RuntimeError(
            f"checkpoint missing config keys: {missing}"
        )

    def decode(row):
        totalcount = float(
            np.asarray(row, dtype=np.float64).sum()
        )

        if (
            not math.isfinite(totalcount)
            or totalcount <= 0
        ):
            raise RuntimeError(
                f"invalid input totalcount: {totalcount}"
            )

        # Official scFoundation get_embedding.py semantics:
        # input_type=singlecell
        # pre_normalized=T
        # tgthighres=t4
        values = row.tolist() + [
            4.0,
            math.log10(totalcount),
        ]

        pretrain_gene_x = torch.tensor(
            values,
            dtype=torch.float32,
            device="cuda",
        ).unsqueeze(0)

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
            pretrain_gene_x.float(),
            pretrain_gene_x.float(),
            config,
        )

        with torch.no_grad():
            out = model.forward(
                x=encoder_data,
                padding_label=encoder_data_padding,
                encoder_position_gene_ids=
                    encoder_position_gene_ids,
                encoder_labels=encoder_labels,
                decoder_data=decoder_data,
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

        if out.shape != (1, 19264):
            raise RuntimeError(
                f"decoder output shape: {out.shape}"
            )

        return out[0]

    decoded = []

    for i in range(4):
        y = decode(X[i])

        if not np.isfinite(y).all():
            raise RuntimeError(
                f"nonfinite output for cell {i}"
            )

        decoded.append(y)

        print(
            f"DECODED_CELL={i+1}/4 "
            f"MIN={float(y.min()):.8g} "
            f"MAX={float(y.max()):.8g} "
            f"MEAN={float(y.mean()):.8g} "
            f"SD={float(y.std()):.8g}",
            flush=True,
        )

    Y = np.stack(decoded).astype(
        np.float32,
        copy=False,
    )

    repeat = decode(X[0])

    repeat_max_abs = float(
        np.max(
            np.abs(
                repeat.astype(np.float32)
                - Y[0]
            )
        )
    )

    if repeat_max_abs > 1e-5:
        raise RuntimeError(
            "repeatability gate failed: "
            f"{repeat_max_abs}"
        )

    if float(Y.std()) <= 1e-8:
        raise RuntimeError(
            "decoder output is constant"
        )

    if np.allclose(
        Y[0],
        Y[1],
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            "different cells have identical output"
        )

    outroot.mkdir(
        parents=True,
        exist_ok=False,
    )

    matrix_path = (
        outroot
        / "decoded_gene_expression.npy"
    )

    np.save(
        matrix_path,
        Y,
        allow_pickle=False,
    )

    summary = {
        "status": "PASS",
        "technical_compatibility": True,
        "scientific_performance_evaluated": False,
        "cells": 4,
        "genes": 19264,
        "finite_fraction":
            float(np.isfinite(Y).mean()),
        "output_min": float(Y.min()),
        "output_max": float(Y.max()),
        "output_mean": float(Y.mean()),
        "output_sd": float(Y.std()),
        "repeat_max_abs_difference":
            repeat_max_abs,
        "different_cells_nonidentical":
            True,
        "checkpoint_sha256":
            sha(ckpt),
        "decoded_sha256":
            sha(matrix_path),
        "target_y_used": False,
        "output_type":
            "gene_expression",
        "official_branch_marked_not_recommended":
            True,
    }

    (
        outroot
        / "smoke_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    print("SCFOUNDATION_DECODER_SMOKE=PASS")
    print("OUTPUT_SHAPE=4x19264")
    print(
        "REPEAT_MAX_ABS_DIFF="
        f"{repeat_max_abs:.8g}"
    )


if __name__ == "__main__":
    main()
