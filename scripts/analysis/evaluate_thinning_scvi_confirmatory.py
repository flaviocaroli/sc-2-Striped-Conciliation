#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import scvi
from scipy import sparse
from scipy.stats import rankdata
import torch


SEEDS = (
    20260728,
    20260729,
    20260730,
)

DATASETS = (
    "internal_test",
    "baron_pancreas",
    "zheng68k",
)

QKEYS = ("085", "070", "050")
ZERO_EPS = 1e-8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def target_rank_state(y, available):
    idx = np.flatnonzero(available)

    ranks = rankdata(
        np.asarray(
            y[:, idx],
            dtype=np.float64,
        ),
        axis=0,
        method="average",
    )

    ranks -= ranks.mean(axis=0)

    norm = np.sqrt(
        np.sum(
            ranks * ranks,
            axis=0,
        )
    )

    return idx, ranks, norm


def gene_spearman(a, idx, target_ranks, target_norm):
    ranks = rankdata(
        np.asarray(
            a[:, idx],
            dtype=np.float64,
        ),
        axis=0,
        method="average",
    )

    ranks -= ranks.mean(axis=0)

    norm = np.sqrt(
        np.sum(
            ranks * ranks,
            axis=0,
        )
    )

    denom = norm * target_norm
    valid = denom > 0

    corr = np.full(
        idx.size,
        np.nan,
        dtype=np.float64,
    )

    corr[valid] = (
        np.sum(
            ranks[:, valid]
            * target_ranks[:, valid],
            axis=0,
        )
        / denom[valid]
    )

    return corr, float(
        np.nanmean(corr)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--task-id", required=True, type=int)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    protocol_path = Path(args.protocol).resolve()
    p = json.loads(protocol_path.read_text())

    expected = os.environ.get(
        "SC2_THINNING_CONFIRMATORY_PROTOCOL_SHA"
    )

    if expected and sha(protocol_path) != expected:
        raise RuntimeError("protocol SHA mismatch")

    if args.task_id not in range(15):
        raise RuntimeError("task outside 0..14")

    seed = SEEDS[
        args.task_id // 5
    ]

    replicate = (
        args.task_id % 5
    ) + 1

    if str(scvi.__version__) != "1.5.0.post1":
        raise RuntimeError(
            f"unexpected scVI version {scvi.__version__}"
        )

    panel_infos = []

    for dataset in DATASETS:
        for qkey in QKEYS:

            key = (
                f"{dataset}__q{qkey}__"
                f"rep{replicate:02d}"
            )

            info = p["panels"][key]
            path = Path(info["path"])

            if (
                not path.is_file()
                or sha(path) != info["sha256"]
            ):
                raise RuntimeError(
                    f"panel integrity failure: {path}"
                )

            panel_infos.append(
                (dataset, qkey, path)
            )

    if args.check_only:
        print(
            "THINNING_SCVI_CONFIRMATORY_CHECK=PASS "
            f"TASK={args.task_id} "
            f"SEED={seed} REP={replicate} "
            "PANELS=9 EVALUATIONS=9"
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    root = (
        Path(p["output_root"])
        / "scvi"
    )

    final = (
        root
        / (
            f"task_{args.task_id:02d}_"
            f"seed{seed}_rep{replicate:02d}"
        )
    )

    temp = (
        root
        / f".tmp_task_{args.task_id:02d}_{os.getpid()}"
    )

    if final.exists() or temp.exists():
        raise RuntimeError("output collision")

    temp.mkdir(parents=True)

    manifest = []

    try:
        for dataset, qkey, panel_path in panel_infos:

            # y deliberately withheld until after fit + decode.
            with np.load(
                panel_path,
                allow_pickle=False,
            ) as d:

                counts = np.asarray(
                    d["thinned_counts"],
                    dtype=np.uint32,
                    order="C",
                ).copy()

                x = np.asarray(
                    d["x"],
                    dtype=np.float32,
                ).copy()

                lost = np.asarray(
                    d["lost_positive_mask"],
                    dtype=bool,
                ).copy()

                original_zero = np.asarray(
                    d["originally_zero_mask"],
                    dtype=bool,
                ).copy()

                available = np.asarray(
                    d["available_gene_mask"],
                    dtype=bool,
                ).copy()

                panel_dataset = str(
                    d["dataset"].item()
                )

                q = float(
                    d["thinning_q"]
                )

                rep = int(
                    d["thinning_replicate"]
                )

            if counts.shape != (5000, 4096):
                raise RuntimeError(
                    f"unexpected count shape {counts.shape}"
                )

            if x.shape != counts.shape:
                raise RuntimeError("x/count shape mismatch")

            if panel_dataset != dataset:
                raise RuntimeError("dataset metadata mismatch")

            if rep != replicate:
                raise RuntimeError("replicate metadata mismatch")

            expected_q = {
                "085": 0.85,
                "070": 0.70,
                "050": 0.50,
            }[qkey]

            if abs(q - expected_q) > 1e-6:
                raise RuntimeError("q metadata mismatch")

            if np.any(
                counts[:, ~available] != 0
            ):
                raise RuntimeError(
                    "unavailable gene has counts"
                )

            if np.any(
                counts.sum(
                    axis=1,
                    dtype=np.uint64,
                ) == 0
            ):
                raise RuntimeError(
                    "zero-library cell"
                )

            if np.any(
                lost & original_zero
            ):
                raise RuntimeError(
                    "label overlap"
                )

            scvi.settings.seed = int(seed)

            adata = ad.AnnData(
                X=sparse.csr_matrix(
                    counts
                )
            )

            adata.obs_names = [
                f"cell_{i:05d}"
                for i in range(
                    counts.shape[0]
                )
            ]

            adata.var_names = [
                f"gene_{j:04d}"
                for j in range(
                    counts.shape[1]
                )
            ]

            scvi.model.SCVI.setup_anndata(
                adata,
                batch_key=None,
            )

            model = scvi.model.SCVI(
                adata,
                n_hidden=128,
                n_latent=10,
                n_layers=1,
                dispersion="gene",
                gene_likelihood="zinb",
                use_observed_lib_size=True,
            )

            model.train(
                max_epochs=400
            )

            decoded = np.asarray(
                model.get_normalized_expression(
                    library_size=10000,
                    return_numpy=True,
                ),
                dtype=np.float32,
                order="C",
            )

            if decoded.shape != counts.shape:
                raise RuntimeError(
                    "scVI decoded shape mismatch"
                )

            if (
                not np.isfinite(decoded).all()
                or np.any(decoded < 0)
            ):
                raise RuntimeError(
                    "invalid scVI decoded expression"
                )

            prediction = np.log1p(
                decoded
            ).astype(
                np.float32,
                copy=False,
            )

            score = prediction

            # Target becomes accessible only now.
            with np.load(
                panel_path,
                allow_pickle=False,
            ) as d:
                y = np.asarray(
                    d["y"],
                    dtype=np.float32,
                ).copy()

            threshold = float(
                p["thresholds"][
                    "scvi"
                ][qkey]["threshold"]
            )

            zero_input = (
                (np.abs(x) <= ZERO_EPS)
                & available[None, :]
            )

            selected = (
                zero_input
                & (score >= threshold)
            )

            reconstruction = x.copy()

            reconstruction[selected] = prediction[selected]

            observed = (
                (np.abs(x) > ZERO_EPS)
                & available[None, :]
            )

            if not np.array_equal(
                reconstruction[observed],
                x[observed],
            ):
                raise RuntimeError(
                    "observed-value preservation failed"
                )

            target = y[lost].astype(np.float64)
            repaired = reconstruction[lost].astype(np.float64)
            candidate = prediction[lost].astype(np.float64)
            corrupted = x[lost].astype(np.float64)

            target_var = float(
                np.var(target)
            )

            if target_var <= 0:
                raise RuntimeError(
                    "nonpositive target variance"
                )

            mse = float(
                np.mean(
                    (repaired - target) ** 2
                )
            )

            candidate_mse = float(
                np.mean(
                    (candidate - target) ** 2
                )
            )

            corrupted_mse = float(
                np.mean(
                    (corrupted - target) ** 2
                )
            )

            recovery = (
                1.0
                - mse / target_var
            )

            candidate_recovery = (
                1.0
                - candidate_mse / target_var
            )

            corrupted_recovery = (
                1.0
                - corrupted_mse / target_var
            )

            tp = int(
                np.count_nonzero(
                    selected & lost
                )
            )

            fp = int(
                np.count_nonzero(
                    selected & original_zero
                )
            )

            recall = (
                tp / int(lost.sum())
            )

            fill = (
                fp / int(
                    original_zero.sum()
                )
            )

            precision = (
                tp / (tp + fp)
                if tp + fp
                else 0.0
            )

            target_state = target_rank_state(
                y,
                available,
            )

            idx, target_ranks, target_norm = target_state

            gene_corr, mean_gene = gene_spearman(
                reconstruction,
                idx,
                target_ranks,
                target_norm,
            )

            _, corrupted_gene = gene_spearman(
                x,
                idx,
                target_ranks,
                target_norm,
            )

            out = (
                temp
                / dataset
                / f"q{qkey}"
            )

            out.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                out / "gene_spearman.npy",
                gene_corr,
                allow_pickle=False,
            )

            summary = {
                "status": "PASS",
                "method": "scvi",
                "dataset": dataset,
                "q": expected_q,
                "replicate": replicate,
                "random_seed": seed,
                "threshold": threshold,
                "threshold_source":
                    "frozen thinning validation",
                "target_used_for_fit": False,
                "hyperparameter_reselection_performed":
                    False,
                "n_latent": 10,
                "n_hidden": 128,
                "n_layers": 1,
                "dispersion": "gene",
                "gene_likelihood": "zinb",
                "use_observed_lib_size": True,
                "max_epochs": 400,
                "scvi_version":
                    str(scvi.__version__),
                "torch_version":
                    str(torch.__version__),
                "masked_mse": mse,
                "masked_mae":
                    float(
                        np.mean(
                            np.abs(
                                repaired - target
                            )
                        )
                    ),
                "target_variance":
                    target_var,
                "recovery_index":
                    recovery,
                "candidate_value_recovery_index":
                    candidate_recovery,
                "corrupted_recovery_index":
                    corrupted_recovery,
                "gene_spearman":
                    mean_gene,
                "corrupted_gene_spearman":
                    corrupted_gene,
                "delta_gene_spearman":
                    mean_gene
                    - corrupted_gene,
                "recall":
                    float(recall),
                "precision":
                    float(precision),
                "true_zero_fill":
                    float(fill),
                "tp": tp,
                "fp": fp,
                "selected_entries":
                    int(selected.sum()),
                "n_lost_positive":
                    int(lost.sum()),
                "n_originally_zero":
                    int(
                        original_zero.sum()
                    ),
                "available_genes":
                    int(available.sum()),
                "panel":
                    str(panel_path),
                "panel_sha256":
                    sha(panel_path),
            }

            summary_path = (
                out
                / "summary.json"
            )

            summary_path.write_text(
                json.dumps(
                    summary,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n"
            )

            manifest.append({
                "dataset": dataset,
                "q": expected_q,
                "replicate": replicate,
                "seed": seed,
                "summary":
                    str(summary_path),
                "summary_sha256":
                    sha(summary_path),
                "gene_spearman_sha256":
                    sha(
                        out
                        / "gene_spearman.npy"
                    ),
            })

            print(
                f"EVALUATED dataset={dataset} "
                f"q={qkey} seed={seed} "
                f"rep={replicate}",
                flush=True,
            )

            del (
                model,
                adata,
                decoded,
                prediction,
                score,
                reconstruction,
                y,
            )

            torch.cuda.empty_cache()

        if len(manifest) != 9:
            raise RuntimeError(
                f"expected 9 evaluations, got {len(manifest)}"
            )

        (
            temp
            / "task_manifest.json"
        ).write_text(
            json.dumps(
                {
                    "task_id":
                        args.task_id,
                    "seed":
                        seed,
                    "replicate":
                        replicate,
                    "evaluations":
                        manifest,
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )

        final.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.replace(
            temp,
            final,
        )

    except Exception:
        shutil.rmtree(
            temp,
            ignore_errors=True,
        )
        raise

    print("THINNING_SCVI_CONFIRMATORY_TASK=PASS")
    print(f"TASK_ID={args.task_id}")
    print(f"SEED={seed}")
    print(f"REPLICATE={replicate}")
    print("PANELS=9")
    print("EVALUATIONS=9")


if __name__ == "__main__":
    main()
