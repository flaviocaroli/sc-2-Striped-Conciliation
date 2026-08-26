#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import scvi
from scipy import sparse
import torch


SEEDS = (
    20260728,
    20260729,
    20260730,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def minimal_safe_threshold(scores, budget_fraction=0.02):
    z = np.asarray(scores, dtype=np.float64)
    n = z.size
    assert n > 0

    budget = int(math.floor(budget_fraction * n))

    if budget <= 0:
        return float(np.nextafter(z.max(), np.inf))

    idx = n - budget
    cutoff = float(np.partition(z, idx)[idx])

    if int(np.sum(z >= cutoff)) <= budget:
        threshold = cutoff
    else:
        threshold = float(np.nextafter(cutoff, np.inf))

    assert float(np.mean(z >= threshold)) <= budget_fraction + 1e-15
    return threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--task-id", required=True, type=int)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    protocol_path = Path(args.protocol).resolve()
    p = json.loads(protocol_path.read_text())

    expected = os.environ.get("SC2_THINNING_COMPARATOR_PROTOCOL_SHA")
    if expected and sha(protocol_path) != expected:
        raise RuntimeError("protocol SHA mismatch")

    if args.task_id not in range(15):
        raise RuntimeError("task outside 0..14")

    seed = SEEDS[args.task_id // 5]
    replicate = (args.task_id % 5) + 1

    if str(scvi.__version__) != "1.5.0.post1":
        raise RuntimeError(f"unexpected scvi version {scvi.__version__}")

    panels = []

    for qkey in ("085", "070", "050"):
        key = f"q{qkey}_rep{replicate:02d}"
        info = p["validation_panels"][key]
        path = Path(info["path"])

        if not path.is_file() or sha(path) != info["sha256"]:
            raise RuntimeError(f"panel integrity failure: {path}")

        panels.append((qkey, path))

    if args.check_only:
        print(
            f"THINNING_SCVI_VALIDATION_CHECK=PASS "
            f"TASK={args.task_id} SEED={seed} REP={replicate} EVALUATIONS=3"
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    root = Path(p["output_root"]) / "scvi"
    final = (
        root
        / f"task_{args.task_id:02d}_seed{seed}_rep{replicate:02d}"
    )
    temp = root / f".tmp_task_{args.task_id:02d}_{os.getpid()}"

    if final.exists() or temp.exists():
        raise RuntimeError("output collision")

    temp.mkdir(parents=True)

    manifest = []

    try:
        for qkey, panel_path in panels:

            # Critically: target y is NOT loaded before scVI fitting.
            with np.load(panel_path, allow_pickle=False) as d:
                counts = np.asarray(
                    d["thinned_counts"],
                    dtype=np.uint32,
                    order="C",
                ).copy()
                positive = np.asarray(
                    d["lost_positive_mask"],
                    dtype=bool,
                ).copy()
                true_zero = np.asarray(
                    d["originally_zero_mask"],
                    dtype=bool,
                ).copy()
                available = np.asarray(
                    d["available_gene_mask"],
                    dtype=bool,
                ).copy()

            if counts.shape != (2500, 4096):
                raise RuntimeError(f"unexpected count shape {counts.shape}")

            if np.any(counts[:, ~available] != 0):
                raise RuntimeError("unavailable gene nonzero")

            if np.any(counts.sum(axis=1, dtype=np.uint64) == 0):
                raise RuntimeError("zero-library cell")

            if np.any(positive & true_zero):
                raise RuntimeError("positive/true-zero overlap")

            scvi.settings.seed = int(seed)

            adata = ad.AnnData(
                X=sparse.csr_matrix(counts)
            )

            adata.obs_names = [
                f"cell_{i:05d}"
                for i in range(counts.shape[0])
            ]

            adata.var_names = [
                f"gene_{i:04d}"
                for i in range(counts.shape[1])
            ]

            SCVI = scvi.model.SCVI
            SCVI.setup_anndata(
                adata,
                batch_key=None,
            )

            model = SCVI(
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

            if decoded.shape != counts.shape or \
               not np.isfinite(decoded).all() or \
               np.any(decoded < 0):
                raise RuntimeError("invalid scVI decoded expression")

            prediction = np.log1p(decoded).astype(
                np.float32,
                copy=False,
            )

            score = prediction

            # Target first accessed HERE, after fit + decode.
            with np.load(panel_path, allow_pickle=False) as d:
                y = np.asarray(d["y"], dtype=np.float32).copy()

            target = y[positive].astype(np.float64)
            pred = prediction[positive].astype(np.float64)

            mse = float(np.mean((pred - target) ** 2))
            mae = float(np.mean(np.abs(pred - target)))
            target_var = float(np.var(target, ddof=0))

            if target_var <= 0:
                raise RuntimeError("nonpositive target variance")

            recovery = 1.0 - mse / target_var
            target_sd = float(np.std(target))
            prediction_sd = float(np.std(pred))
            sd_ratio = prediction_sd / target_sd

            pos_scores = np.asarray(
                score[positive],
                dtype=np.float32,
            )
            zero_scores = np.asarray(
                score[true_zero],
                dtype=np.float32,
            )

            threshold = minimal_safe_threshold(zero_scores)
            fill = float(np.mean(zero_scores >= threshold))
            recall = float(np.mean(pos_scores >= threshold))

            tp = int(np.sum(pos_scores >= threshold))
            fp = int(np.sum(zero_scores >= threshold))
            selected = tp + fp
            precision = tp / selected if selected else 0.0

            out = temp / f"q{qkey}"
            out.mkdir(parents=True)

            np.save(
                out / "positive_scores.npy",
                pos_scores,
                allow_pickle=False,
            )

            np.save(
                out / "true_zero_scores.npy",
                zero_scores,
                allow_pickle=False,
            )

            summary = {
                "status": "PASS",
                "method": "scvi",
                "dataset": "internal_validation",
                "q": {"085": 0.85, "070": 0.70, "050": 0.50}[qkey],
                "replicate": replicate,
                "random_seed": seed,
                "target_used_for_fit": False,
                "input": "panel thinned_counts uint32",
                "n_latent": 10,
                "n_hidden": 128,
                "n_layers": 1,
                "dispersion": "gene",
                "gene_likelihood": "zinb",
                "use_observed_lib_size": True,
                "max_epochs": 400,
                "scvi_version": str(scvi.__version__),
                "torch_version": str(torch.__version__),
                "masked_mse": mse,
                "masked_mae": mae,
                "target_variance": target_var,
                "recovery_r": recovery,
                "target_sd": target_sd,
                "prediction_sd": prediction_sd,
                "sd_ratio": sd_ratio,
                "n_positive": int(pos_scores.size),
                "n_true_zero": int(zero_scores.size),
                "run_minimum_safe_threshold_2pct": threshold,
                "run_true_zero_fill_at_minimum_safe_threshold": fill,
                "run_recall_at_minimum_safe_threshold": recall,
                "run_precision_at_minimum_safe_threshold": precision,
                "panel": str(panel_path),
                "panel_sha256": sha(panel_path),
            }

            (out / "summary.json").write_text(
                json.dumps(
                    summary,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n"
            )

            manifest.append({
                "q": summary["q"],
                "replicate": replicate,
                "seed": seed,
                "summary": str(out / "summary.json"),
                "summary_sha256": sha(out / "summary.json"),
                "positive_scores_sha256":
                    sha(out / "positive_scores.npy"),
                "true_zero_scores_sha256":
                    sha(out / "true_zero_scores.npy"),
            })

            print(
                f"VALIDATED method=scvi q={qkey} "
                f"seed={seed} rep={replicate}",
                flush=True,
            )

            del model, adata, decoded, prediction, score
            torch.cuda.empty_cache()

        assert len(manifest) == 3

        (temp / "task_manifest.json").write_text(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "seed": seed,
                    "replicate": replicate,
                    "evaluations": manifest,
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )

        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, final)

    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print("THINNING_SCVI_VALIDATION_TASK=PASS")
    print(f"TASK_ID={args.task_id}")
    print(f"SEED={seed}")
    print(f"REPLICATE={replicate}")
    print("EVALUATIONS=3")


if __name__ == "__main__":
    main()
