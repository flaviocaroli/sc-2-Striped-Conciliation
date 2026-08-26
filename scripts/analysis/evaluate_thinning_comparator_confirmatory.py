#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


METHODS = (
    "positive_train_mean",
    "positive_train_median",
    "truncated_low_rank",
    "knn",
    "alra",
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


def load_old_module(path):
    spec = importlib.util.spec_from_file_location(
        "p2_remaining_frozen",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen P2 evaluator")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_rank_state(y, available):
    idx = np.flatnonzero(available)

    ranks = rankdata(
        np.asarray(y[:, idx], dtype=np.float64),
        axis=0,
        method="average",
    )

    ranks -= ranks.mean(axis=0)

    norm = np.sqrt(
        np.sum(ranks * ranks, axis=0)
    )

    return idx, ranks, norm


def gene_spearman(a, idx, target_ranks, target_norm):
    ranks = rankdata(
        np.asarray(a[:, idx], dtype=np.float64),
        axis=0,
        method="average",
    )

    ranks -= ranks.mean(axis=0)

    norm = np.sqrt(
        np.sum(ranks * ranks, axis=0)
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

    return corr, float(np.nanmean(corr))


def evaluate(
    x,
    y,
    prediction,
    score,
    lost_positive,
    originally_zero,
    available,
    threshold,
    target_state,
):
    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )
    score = np.asarray(
        score,
        dtype=np.float32,
    )

    if prediction.shape != x.shape or score.shape != x.shape:
        raise RuntimeError("prediction/score shape mismatch")

    if not np.isfinite(prediction).all():
        raise RuntimeError("nonfinite prediction")

    if not np.isfinite(score).all():
        raise RuntimeError("nonfinite score")

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

    # Observed values must never be changed.
    observed = (
        (np.abs(x) > ZERO_EPS)
        & available[None, :]
    )

    if not np.array_equal(
        reconstruction[observed],
        x[observed],
    ):
        raise RuntimeError("observed-value preservation failed")

    target = y[lost_positive].astype(np.float64)
    repaired = reconstruction[lost_positive].astype(np.float64)
    candidate = prediction[lost_positive].astype(np.float64)

    if target.size == 0:
        raise RuntimeError("no lost-positive entries")

    target_var = float(np.var(target))

    if target_var <= 0:
        raise RuntimeError("nonpositive target variance")

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

    corrupted = x[lost_positive].astype(np.float64)

    corrupted_mse = float(
        np.mean(
            (corrupted - target) ** 2
        )
    )

    recovery = 1.0 - mse / target_var
    candidate_recovery = 1.0 - candidate_mse / target_var
    corrupted_recovery = 1.0 - corrupted_mse / target_var

    tp = int(
        np.count_nonzero(
            selected & lost_positive
        )
    )

    fp = int(
        np.count_nonzero(
            selected & originally_zero
        )
    )

    recall = tp / int(lost_positive.sum())
    fill = fp / int(originally_zero.sum())

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    idx, target_ranks, target_norm = target_state

    gene_corr, mean_gene = gene_spearman(
        reconstruction,
        idx,
        target_ranks,
        target_norm,
    )

    return {
        "reconstruction": reconstruction,
        "gene_corr": gene_corr,
        "masked_mse": mse,
        "masked_mae": float(
            np.mean(
                np.abs(
                    repaired - target
                )
            )
        ),
        "target_variance": target_var,
        "recovery_index": recovery,
        "candidate_value_recovery_index":
            candidate_recovery,
        "corrupted_recovery_index":
            corrupted_recovery,
        "recall": float(recall),
        "precision": float(precision),
        "true_zero_fill": float(fill),
        "tp": tp,
        "fp": fp,
        "gene_spearman": mean_gene,
        "selected_entries": int(selected.sum()),
    }


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

    if args.task_id not in range(5):
        raise RuntimeError("task outside 0..4")

    replicate = args.task_id + 1

    old_path = Path(
        p["implementation"][
            "old_remaining_evaluator"
        ]["path"]
    )

    if sha(old_path) != p["implementation"][
        "old_remaining_evaluator"
    ]["sha256"]:
        raise RuntimeError("frozen evaluator changed")

    old = load_old_module(old_path)

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
            "THINNING_COMPARATOR_CONFIRMATORY_CHECK=PASS "
            f"TASK={args.task_id} "
            f"REP={replicate} "
            "PANELS=9 EVALUATIONS=45"
        )
        return

    root = (
        Path(p["output_root"])
        / "deterministic"
    )

    final = (
        root
        / f"task_{args.task_id:02d}_rep{replicate:02d}"
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

            # Do NOT load y before all x-only predictors have run.
            with np.load(
                panel_path,
                allow_pickle=False,
            ) as d:
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

            if x.shape != (5000, 4096):
                raise RuntimeError(
                    f"unexpected panel shape {x.shape}"
                )

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

            if np.any(lost & original_zero):
                raise RuntimeError("label overlap")

            predictions = {}

            for method in METHODS:

                alra_ctx = None

                if method in (
                    "positive_train_mean",
                    "positive_train_median",
                ):
                    pred, score, details = old.train_stat_predict(
                        x.shape,
                        method=method,
                        stats_path=Path(
                            p["inputs"]["train_stats"]["path"]
                        ),
                    )

                elif method == "truncated_low_rank":
                    pred, score, details = old.low_rank_predict(
                        x,
                        rank=16,
                    )

                elif method == "knn":
                    pred, score, details = old.knn_predict(
                        x,
                        k=50,
                        chunk_size=64,
                    )

                elif method == "alra":
                    alra_ctx = tempfile.TemporaryDirectory(
                        prefix="thin_confirm_alra_"
                    )

                    pred, metadata, proc, artifacts = old.alra_predict(
                        x,
                        alra_r_script=Path(
                            p["alra"]["r_script"]
                        ),
                        alra_rlib=Path(
                            p["alra"]["r_library"]
                        ),
                        alra_source=Path(
                            p["alra"]["source"]
                        ),
                        rscript=Path(
                            p["alra"]["rscript"]
                        ),
                        work_dir=Path(
                            alra_ctx.name
                        ),
                    )

                    score = pred

                    details = {
                        "chosen_k":
                            int(float(metadata["chosen_k"])),
                        "rank_policy":
                            "ALRA automatic choose_k on corrupted x",
                    }

                else:
                    raise RuntimeError(method)

                predictions[method] = (
                    np.asarray(
                        pred,
                        dtype=np.float32,
                    ).copy(),
                    np.asarray(
                        score,
                        dtype=np.float32,
                    ).copy(),
                    details,
                )

                if alra_ctx is not None:
                    alra_ctx.cleanup()

            # Target opened only after all x-only predictions exist.
            with np.load(
                panel_path,
                allow_pickle=False,
            ) as d:
                y = np.asarray(
                    d["y"],
                    dtype=np.float32,
                ).copy()

            target_state = target_rank_state(
                y,
                available,
            )

            # Corrupted gene Spearman is panel-level and shared.
            idx, target_ranks, target_norm = target_state

            _, corrupted_gene_spearman = gene_spearman(
                x,
                idx,
                target_ranks,
                target_norm,
            )

            for method in METHODS:

                pred, score, details = predictions.pop(method)

                threshold = float(
                    p["thresholds"][
                        method
                    ][qkey]["threshold"]
                )

                metrics = evaluate(
                    x=x,
                    y=y,
                    prediction=pred,
                    score=score,
                    lost_positive=lost,
                    originally_zero=original_zero,
                    available=available,
                    threshold=threshold,
                    target_state=target_state,
                )

                out = (
                    temp
                    / dataset
                    / f"q{qkey}"
                    / method
                )

                out.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                np.save(
                    out / "gene_spearman.npy",
                    metrics["gene_corr"],
                    allow_pickle=False,
                )

                summary = {
                    "status": "PASS",
                    "method": method,
                    "dataset": dataset,
                    "q": expected_q,
                    "replicate": replicate,
                    "threshold": threshold,
                    "threshold_source":
                        "frozen thinning validation",
                    "target_used_for_fit": False,
                    "hyperparameter_reselection_performed":
                        False,
                    "selected_rank":
                        16
                        if method == "truncated_low_rank"
                        else None,
                    "selected_k":
                        50
                        if method == "knn"
                        else None,
                    "masked_mse":
                        metrics["masked_mse"],
                    "masked_mae":
                        metrics["masked_mae"],
                    "target_variance":
                        metrics["target_variance"],
                    "recovery_index":
                        metrics["recovery_index"],
                    "candidate_value_recovery_index":
                        metrics[
                            "candidate_value_recovery_index"
                        ],
                    "corrupted_recovery_index":
                        metrics["corrupted_recovery_index"],
                    "gene_spearman":
                        metrics["gene_spearman"],
                    "corrupted_gene_spearman":
                        corrupted_gene_spearman,
                    "delta_gene_spearman":
                        metrics["gene_spearman"]
                        - corrupted_gene_spearman,
                    "recall":
                        metrics["recall"],
                    "precision":
                        metrics["precision"],
                    "true_zero_fill":
                        metrics["true_zero_fill"],
                    "tp":
                        metrics["tp"],
                    "fp":
                        metrics["fp"],
                    "selected_entries":
                        metrics["selected_entries"],
                    "n_lost_positive":
                        int(lost.sum()),
                    "n_originally_zero":
                        int(original_zero.sum()),
                    "available_genes":
                        int(available.sum()),
                    "method_details":
                        old.json_safe(details),
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
                        old.json_safe(summary),
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    ) + "\n"
                )

                manifest.append({
                    "dataset": dataset,
                    "q": expected_q,
                    "replicate": replicate,
                    "method": method,
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
                    f"q={qkey} rep={replicate} "
                    f"method={method}",
                    flush=True,
                )

        if len(manifest) != 45:
            raise RuntimeError(
                f"expected 45 evaluations, got {len(manifest)}"
            )

        (
            temp
            / "task_manifest.json"
        ).write_text(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "replicate": replicate,
                    "evaluations": manifest,
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

    print("THINNING_COMPARATOR_CONFIRMATORY_TASK=PASS")
    print(f"TASK_ID={args.task_id}")
    print(f"REPLICATE={replicate}")
    print("PANELS=9")
    print("EVALUATIONS=45")


if __name__ == "__main__":
    main()
