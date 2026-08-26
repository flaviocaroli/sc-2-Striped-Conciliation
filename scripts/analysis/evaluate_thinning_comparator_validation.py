#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


METHODS = (
    "positive_train_mean",
    "positive_train_median",
    "truncated_low_rank",
    "knn",
    "alra",
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

    # budget-th largest value.
    idx = n - budget
    cutoff = float(np.partition(z, idx)[idx])

    if int(np.sum(z >= cutoff)) <= budget:
        threshold = cutoff
    else:
        threshold = float(np.nextafter(cutoff, np.inf))

    assert float(np.mean(z >= threshold)) <= budget_fraction + 1e-15
    return threshold


def load_old_module(path):
    spec = importlib.util.spec_from_file_location(
        "p2_remaining_frozen",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen comparator implementation")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    if args.task_id not in range(5):
        raise RuntimeError("task outside 0..4")

    replicate = args.task_id + 1

    old = load_old_module(
        p["implementation"]["old_remaining_evaluator"]["path"]
    )

    # Exact frozen implementation identity.
    if sha(p["implementation"]["old_remaining_evaluator"]["path"]) != \
       p["implementation"]["old_remaining_evaluator"]["sha256"]:
        raise RuntimeError("old evaluator changed")

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
            f"THINNING_DETERMINISTIC_VALIDATION_CHECK=PASS "
            f"TASK={args.task_id} REP={replicate} EVALUATIONS=15"
        )
        return

    root = Path(p["output_root"]) / "deterministic"
    final = root / f"task_{args.task_id:02d}_rep{replicate:02d}"
    temp = root / f".tmp_task_{args.task_id:02d}_{os.getpid()}"

    if final.exists() or temp.exists():
        raise RuntimeError("output collision")

    temp.mkdir(parents=True)

    manifest = []

    try:
        for qkey, panel_path in panels:

            with np.load(panel_path, allow_pickle=False) as d:
                x = np.asarray(d["x"], dtype=np.float32).copy()
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

            if x.shape != (2500, 4096):
                raise RuntimeError(f"unexpected validation shape {x.shape}")

            if positive.shape != x.shape or true_zero.shape != x.shape:
                raise RuntimeError("mask shape mismatch")

            if np.any(positive & true_zero):
                raise RuntimeError("positive/true-zero overlap")

            if np.any(positive & ~available[None, :]):
                raise RuntimeError("positive unavailable gene")

            for method in METHODS:

                alra_context = None

                if method in (
                    "positive_train_mean",
                    "positive_train_median",
                ):
                    prediction, score, details = old.train_stat_predict(
                        x.shape,
                        method=method,
                        stats_path=Path(p["inputs"]["train_stats"]["path"]),
                    )

                elif method == "truncated_low_rank":
                    prediction, score, details = old.low_rank_predict(
                        x,
                        rank=16,
                    )

                elif method == "knn":
                    prediction, score, details = old.knn_predict(
                        x,
                        k=50,
                        chunk_size=64,
                    )

                elif method == "alra":
                    alra_context = tempfile.TemporaryDirectory(
                        prefix="thin_val_alra_"
                    )

                    prediction, metadata, proc, artifacts = old.alra_predict(
                        x,
                        alra_r_script=Path(p["alra"]["r_script"]),
                        alra_rlib=Path(p["alra"]["r_library"]),
                        alra_source=Path(p["alra"]["source"]),
                        rscript=Path(p["alra"]["rscript"]),
                        work_dir=Path(alra_context.name),
                    )

                    score = prediction

                    details = {
                        "chosen_k": int(float(metadata["chosen_k"])),
                        "source_commit":
                            "f34d46570b1221179047d9f99c235ed880cc3bae",
                        "rank_policy": "ALRA automatic choose_k on corrupted x",
                    }

                else:
                    raise RuntimeError(method)

                try:
                    prediction = np.asarray(prediction, dtype=np.float32)
                    score = np.asarray(score, dtype=np.float32)

                    if prediction.shape != x.shape or score.shape != x.shape:
                        raise RuntimeError("prediction shape mismatch")

                    if not np.isfinite(prediction).all() or \
                       not np.isfinite(score).all():
                        raise RuntimeError("nonfinite comparator output")

                    # Target first accessed only after x-only fitting/prediction.
                    with np.load(panel_path, allow_pickle=False) as d:
                        y = np.asarray(d["y"], dtype=np.float32).copy()

                    target = y[positive].astype(np.float64)
                    pred = prediction[positive].astype(np.float64)

                    if target.size == 0:
                        raise RuntimeError("no lost-positive targets")

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

                    selected = int(np.sum(pos_scores >= threshold)) + \
                               int(np.sum(zero_scores >= threshold))
                    tp = int(np.sum(pos_scores >= threshold))
                    precision = tp / selected if selected else 0.0

                    disc = old.score_discrimination(
                        score,
                        positive,
                        true_zero,
                    )

                    out = temp / f"q{qkey}" / method
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
                        "method": method,
                        "dataset": "internal_validation",
                        "q": {"085": 0.85, "070": 0.70, "050": 0.50}[qkey],
                        "replicate": replicate,
                        "target_used_for_fit": False,
                        "hyperparameter_reselection_performed": False,
                        "selected_rank": 16
                            if method == "truncated_low_rank" else None,
                        "selected_k": 50
                            if method == "knn" else None,
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
                        "method_details": details,
                        "score_discrimination": old.json_safe(disc),
                    }

                    (out / "summary.json").write_text(
                        json.dumps(
                            old.json_safe(summary),
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        ) + "\n"
                    )

                    manifest.append({
                        "q": summary["q"],
                        "replicate": replicate,
                        "method": method,
                        "summary": str(out / "summary.json"),
                        "summary_sha256": sha(out / "summary.json"),
                        "positive_scores_sha256":
                            sha(out / "positive_scores.npy"),
                        "true_zero_scores_sha256":
                            sha(out / "true_zero_scores.npy"),
                    })

                finally:
                    if alra_context is not None:
                        alra_context.cleanup()

                print(
                    f"VALIDATED method={method} q={qkey} rep={replicate}",
                    flush=True,
                )

        assert len(manifest) == 15

        (temp / "task_manifest.json").write_text(
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

        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, final)

    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print("THINNING_DETERMINISTIC_VALIDATION_TASK=PASS")
    print(f"TASK_ID={args.task_id}")
    print(f"REPLICATE={replicate}")
    print("EVALUATIONS=15")


if __name__ == "__main__":
    main()
