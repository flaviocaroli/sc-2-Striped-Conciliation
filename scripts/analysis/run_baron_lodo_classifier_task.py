#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def evaluate_fold(X, labels, donors, heldout, cfg, global_classes):
    train = donors != heldout
    test = donors == heldout

    Xtr = np.asarray(X[train], dtype=np.float32)
    Xte = np.asarray(X[test], dtype=np.float32)
    ytr = labels[train]
    yte = labels[test]

    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    pca = PCA(
        n_components=int(cfg["pca_components"]),
        svd_solver=cfg["pca_solver"],
        random_state=int(cfg["random_state"]),
    )

    Xtr = pca.fit_transform(Xtr)
    Xte = pca.transform(Xte)

    clf = LogisticRegression(
        C=float(cfg["C"]),
        class_weight=cfg["class_weight"],
        solver=cfg["solver"],
        max_iter=int(cfg["max_iter"]),
        random_state=int(cfg["random_state"]),
    )

    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)

    test_classes = sorted(set(yte))

    macro_f1 = f1_score(
        yte,
        pred,
        labels=test_classes,
        average="macro",
        zero_division=0,
    )

    bal = balanced_accuracy_score(yte, pred)

    ll = log_loss(
        yte,
        prob,
        labels=clf.classes_,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            yte,
            pred,
            labels=global_classes,
            zero_division=0,
        )
    )

    per_class = []
    for i, cls in enumerate(global_classes):
        per_class.append({
            "heldout_donor": heldout,
            "cell_type": cls,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    cm = confusion_matrix(
        yte,
        pred,
        labels=global_classes,
    )

    confusion = []
    for i, true_cls in enumerate(global_classes):
        for j, pred_cls in enumerate(global_classes):
            confusion.append({
                "heldout_donor": heldout,
                "true_cell_type": true_cls,
                "pred_cell_type": pred_cls,
                "n": int(cm[i, j]),
            })

    return (
        {
            "heldout_donor": heldout,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "macro_f1": float(macro_f1),
            "balanced_accuracy": float(bal),
            "log_loss": float(ll),
            "pca_explained_variance_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
        },
        per_class,
        confusion,
    )


def evaluate_representation(
    X,
    labels,
    donors,
    cfg,
    global_classes,
):
    heldouts = sorted(set(donors))

    results = joblib.Parallel(
        n_jobs=4,
        prefer="threads",
    )(
        joblib.delayed(evaluate_fold)(
            X,
            labels,
            donors,
            donor,
            cfg,
            global_classes,
        )
        for donor in heldouts
    )

    fold_rows = []
    class_rows = []
    confusion_rows = []

    for fold, classes, confusion in results:
        fold_rows.append(fold)
        class_rows.extend(classes)
        confusion_rows.extend(confusion)

    return fold_rows, class_rows, confusion_rows


def add_metadata(rows, metadata):
    for row in rows:
        row.update(metadata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    protocol_path = Path(args.protocol).resolve()
    p = json.loads(protocol_path.read_text())

    expected = os.environ.get(
        "SC2_BARON_CLASSIFIER_PROTOCOL_SHA"
    )
    if expected and sha(protocol_path) != expected:
        raise RuntimeError("classifier protocol SHA mismatch")

    source_path = Path(
        p["source_protocol"]["path"]
    )

    if sha(source_path) != p["source_protocol"]["sha256"]:
        raise RuntimeError("source protocol changed")

    source = json.loads(source_path.read_text())

    manifest_path = Path(
        p["manifest"]["path"]
    )

    if sha(manifest_path) != p["manifest"]["sha256"]:
        raise RuntimeError("manifest changed")

    manifest = pd.read_csv(manifest_path)

    assert len(manifest) == 5000

    labels = manifest["cell_type"].astype(str).to_numpy()
    donors = manifest["donor"].astype(str).to_numpy()

    assert set(donors) == {
        "human1", "human2", "human3", "human4"
    }

    global_classes = sorted(set(labels))
    assert len(global_classes) == 14

    tasks = {
        int(x["task_id"]): x
        for x in p["tasks"]
    }

    if args.task_id not in tasks:
        raise RuntimeError("unknown task")

    task = tasks[args.task_id]

    output_root = Path(args.output_root).resolve()

    if output_root != Path(p["output_root"]).resolve():
        raise RuntimeError("output root mismatch")

    export_tasks = {
        int(x["task_id"]): x
        for x in source["export_tasks"]
    }

    export_root = Path(source["output_root"])

    # --------------------------------------------------------
    # Resolve representations before touching output.
    # --------------------------------------------------------

    representations = []

    if task["kind"] == "sc2":

        source_task = export_tasks[
            int(task["source_export_task_id"])
        ]

        model_seed = int(source_task["model_seed"])

        export_dir = (
            export_root
            / (
                f"task_{task['source_export_task_id']:02d}_"
                f"model{model_seed}"
            )
        )

        if not export_dir.is_dir():
            raise RuntimeError(
                f"missing SC2 export: {export_dir}"
            )

        for run in source_task["runs"]:

            rep = int(run["replicate"])
            q = float(run["q"])
            qlabel = f"q{int(round(q * 100)):03d}"

            matrix = (
                export_dir
                / f"rep{rep:02d}"
                / qlabel
                / "selective_reconstruction.npy"
            )

            if not matrix.is_file():
                raise RuntimeError(
                    f"missing repaired matrix: {matrix}"
                )

            representations.append({
                "representation": "sc2_repaired",
                "model_seed": model_seed,
                "replicate": rep,
                "q": q,
                "matrix": matrix,
            })

    elif task["kind"] == "baseline":

        # Panels are identical across model seeds, so task 0
        # supplies the unique 15 thinning panels.
        source_task = export_tasks[0]

        # Full-depth target is common to all panels.
        first_panel = Path(
            source_task["runs"][0]["panel"]
        )

        representations.append({
            "representation": "clean",
            "model_seed": None,
            "replicate": 0,
            "q": 1.0,
            "panel": first_panel,
            "panel_key": "y",
        })

        for run in source_task["runs"]:
            representations.append({
                "representation": "corrupted",
                "model_seed": None,
                "replicate": int(run["replicate"]),
                "q": float(run["q"]),
                "panel": Path(run["panel"]),
                "panel_key": "x",
            })

    else:
        raise RuntimeError("invalid task kind")

    expected_n = (
        15 if task["kind"] == "sc2"
        else 16
    )

    assert len(representations) == expected_n

    if args.check_only:
        print(
            f"BARON_LODO_CLASSIFIER_CHECK=PASS "
            f"TASK={args.task_id} "
            f"KIND={task['kind']} "
            f"REPRESENTATIONS={len(representations)}"
        )
        return

    task_dir = (
        output_root
        / f"task_{args.task_id:02d}_{task['kind']}"
    )

    temp = (
        output_root
        / f".tmp_task_{args.task_id:02d}_{os.getpid()}"
    )

    if task_dir.exists() or temp.exists():
        raise RuntimeError("output collision")

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp.mkdir()

    fold_all = []
    class_all = []
    confusion_all = []

    try:
        for i, rep in enumerate(representations, start=1):

            if "matrix" in rep:
                X = np.load(
                    rep["matrix"],
                    mmap_mode="r",
                    allow_pickle=False,
                )
            else:
                with np.load(
                    rep["panel"],
                    allow_pickle=False,
                ) as d:
                    X = np.asarray(
                        d[rep["panel_key"]],
                        dtype=np.float32,
                    )

            if X.shape != (5000, 4096):
                raise RuntimeError(
                    f"bad matrix shape {X.shape}"
                )

            if not np.isfinite(X).all():
                raise RuntimeError(
                    "nonfinite representation"
                )

            folds, classes, confusion = (
                evaluate_representation(
                    X,
                    labels,
                    donors,
                    p["classifier"],
                    global_classes,
                )
            )

            meta = {
                "representation":
                    rep["representation"],

                "model_seed":
                    rep["model_seed"],

                "replicate":
                    rep["replicate"],

                "q":
                    rep["q"],
            }

            add_metadata(folds, meta)
            add_metadata(classes, meta)
            add_metadata(confusion, meta)

            fold_all.extend(folds)
            class_all.extend(classes)
            confusion_all.extend(confusion)

            print(
                f"CLASSIFIED={i}/{len(representations)} "
                f"representation={rep['representation']} "
                f"seed={rep['model_seed']} "
                f"rep={rep['replicate']} "
                f"q={rep['q']}",
                flush=True,
            )

        folds = pd.DataFrame(fold_all)
        classes = pd.DataFrame(class_all)
        confusion = pd.DataFrame(confusion_all)

        assert len(folds) == expected_n * 4

        folds.to_csv(
            temp / "fold_metrics.csv",
            index=False,
        )

        classes.to_csv(
            temp / "per_class_metrics.csv",
            index=False,
        )

        confusion.to_csv(
            temp / "confusion_long.csv",
            index=False,
        )

        summary = (
            folds.groupby(
                [
                    "representation",
                    "model_seed",
                    "replicate",
                    "q",
                ],
                dropna=False,
            )
            .agg(
                donor_macro_f1_mean=(
                    "macro_f1", "mean"
                ),
                donor_macro_f1_min=(
                    "macro_f1", "min"
                ),
                donor_macro_f1_max=(
                    "macro_f1", "max"
                ),
                donor_balanced_accuracy_mean=(
                    "balanced_accuracy", "mean"
                ),
                donor_log_loss_mean=(
                    "log_loss", "mean"
                ),
                n_donors=(
                    "heldout_donor", "nunique"
                ),
            )
            .reset_index()
        )

        summary.to_csv(
            temp / "representation_summary.csv",
            index=False,
        )

        (
            temp / "task_manifest.json"
        ).write_text(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "kind": task["kind"],
                    "representations": expected_n,
                    "folds_per_representation": 4,
                    "classifier_protocol_sha256":
                        sha(protocol_path),
                    "source_protocol_sha256":
                        sha(source_path),
                    "training_only_preprocessing": True,
                    "heldout_donors": [
                        "human1",
                        "human2",
                        "human3",
                        "human4",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        files = sorted(
            [
                x for x in temp.iterdir()
                if x.is_file()
            ],
            key=lambda x: x.name,
        )

        (
            temp / "SHA256SUMS.txt"
        ).write_text(
            "".join(
                f"{sha(x)}  {x.name}\n"
                for x in files
            )
        )

        os.replace(temp, task_dir)

    except Exception:
        shutil.rmtree(
            temp,
            ignore_errors=True,
        )
        raise

    print("BARON_LODO_CLASSIFIER_TASK=PASS")
    print(f"TASK_ID={args.task_id}")
    print(f"KIND={task['kind']}")
    print(f"REPRESENTATIONS={expected_n}")
    print(f"FOLD_EVALUATIONS={expected_n * 4}")


if __name__ == "__main__":
    main()
