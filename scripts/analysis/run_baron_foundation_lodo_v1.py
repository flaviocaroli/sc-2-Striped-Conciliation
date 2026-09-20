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
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def fold_eval(
    X,
    labels,
    donors,
    heldout,
    cfg,
    global_classes,
):
    train = donors != heldout
    test = donors == heldout

    Xtr = np.asarray(
        X[train],
        dtype=np.float32,
    )

    Xte = np.asarray(
        X[test],
        dtype=np.float32,
    )

    ytr = labels[train]
    yte = labels[test]

    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    pca = PCA(
        n_components=int(
            cfg["pca_components"]
        ),
        svd_solver=cfg["pca_solver"],
        random_state=int(
            cfg["random_state"]
        ),
    )

    Xtr = pca.fit_transform(Xtr)
    Xte = pca.transform(Xte)

    clf = LogisticRegression(
        C=float(cfg["C"]),
        class_weight=cfg["class_weight"],
        solver=cfg["solver"],
        max_iter=int(cfg["max_iter"]),
        random_state=int(
            cfg["random_state"]
        ),
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

    bal = balanced_accuracy_score(
        yte,
        pred,
    )

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

    for i, true_cls in enumerate(
        global_classes
    ):
        for j, pred_cls in enumerate(
            global_classes
        ):
            confusion.append({
                "heldout_donor": heldout,
                "true_cell_type": true_cls,
                "pred_cell_type": pred_cls,
                "n": int(cm[i, j]),
            })

    fold = {
        "heldout_donor": heldout,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal),
        "log_loss": float(ll),
        "pca_explained_variance_sum":
            float(
                pca.explained_variance_ratio_.sum()
            ),
    }

    return fold, per_class, confusion


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--protocol",
        required=True,
    )

    ap.add_argument(
        "--task-id",
        required=True,
        type=int,
    )

    ap.add_argument(
        "--output-root",
        required=True,
    )

    args = ap.parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    p = json.loads(
        protocol_path.read_text()
    )

    expected = os.environ.get(
        "SC2_FOUNDATION_REP_PROTOCOL_SHA"
    )

    if expected and sha(protocol_path) != expected:
        raise RuntimeError(
            "protocol SHA mismatch"
        )

    tasks = {
        int(x["task_id"]): x
        for x in p["lodo_tasks"]
    }

    if args.task_id not in tasks:
        raise RuntimeError(
            "invalid LODO task"
        )

    task = tasks[args.task_id]
    model = task["model"]

    model_cfg = p["models"][model]

    raw = Path(
        p["input_bundle"]["path"]
    )

    if sha(raw) != \
            p["input_bundle"]["sha256"]:
        raise RuntimeError(
            "raw bundle SHA changed"
        )

    embed_dir = Path(
        model_cfg["output_dir"]
    )

    embedding_path = (
        embed_dir / "embedding.npz"
    )

    receipt_path = (
        embed_dir / "receipt.json"
    )

    for x in (
        embedding_path,
        receipt_path,
        embed_dir / "SHA256SUMS.txt",
    ):
        if not x.is_file():
            raise RuntimeError(
                f"missing embedding output {x}"
            )

    (
        os.chdir(embed_dir)
    )

    import subprocess

    subprocess.run(
        [
            "sha256sum",
            "-c",
            "SHA256SUMS.txt",
        ],
        check=True,
    )

    receipt = json.loads(
        receipt_path.read_text()
    )

    assert receipt["status"] == "PASS"
    assert receipt["model"] == model
    assert receipt["dataset"] == "baron_pancreas"
    assert receipt["cells"] == 5000
    assert receipt["input_bundle_sha256"] == \
        p["input_bundle"]["sha256"]

    with np.load(
        raw,
        allow_pickle=False,
    ) as d:

        raw_rows = np.asarray(
            d["source_row"],
            dtype=np.int64,
        )

        raw_cell = np.asarray(
            d["cell_id"]
        ).astype(str)

        labels = np.asarray(
            d["cell_type"]
        ).astype(str)

        donors = np.asarray(
            d["donor"]
        ).astype(str)

    with np.load(
        embedding_path,
        allow_pickle=False,
    ) as d:

        X = np.asarray(
            d["embedding"],
            dtype=np.float32,
        )

        rows = np.asarray(
            d["source_row"],
            dtype=np.int64,
        )

        cells = np.asarray(
            d["cell_id"]
        ).astype(str)

        emb_donor = np.asarray(
            d["donor"]
        ).astype(str)

        emb_type = np.asarray(
            d["cell_type"]
        ).astype(str)

    expected_dim = int(
        model_cfg["embedding_dim"]
    )

    if X.shape != (
        5000,
        expected_dim,
    ):
        raise RuntimeError(
            f"bad embedding shape {X.shape}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "nonfinite embeddings"
        )

    if not np.array_equal(
        rows,
        raw_rows,
    ):
        raise RuntimeError(
            "source_row alignment mismatch"
        )

    if not np.array_equal(
        cells,
        raw_cell,
    ):
        raise RuntimeError(
            "cell_id alignment mismatch"
        )

    if not np.array_equal(
        emb_donor,
        donors,
    ):
        raise RuntimeError(
            "donor alignment mismatch"
        )

    if not np.array_equal(
        emb_type,
        labels,
    ):
        raise RuntimeError(
            "cell_type alignment mismatch"
        )

    assert set(donors) == {
        "human1",
        "human2",
        "human3",
        "human4",
    }

    global_classes = sorted(
        set(labels)
    )

    assert len(global_classes) == 14

    heldouts = [
        "human1",
        "human2",
        "human3",
        "human4",
    ]

    results = joblib.Parallel(
        n_jobs=4,
        prefer="threads",
    )(
        joblib.delayed(fold_eval)(
            X,
            labels,
            donors,
            donor,
            p["classifier"],
            global_classes,
        )
        for donor in heldouts
    )

    fold_rows = []
    class_rows = []
    confusion_rows = []

    for fold, classes, confusion in results:
        fold["model"] = model

        for x in classes:
            x["model"] = model

        for x in confusion:
            x["model"] = model

        fold_rows.append(fold)
        class_rows.extend(classes)
        confusion_rows.extend(confusion)

    assert len(fold_rows) == 4

    output_root = Path(
        args.output_root
    ).resolve()

    if output_root != Path(
        p["lodo_output_root"]
    ).resolve():
        raise RuntimeError(
            "LODO output root mismatch"
        )

    final = (
        output_root
        / f"task_{args.task_id:02d}_{model}"
    )

    temp = (
        output_root
        / f".tmp_{args.task_id:02d}_{os.getpid()}"
    )

    if final.exists() or temp.exists():
        raise RuntimeError(
            "LODO output collision"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp.mkdir()

    try:

        folds = pd.DataFrame(fold_rows)
        classes = pd.DataFrame(class_rows)
        confusion = pd.DataFrame(
            confusion_rows
        )

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

        summary = {
            "status": "PASS",
            "model": model,
            "representation":
                "foundation_embedding",
            "cells": 5000,
            "embedding_dim":
                expected_dim,
            "heldout_donors":
                heldouts,
            "macro_f1_mean":
                float(
                    folds[
                        "macro_f1"
                    ].mean()
                ),
            "macro_f1_min":
                float(
                    folds[
                        "macro_f1"
                    ].min()
                ),
            "balanced_accuracy_mean":
                float(
                    folds[
                        "balanced_accuracy"
                    ].mean()
                ),
            "log_loss_mean":
                float(
                    folds[
                        "log_loss"
                    ].mean()
                ),
            "training_only_preprocessing":
                True,
            "pca_components":
                int(
                    p["classifier"][
                        "pca_components"
                    ]
                ),
            "protocol_sha256":
                sha(protocol_path),
            "embedding_sha256":
                sha(embedding_path),
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
            x for x in temp.iterdir()
            if x.is_file()
        )

        (
            temp / "SHA256SUMS.txt"
        ).write_text(
            "".join(
                f"{sha(x)}  {x.name}\n"
                for x in files
            )
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

    print(
        json.dumps(
            summary,
            sort_keys=True,
        )
    )

    print(
        "P3_FOUNDATION_BARON_LODO=PASS"
    )


if __name__ == "__main__":
    main()
