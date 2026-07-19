from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader

from sc2.eval.benchmarks.evaluate_masked_reconstruction import (
    MatrixMaskDataset,
    build_model,
    count_parameters,
    dense_matrix,
    get_root,
    load_yaml,
    make_fixed_mask,
    resolve_path,
    safe_corr,
    samplewise_metrics,
    subset_adata_by_manifest,
    subset_genes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate and evaluate the SC2-medium dropout head separately "
            "from its reconstruction-value head."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    return parser.parse_args()


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    ok = np.isfinite(scores)
    labels = labels[ok]
    scores = scores[ok]

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(scores, method="average")
    rank_sum_pos = float(ranks[labels].sum())

    return float(
        (rank_sum_pos - n_pos * (n_pos + 1) / 2.0)
        / float(n_pos * n_neg)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    ok = np.isfinite(scores)
    labels = labels[ok]
    scores = scores[ok]

    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order].astype(np.float64)

    cumulative_tp = np.cumsum(ordered_labels)
    precision = cumulative_tp / np.arange(
        1, ordered_labels.size + 1, dtype=np.float64
    )

    return float(precision[ordered_labels == 1].mean())


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> float:
    labels = np.asarray(labels, dtype=np.float64).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()

    ok = np.isfinite(labels) & np.isfinite(probabilities)
    labels = labels[ok]
    probabilities = np.clip(probabilities[ok], 0.0, 1.0)

    if labels.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(labels.size)
    ece = 0.0

    for index in range(n_bins):
        left = edges[index]
        right = edges[index + 1]

        if index == n_bins - 1:
            in_bin = (
                (probabilities >= left)
                & (probabilities <= right)
            )
        else:
            in_bin = (
                (probabilities >= left)
                & (probabilities < right)
            )

        count = int(in_bin.sum())
        if count == 0:
            continue

        confidence = float(probabilities[in_bin].mean())
        accuracy = float(labels[in_bin].mean())
        ece += (count / total) * abs(confidence - accuracy)

    return float(ece)


def safe_quantile(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(np.quantile(values, quantile))


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    evaluation_mask: np.ndarray,
) -> dict[str, float]:
    predicted_values = np.asarray(prediction)[evaluation_mask]
    target_values = np.asarray(target)[evaluation_mask]

    if predicted_values.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "prediction_mean": float("nan"),
            "prediction_std": float("nan"),
            "target_mean": float("nan"),
            "target_std": float("nan"),
            "std_ratio": float("nan"),
        }

    squared_error = (predicted_values - target_values) ** 2

    target_std = float(np.std(target_values))
    prediction_std = float(np.std(predicted_values))

    return {
        "mse": float(np.mean(squared_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mae": float(
            np.mean(np.abs(predicted_values - target_values))
        ),
        "pearson": safe_corr(
            predicted_values, target_values, "pearson"
        ),
        "spearman": safe_corr(
            predicted_values, target_values, "spearman"
        ),
        "prediction_mean": float(predicted_values.mean()),
        "prediction_std": prediction_std,
        "target_mean": float(target_values.mean()),
        "target_std": target_std,
        "std_ratio": (
            float(prediction_std / target_std)
            if target_std > 0.0
            else float("nan")
        ),
    }


def binary_counts(
    predicted_positive: np.ndarray,
    actual_positive: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float | int]:
    predicted_positive = (
        np.asarray(predicted_positive, dtype=bool) & eligible
    )
    actual_positive = (
        np.asarray(actual_positive, dtype=bool) & eligible
    )
    actual_negative = eligible & ~actual_positive

    tp = int((predicted_positive & actual_positive).sum())
    fp = int((predicted_positive & actual_negative).sum())
    fn = int((~predicted_positive & actual_positive).sum())
    tn = int((~predicted_positive & actual_negative).sum())

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(
        2.0 * precision * recall
        / max(precision + recall, 1e-12)
    )
    false_positive_rate = float(fp / max(fp + tn, 1))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


@torch.inference_mode()
def run_prediction_with_dropout(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()

    reconstruction_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    corrupted_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []

    for batch in loader:
        corrupted = batch["x"].to(device)

        output = model(
            corrupted,
            modality=modality,
            return_dict=True,
        )

        if not isinstance(output, dict):
            raise TypeError(
                "The benchmark requires return_dict=True."
            )

        if "dropout_logits" not in output:
            raise ValueError(
                "The checkpoint does not expose dropout logits. "
                "Use an SC2-medium checkpoint with dropout_head=true."
            )

        reconstruction = output["reconstruction"]
        probabilities = torch.sigmoid(
            output["dropout_logits"]
        )

        reconstruction_chunks.append(
            reconstruction.detach().cpu().numpy()
        )
        probability_chunks.append(
            probabilities.detach().cpu().numpy()
        )
        corrupted_chunks.append(batch["x"].numpy())
        target_chunks.append(batch["target"].numpy())
        mask_chunks.append(batch["mask"].numpy())

    return (
        np.concatenate(
            reconstruction_chunks, axis=0
        ).astype(np.float32),
        np.concatenate(
            probability_chunks, axis=0
        ).astype(np.float32),
        np.concatenate(
            corrupted_chunks, axis=0
        ).astype(np.float32),
        np.concatenate(
            target_chunks, axis=0
        ).astype(np.float32),
        np.concatenate(
            mask_chunks, axis=0
        ).astype(bool),
    )


def dropout_head_metrics(
    probabilities: np.ndarray,
    corrupted: np.ndarray,
    target: np.ndarray,
    synthetic_mask: np.ndarray,
    zero_threshold: float,
) -> dict[str, float | int]:
    observed_zero = corrupted <= zero_threshold

    positive = synthetic_mask & observed_zero
    negative = (
        observed_zero
        & ~positive
        & (target <= zero_threshold)
    )
    eligible = positive | negative

    labels = positive[eligible]
    scores = probabilities[eligible]

    positive_scores = probabilities[positive]
    negative_scores = probabilities[negative]

    return {
        "n_eligible_zeros": int(eligible.sum()),
        "n_positive_synthetic_dropouts": int(
            positive.sum()
        ),
        "n_negative_true_zeros": int(negative.sum()),
        "positive_prevalence": (
            float(labels.mean())
            if labels.size
            else float("nan")
        ),
        "dropout_auroc": binary_auroc(labels, scores),
        "dropout_auprc": average_precision(
            labels, scores
        ),
        "dropout_brier": (
            float(
                np.mean(
                    (
                        scores
                        - labels.astype(np.float32)
                    )
                    ** 2
                )
            )
            if scores.size
            else float("nan")
        ),
        "dropout_ece": expected_calibration_error(
            labels, scores
        ),
        "positive_probability_mean": (
            float(positive_scores.mean())
            if positive_scores.size
            else float("nan")
        ),
        "positive_probability_p05": safe_quantile(
            positive_scores, 0.05
        ),
        "positive_probability_p50": safe_quantile(
            positive_scores, 0.50
        ),
        "positive_probability_p95": safe_quantile(
            positive_scores, 0.95
        ),
        "negative_probability_mean": (
            float(negative_scores.mean())
            if negative_scores.size
            else float("nan")
        ),
        "negative_probability_p05": safe_quantile(
            negative_scores, 0.05
        ),
        "negative_probability_p50": safe_quantile(
            negative_scores, 0.50
        ),
        "negative_probability_p95": safe_quantile(
            negative_scores, 0.95
        ),
    }


def threshold_metrics(
    threshold: float,
    reconstruction: np.ndarray,
    probabilities: np.ndarray,
    corrupted: np.ndarray,
    target: np.ndarray,
    synthetic_mask: np.ndarray,
    zero_threshold: float,
    value_threshold: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    observed_zero = corrupted <= zero_threshold

    positive = synthetic_mask & observed_zero
    negative = (
        observed_zero
        & ~positive
        & (target <= zero_threshold)
    )
    eligible = positive | negative

    selected_by_head = (
        observed_zero
        & (probabilities >= threshold)
    )

    nonnegative_reconstruction = np.maximum(
        reconstruction, 0.0
    )

    effective_fill = (
        selected_by_head
        & (
            nonnegative_reconstruction
            > value_threshold
        )
    )

    detection = binary_counts(
        selected_by_head,
        positive,
        eligible,
    )

    effective = binary_counts(
        effective_fill,
        positive,
        eligible,
    )

    gated = corrupted.copy()
    gated[effective_fill] = (
        nonnegative_reconstruction[effective_fill]
    )

    masked_value = regression_metrics(
        gated,
        target,
        synthetic_mask,
    )

    negative_values = gated[negative]

    row: dict[str, float | int] = {
        "threshold": float(threshold),
        "value_threshold": float(value_threshold),
        "detection_tp": int(detection["tp"]),
        "detection_fp": int(detection["fp"]),
        "detection_fn": int(detection["fn"]),
        "detection_tn": int(detection["tn"]),
        "detection_precision": float(
            detection["precision"]
        ),
        "detection_recall": float(
            detection["recall"]
        ),
        "detection_f1": float(detection["f1"]),
        "detection_false_positive_rate": float(
            detection["false_positive_rate"]
        ),
        "effective_tp": int(effective["tp"]),
        "effective_fp": int(effective["fp"]),
        "effective_fn": int(effective["fn"]),
        "effective_tn": int(effective["tn"]),
        "effective_precision": float(
            effective["precision"]
        ),
        "effective_recall": float(
            effective["recall"]
        ),
        "effective_f1": float(effective["f1"]),
        "effective_true_zero_fill_rate": float(
            effective["false_positive_rate"]
        ),
        "true_zero_imputed_value_mean": (
            float(negative_values.mean())
            if negative_values.size
            else float("nan")
        ),
    }

    row.update(
        {
            f"masked_{key}": value
            for key, value in masked_value.items()
        }
    )

    return row, gated


def choose_global_threshold(
    threshold_df: pd.DataFrame,
    max_true_zero_fill_rate: float,
) -> tuple[float, pd.DataFrame, bool]:
    aggregate = (
        threshold_df.groupby(
            "threshold", as_index=False
        )
        .agg(
            mean_detection_f1=(
                "detection_f1", "mean"
            ),
            mean_detection_recall=(
                "detection_recall", "mean"
            ),
            mean_effective_f1=(
                "effective_f1", "mean"
            ),
            mean_effective_recall=(
                "effective_recall", "mean"
            ),
            mean_true_zero_fill_rate=(
                "effective_true_zero_fill_rate",
                "mean",
            ),
            mean_masked_mse=(
                "masked_mse", "mean"
            ),
            mean_masked_spearman=(
                "masked_spearman", "mean"
            ),
        )
        .sort_values("threshold")
        .reset_index(drop=True)
    )

    feasible = aggregate[
        aggregate["mean_true_zero_fill_rate"]
        <= float(max_true_zero_fill_rate)
    ].copy()

    constraint_satisfied = not feasible.empty
    candidates = (
        feasible
        if constraint_satisfied
        else aggregate.copy()
    )

    candidates = candidates.sort_values(
        [
            "mean_effective_f1",
            "mean_detection_f1",
            "mean_masked_spearman",
            "mean_masked_mse",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            False,
        ],
    )

    selected = float(
        candidates.iloc[0]["threshold"]
    )

    aggregate["selected"] = np.isclose(
        aggregate["threshold"],
        selected,
    )

    return (
        selected,
        aggregate,
        constraint_satisfied,
    )


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    paths_cfg = load_yaml(args.paths)

    data_root = get_root(
        paths_cfg,
        "data_root",
        "SC2_DATA_ROOT",
    )
    output_root = get_root(
        paths_cfg,
        "output_root",
        "SC2_OUTPUT_ROOT",
    )

    eval_name = str(cfg["eval_name"])
    seed = int(cfg.get("seed", 42))

    device_cfg = str(
        cfg.get("device", "auto")
    ).lower()

    if device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    out_dir = (
        output_root
        / "evals"
        / eval_name
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        out_dir / "resolved_config.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    benchmark_cfg = cfg["benchmark"]

    h5ad_path = resolve_path(
        data_root,
        data_cfg["h5ad_path"],
    )
    split_manifest_path = resolve_path(
        data_root,
        data_cfg.get(
            "split_manifest_path"
        ),
    )
    shared_gene_table_path = resolve_path(
        data_root,
        data_cfg.get(
            "shared_gene_table_path"
        ),
    )
    checkpoint_path = resolve_path(
        output_root,
        eval_cfg["checkpoint_path"],
    )

    if (
        h5ad_path is None
        or checkpoint_path is None
    ):
        raise ValueError(
            "Both data.h5ad_path and "
            "eval.checkpoint_path are required."
        )

    print(f"device={device}")
    print(f"eval_name={eval_name}")
    print(f"h5ad_path={h5ad_path}")
    print(f"checkpoint_path={checkpoint_path}")

    adata = ad.read_h5ad(h5ad_path)

    adata = subset_adata_by_manifest(
        adata=adata,
        manifest_path=split_manifest_path,
        split=data_cfg.get("split"),
        split_col=str(
            data_cfg.get("split_col", "split")
        ),
    )

    adata = subset_genes(
        adata=adata,
        shared_gene_table_path=(
            shared_gene_table_path
        ),
        n_genes=int(
            data_cfg.get(
                "n_genes",
                adata.n_vars,
            )
        ),
    )

    matrix = dense_matrix(adata)

    if bool(
        data_cfg.get("log1p_input", False)
    ):
        matrix = np.log1p(
            np.maximum(matrix, 0.0)
        ).astype(np.float32)

    n_genes = int(matrix.shape[1])

    print(
        f"benchmark_matrix_shape={matrix.shape}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    checkpoint_cfg = checkpoint.get(
        "config", {}
    )

    model_cfg = checkpoint_cfg.get(
        "model",
        cfg["model"],
    )

    checkpoint_n_genes = int(
        checkpoint_cfg.get(
            "data", {}
        ).get(
            "n_genes",
            n_genes,
        )
    )

    if checkpoint_n_genes != n_genes:
        raise ValueError(
            f"Checkpoint expects "
            f"{checkpoint_n_genes} genes, "
            f"benchmark has {n_genes}."
        )

    model_kind, model = build_model(
        model_cfg,
        n_genes=checkpoint_n_genes,
    )

    model = model.to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    parameters = count_parameters(model)

    print(f"model_kind={model_kind}")
    print(
        f"parameters_total="
        f"{parameters['total']}"
    )
    print(
        f"parameters_trainable="
        f"{parameters['trainable']}"
    )

    mode = str(
        benchmark_cfg.get(
            "mode", "calibrate"
        )
    ).lower()

    if mode not in {
        "calibrate",
        "evaluate",
    }:
        raise ValueError(
            "benchmark.mode must be "
            "'calibrate' or 'evaluate'."
        )

    zero_threshold = float(
        benchmark_cfg.get(
            "zero_threshold",
            1e-8,
        )
    )

    value_threshold = float(
        benchmark_cfg.get(
            "value_threshold",
            0.1,
        )
    )

    max_true_zero_fill_rate = float(
        benchmark_cfg.get(
            "max_true_zero_fill_rate",
            0.01,
        )
    )

    mask_probs = [
        float(value)
        for value in benchmark_cfg[
            "mask_probs"
        ]
    ]

    nonzero_only = bool(
        benchmark_cfg.get(
            "nonzero_only", True
        )
    )

    modality = str(
        benchmark_cfg.get(
            "modality", "sc"
        )
    )

    if mode == "calibrate":
        thresholds = [
            float(value)
            for value in benchmark_cfg[
                "thresholds"
            ]
        ]
        selected_threshold: float | None = (
            None
        )
    else:
        threshold_path = resolve_path(
            output_root,
            benchmark_cfg.get(
                "selected_threshold_path"
            ),
        )

        if (
            threshold_path is None
            or not threshold_path.exists()
        ):
            raise FileNotFoundError(
                "Evaluation mode requires "
                "benchmark.selected_threshold_path "
                "from the validation run."
            )

        selected_payload = json.loads(
            threshold_path.read_text()
        )

        selected_threshold = float(
            selected_payload[
                "selected_threshold"
            ]
        )

        thresholds = [selected_threshold]

        print(
            f"selected_threshold="
            f"{selected_threshold}"
        )

    batch_size = int(
        data_cfg.get("batch_size", 64)
    )
    num_workers = int(
        data_cfg.get("num_workers", 4)
    )

    raw_rows: list[
        dict[str, float | int]
    ] = []
    dropout_rows: list[
        dict[str, float | int]
    ] = []
    threshold_rows: list[
        dict[str, float | int]
    ] = []
    selected_rows: list[
        dict[str, float | int]
    ] = []

    for mask_prob in mask_probs:
        print(
            f"running_mask_prob={mask_prob}"
        )

        mask = make_fixed_mask(
            matrix=matrix,
            mask_prob=mask_prob,
            seed=(
                seed
                + int(
                    round(mask_prob * 1000)
                )
            ),
            nonzero_only=nonzero_only,
        )

        dataset = MatrixMaskDataset(
            x=matrix,
            mask=mask,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(
                torch.cuda.is_available()
            ),
        )

        (
            reconstruction,
            probabilities,
            corrupted,
            target,
            used_mask,
        ) = run_prediction_with_dropout(
            model=model,
            loader=loader,
            device=device,
            modality=modality,
        )

        raw_value = regression_metrics(
            reconstruction,
            target,
            used_mask,
        )

        zero_baseline = regression_metrics(
            np.zeros_like(target),
            target,
            used_mask,
        )

        raw_sample = samplewise_metrics(
            reconstruction,
            target,
            used_mask,
        )

        true_zero = (
            target <= zero_threshold
        )

        raw_true_zero_fill = (
            np.maximum(
                reconstruction, 0.0
            )
            > value_threshold
        ) & true_zero

        raw_row: dict[
            str, float | int
        ] = {
            "mask_prob": mask_prob,
            "n_cells": int(
                target.shape[0]
            ),
            "n_genes": int(
                target.shape[1]
            ),
            "n_masked_entries": int(
                used_mask.sum()
            ),
            "raw_sample_pearson_mean": (
                float(
                    raw_sample[
                        "pearson"
                    ].mean()
                )
            ),
            "raw_sample_spearman_mean": (
                float(
                    raw_sample[
                        "spearman"
                    ].mean()
                )
            ),
            "raw_true_zero_fill_rate": (
                float(
                    raw_true_zero_fill[
                        true_zero
                    ].mean()
                )
            ),
            "zero_baseline_mse": float(
                zero_baseline["mse"]
            ),
            "zero_baseline_mae": float(
                zero_baseline["mae"]
            ),
        }

        raw_row.update(
            {
                f"raw_masked_{key}": value
                for key, value
                in raw_value.items()
            }
        )

        raw_rows.append(raw_row)

        dropout_row = {
            "mask_prob": mask_prob
        }

        dropout_row.update(
            dropout_head_metrics(
                probabilities=probabilities,
                corrupted=corrupted,
                target=target,
                synthetic_mask=used_mask,
                zero_threshold=(
                    zero_threshold
                ),
            )
        )

        dropout_rows.append(dropout_row)

        for threshold in thresholds:
            row, gated = threshold_metrics(
                threshold=threshold,
                reconstruction=(
                    reconstruction
                ),
                probabilities=(
                    probabilities
                ),
                corrupted=corrupted,
                target=target,
                synthetic_mask=used_mask,
                zero_threshold=(
                    zero_threshold
                ),
                value_threshold=(
                    value_threshold
                ),
            )

            row["mask_prob"] = mask_prob
            threshold_rows.append(row)

            if (
                selected_threshold
                is not None
            ):
                gated_sample = (
                    samplewise_metrics(
                        gated,
                        target,
                        used_mask,
                    )
                )

                selected_row = dict(row)

                selected_row[
                    "gated_sample_pearson_mean"
                ] = float(
                    gated_sample[
                        "pearson"
                    ].mean()
                )

                selected_row[
                    "gated_sample_spearman_mean"
                ] = float(
                    gated_sample[
                        "spearman"
                    ].mean()
                )

                selected_rows.append(
                    selected_row
                )

    raw_df = pd.DataFrame(raw_rows)
    dropout_df = pd.DataFrame(
        dropout_rows
    )
    threshold_df = pd.DataFrame(
        threshold_rows
    )

    raw_df.to_csv(
        out_dir
        / "raw_reconstruction_diagnostics.csv",
        index=False,
    )

    dropout_df.to_csv(
        out_dir
        / "dropout_head_diagnostics.csv",
        index=False,
    )

    threshold_df.to_csv(
        out_dir / "threshold_sweep.csv",
        index=False,
    )

    summary: dict[str, Any] = {
        "mode": mode,
        "model_kind": model_kind,
        "parameter_counts": parameters,
        "checkpoint_path": str(
            checkpoint_path
        ),
        "raw_reconstruction": raw_rows,
        "dropout_head": dropout_rows,
    }

    if mode == "calibrate":
        (
            selected_threshold,
            aggregate_df,
            constraint_satisfied,
        ) = choose_global_threshold(
            threshold_df=threshold_df,
            max_true_zero_fill_rate=(
                max_true_zero_fill_rate
            ),
        )

        aggregate_df.to_csv(
            out_dir
            / "threshold_aggregate.csv",
            index=False,
        )

        selected_payload = {
            "selected_threshold": (
                selected_threshold
            ),
            "selection_split": str(
                data_cfg.get("split")
            ),
            "selection_rule": (
                "maximize mean effective F1 "
                "across mask probabilities "
                "subject to mean effective "
                "true-zero fill rate <= "
                f"{max_true_zero_fill_rate}"
            ),
            "constraint_satisfied": (
                constraint_satisfied
            ),
            "max_true_zero_fill_rate": (
                max_true_zero_fill_rate
            ),
            "value_threshold": (
                value_threshold
            ),
            "mask_probs": mask_probs,
        }

        with (
            out_dir
            / "selected_threshold.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                selected_payload,
                handle,
                indent=2,
            )

        summary["selected_threshold"] = (
            selected_payload
        )

        print(
            f"selected_threshold="
            f"{selected_threshold}"
        )
        print(
            f"constraint_satisfied="
            f"{constraint_satisfied}"
        )
        print(
            aggregate_df.to_string(
                index=False
            )
        )

    else:
        selected_df = pd.DataFrame(
            selected_rows
        )

        selected_df.to_csv(
            out_dir
            / "selected_threshold_metrics.csv",
            index=False,
        )

        summary["selected_threshold"] = (
            selected_threshold
        )
        summary[
            "selected_threshold_metrics"
        ] = selected_rows

        print(
            "selected threshold test metrics:"
        )
        print(
            selected_df.to_string(
                index=False
            )
        )

    with (
        out_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "saved evaluation outputs to:"
    )
    print(out_dir)

    print(
        "raw reconstruction diagnostics:"
    )
    print(
        raw_df.to_string(index=False)
    )

    print("dropout head diagnostics:")
    print(
        dropout_df.to_string(index=False)
    )


if __name__ == "__main__":
    main()
