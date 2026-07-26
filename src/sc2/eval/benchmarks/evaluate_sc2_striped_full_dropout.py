from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from sc2.config import load_yaml, merge_train_and_paths
from sc2.eval.benchmarks.evaluate_masked_reconstruction import (
    MatrixMaskDataset,
    dense_matrix,
    make_fixed_mask,
    subset_adata_by_manifest,
    subset_genes,
)
from sc2.models.striped.sc2_striped_full import build_sc2_striped_full_from_config
from sc2.train import train_sc2_mamba_bridge as bridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dropout calibration for SC2 20+4.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool).ravel()
    scores = scores.astype(np.float64).ravel()
    n_positive = int(labels.sum())
    n_negative = int((~labels).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[labels].sum())
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2.0)
        / float(n_positive * n_negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool).ravel()
    scores = scores.astype(np.float64).ravel()
    n_positive = int(labels.sum())
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order].astype(np.float64)
    precision = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    return float(precision[ordered == 1].mean())


def binary_metrics(predicted: np.ndarray, positive: np.ndarray, eligible: np.ndarray) -> dict[str, float | int]:
    predicted = predicted.astype(bool) & eligible
    positive = positive.astype(bool) & eligible
    negative = eligible & ~positive
    tp = int((predicted & positive).sum())
    fp = int((predicted & negative).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & negative).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1.0e-12))
    fpr = float(fp / max(fp + tn, 1))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


@torch.inference_mode()
def predict_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability_chunks: list[np.ndarray] = []
    positive_chunks: list[np.ndarray] = []
    expected_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        observed = bridge.move_tensor(batch["x"], device)
        with bridge.autocast_context(device, amp_enabled, amp_dtype):
            output = model(observed, modality="sc", return_dict=True)
            if not isinstance(output, dict):
                raise TypeError("Expected dictionary model output")
        probability_chunks.append(output["dropout_probability"].float().cpu().numpy())
        positive_chunks.append(output["positive_value"].float().cpu().numpy())
        expected_chunks.append(output["expected_repair"].float().cpu().numpy())
        target_chunks.append(batch["target"].numpy())
        mask_chunks.append(batch["mask"].numpy())
    return (
        np.concatenate(probability_chunks, axis=0),
        np.concatenate(positive_chunks, axis=0),
        np.concatenate(expected_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
        np.concatenate(mask_chunks, axis=0).astype(bool),
    )


def main() -> None:
    args = parse_args()
    eval_cfg = load_yaml(args.config)
    paths_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(eval_cfg, paths_cfg)
    seed = int(cfg.get("seed", 42))
    bridge.seed_everything(seed)
    device = bridge.get_device(cfg.get("device", "auto"))

    data_root = Path(cfg["paths"]["data_root"])
    output_root = Path(cfg["paths"]["output_root"])
    data_cfg = cfg["data"]
    benchmark_cfg = cfg["benchmark"]
    mode = str(benchmark_cfg.get("mode", "calibrate")).lower()
    if mode not in {"calibrate", "evaluate"}:
        raise ValueError("benchmark.mode must be calibrate or evaluate")

    h5ad_path = resolve_path(data_root, data_cfg["h5ad_path"])
    manifest_path = resolve_path(data_root, data_cfg.get("split_manifest_path"))
    gene_table_path = resolve_path(data_root, data_cfg.get("shared_gene_table_path"))
    checkpoint_path = resolve_path(output_root, cfg["eval"]["checkpoint_path"])
    assert h5ad_path is not None and checkpoint_path is not None

    adata = ad.read_h5ad(h5ad_path)
    adata = subset_adata_by_manifest(
        adata=adata,
        manifest_path=manifest_path,
        split=data_cfg.get("split"),
        split_col=str(data_cfg.get("split_col", "split")),
    )
    adata = subset_genes(
        adata=adata,
        shared_gene_table_path=gene_table_path,
        n_genes=int(data_cfg.get("n_genes", adata.n_vars)),
    )
    matrix = dense_matrix(adata)
    if bool(data_cfg.get("log1p_input", False)):
        matrix = np.log1p(np.maximum(matrix, 0.0)).astype(np.float32)

    checkpoint_data = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_cfg = checkpoint_data.get("config")
    if not isinstance(checkpoint_cfg, dict) or "model" not in checkpoint_cfg:
        raise ValueError("Checkpoint must contain model config")
    model = build_sc2_striped_full_from_config(checkpoint_cfg["model"], n_genes=matrix.shape[1]).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(False)

    amp_enabled = bridge.resolve_amp(cfg.get("eval", {}), device)
    amp_dtype = bridge.resolve_amp_dtype(cfg.get("eval", {}), device) if amp_enabled else None
    value_threshold = float(benchmark_cfg.get("value_threshold", 0.10))
    max_fill = float(benchmark_cfg.get("max_true_zero_fill_rate", 0.01))
    thresholds = [float(value) for value in benchmark_cfg.get("thresholds", [])]
    if mode == "evaluate":
        selected_path = resolve_path(output_root, benchmark_cfg.get("selected_threshold_path"))
        if selected_path is None or not selected_path.exists():
            raise FileNotFoundError("selected_threshold_path from validation is required")
        selected_payload = json.loads(selected_path.read_text())
        thresholds = [float(selected_payload["selected_threshold"])]

    out_dir = output_root / "evals" / str(cfg["eval_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_rows: list[dict[str, float | int]] = []
    sweep_rows: list[dict[str, float | int]] = []

    for mask_prob in [float(value) for value in benchmark_cfg["mask_probs"]]:
        mask = make_fixed_mask(
            matrix=matrix,
            mask_prob=mask_prob,
            seed=seed + int(round(mask_prob * 1000)),
            nonzero_only=bool(benchmark_cfg.get("nonzero_only", True)),
        )
        loader = DataLoader(
            MatrixMaskDataset(x=matrix, mask=mask),
            batch_size=int(data_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
        )
        probability, positive_value, expected_repair, target, used_mask = predict_outputs(
            model,
            loader,
            device,
            amp_enabled,
            amp_dtype,
        )
        true_zero = target <= 1.0e-8
        positive = used_mask
        eligible = positive | true_zero
        labels = positive[eligible]
        scores = probability[eligible]
        diagnostic_rows.append(
            {
                "mask_prob": mask_prob,
                "n_positive": int(positive.sum()),
                "n_true_zero": int(true_zero.sum()),
                "positive_prevalence": float(labels.mean()),
                "dropout_auroc": auroc(labels, scores),
                "dropout_auprc": average_precision(labels, scores),
                "positive_probability_mean": float(probability[positive].mean()),
                "negative_probability_mean": float(probability[true_zero].mean()),
                "positive_value_mean": float(positive_value[positive].mean()),
                "expected_repair_mean": float(expected_repair[positive].mean()),
                "target_mean": float(target[positive].mean()),
            }
        )
        for threshold in thresholds:
            detected = probability >= threshold
            effective = detected & (expected_repair > value_threshold)
            detection_metrics = binary_metrics(detected, positive, eligible)
            effective_metrics = binary_metrics(effective, positive, eligible)
            masked_difference = expected_repair[positive] - target[positive]
            sweep_rows.append(
                {
                    "mask_prob": mask_prob,
                    "threshold": threshold,
                    "detection_precision": detection_metrics["precision"],
                    "detection_recall": detection_metrics["recall"],
                    "detection_f1": detection_metrics["f1"],
                    "detection_false_positive_rate": detection_metrics["fpr"],
                    "effective_precision": effective_metrics["precision"],
                    "effective_recall": effective_metrics["recall"],
                    "effective_f1": effective_metrics["f1"],
                    "effective_true_zero_fill_rate": effective_metrics["fpr"],
                    "masked_expected_mse": float(np.mean(masked_difference**2)),
                    "masked_expected_mae": float(np.mean(np.abs(masked_difference))),
                }
            )

    diagnostics = pd.DataFrame(diagnostic_rows)
    sweep = pd.DataFrame(sweep_rows)
    diagnostics.to_csv(out_dir / "dropout_head_diagnostics.csv", index=False)
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)

    selected_payload: dict[str, Any] | None = None
    if mode == "calibrate":
        aggregate = (
            sweep.groupby("threshold", as_index=False)
            .agg(
                mean_detection_f1=("detection_f1", "mean"),
                mean_effective_f1=("effective_f1", "mean"),
                mean_effective_recall=("effective_recall", "mean"),
                mean_true_zero_fill_rate=("effective_true_zero_fill_rate", "mean"),
                mean_masked_expected_mse=("masked_expected_mse", "mean"),
            )
        )
        feasible = aggregate[aggregate["mean_true_zero_fill_rate"] <= max_fill]
        candidates = feasible if not feasible.empty else aggregate
        candidates = candidates.sort_values(
            ["mean_effective_f1", "mean_effective_recall", "mean_detection_f1", "mean_masked_expected_mse"],
            ascending=[False, False, False, True],
        )
        selected_threshold = float(candidates.iloc[0]["threshold"])
        aggregate["selected"] = np.isclose(aggregate["threshold"], selected_threshold)
        aggregate.to_csv(out_dir / "threshold_aggregate.csv", index=False)
        selected_payload = {
            "selected_threshold": selected_threshold,
            "selection_split": data_cfg.get("split"),
            "max_true_zero_fill_rate": max_fill,
            "value_threshold": value_threshold,
            "constraint_satisfied": not feasible.empty,
        }
        atomic_json_dump(selected_payload, out_dir / "selected_threshold.json")
        print(f"selected_threshold={selected_threshold}")
        print(aggregate.to_string(index=False))
    else:
        print("frozen threshold metrics:")
        print(sweep.to_string(index=False))

    atomic_json_dump(
        {
            "mode": mode,
            "checkpoint_path": str(checkpoint_path),
            "diagnostics": diagnostic_rows,
            "threshold_metrics": sweep_rows,
            "selection": selected_payload,
        },
        out_dir / "summary.json",
    )
    print(f"saved evaluation outputs to:\n{out_dir}")
    print("dropout diagnostics:")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
