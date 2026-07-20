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
from torch.utils.data import DataLoader

from sc2.config import load_yaml, merge_train_and_paths
from sc2.eval.benchmarks.evaluate_masked_reconstruction import (
    MatrixMaskDataset,
    dense_matrix,
    genewise_metrics,
    make_fixed_mask,
    samplewise_metrics,
    subset_adata_by_manifest,
    subset_genes,
)
from sc2.models.striped.sc2_striped_full import build_sc2_striped_full_from_config
from sc2.train import train_sc2_mamba_bridge as bridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked reconstruction benchmark for SC2 20+4.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    prediction_key: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        observed = bridge.move_tensor(batch["x"], device)
        with bridge.autocast_context(device, amp_enabled, amp_dtype):
            output = model(observed, modality="sc", return_dict=True)
            if not isinstance(output, dict):
                raise TypeError("Expected dictionary model output")
            prediction = output[prediction_key]
        prediction_chunks.append(prediction.float().cpu().numpy())
        target_chunks.append(batch["target"].numpy())
        mask_chunks.append(batch["mask"].numpy())
    return (
        np.concatenate(prediction_chunks, axis=0).astype(np.float32),
        np.concatenate(target_chunks, axis=0).astype(np.float32),
        np.concatenate(mask_chunks, axis=0).astype(bool),
    )


def summarize(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    mask_prob: float,
    zero_fill_threshold: float,
) -> dict[str, float | int]:
    difference = prediction[mask] - target[mask]
    sample_df = samplewise_metrics(prediction, target, mask)
    gene_df = genewise_metrics(
        prediction,
        target,
        mask,
        gene_names=[str(index) for index in range(prediction.shape[1])],
    )
    masked_prediction = prediction[mask]
    masked_target = target[mask]
    prediction_std = float(np.std(masked_prediction))
    target_std = float(np.std(masked_target))
    true_zero = target <= 1.0e-8
    false_positive_zero_fill_rate = (
        float((prediction[true_zero] > float(zero_fill_threshold)).mean())
        if bool(true_zero.any())
        else float("nan")
    )
    return {
        "mask_prob": float(mask_prob),
        "n_cells": int(target.shape[0]),
        "n_genes": int(target.shape[1]),
        "n_masked_entries": int(mask.sum()),
        "sample_mse_mean": float(np.mean(difference**2)),
        "sample_rmse_mean": float(np.sqrt(np.mean(difference**2))),
        "sample_mae_mean": float(np.mean(np.abs(difference))),
        "sample_pearson_mean": finite_mean(sample_df["pearson"].to_numpy()),
        "sample_spearman_mean": finite_mean(sample_df["spearman"].to_numpy()),
        "gene_mse_mean": float(np.nanmean(gene_df["mse"].to_numpy())),
        "gene_rmse_mean": float(np.nanmean(np.sqrt(gene_df["mse"].to_numpy()))),
        "gene_mae_mean": float(np.nanmean(gene_df["mae"].to_numpy())),
        "gene_pearson_mean": finite_mean(gene_df["pearson"].to_numpy()),
        "gene_spearman_mean": finite_mean(gene_df["spearman"].to_numpy()),
        "false_positive_zero_fill_rate": false_positive_zero_fill_rate,
        "masked_prediction_mean": float(masked_prediction.mean()),
        "masked_prediction_std": prediction_std,
        "masked_target_mean": float(masked_target.mean()),
        "masked_target_std": target_std,
        "masked_std_ratio": prediction_std / target_std if target_std > 0.0 else float("nan"),
    }


def main() -> None:
    args = parse_args()
    benchmark_cfg = load_yaml(args.config)
    paths_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(benchmark_cfg, paths_cfg)
    seed = int(cfg.get("seed", 42))
    bridge.seed_everything(seed)
    device = bridge.get_device(cfg.get("device", "auto"))

    data_root = Path(cfg["paths"]["data_root"])
    output_root = Path(cfg["paths"]["output_root"])
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    bench_cfg = cfg["benchmark"]

    h5ad_path = resolve_path(data_root, data_cfg["h5ad_path"])
    manifest_path = resolve_path(data_root, data_cfg.get("split_manifest_path"))
    gene_table_path = resolve_path(data_root, data_cfg.get("shared_gene_table_path"))
    checkpoint_path = resolve_path(output_root, eval_cfg["checkpoint_path"])
    assert h5ad_path is not None and checkpoint_path is not None

    adata = ad.read_h5ad(h5ad_path)
    print(f"loaded adata shape={adata.shape}")
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
    print(f"benchmark matrix shape={matrix.shape}")

    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    checkpoint_cfg = checkpoint_data.get("config")
    if not isinstance(checkpoint_cfg, dict) or "model" not in checkpoint_cfg:
        raise ValueError("Checkpoint must contain its resolved model config")
    model = build_sc2_striped_full_from_config(
        checkpoint_cfg["model"],
        n_genes=matrix.shape[1],
    ).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(False)

    counts = bridge.count_parameters(model)
    prediction_key = str(eval_cfg.get("prediction_key", "reconstruction"))
    amp_enabled = bridge.resolve_amp(eval_cfg, device)
    amp_dtype = bridge.resolve_amp_dtype(eval_cfg, device) if amp_enabled else None
    print(f"device={device}")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"parameters_total={counts['total']}")
    print(f"attention_positions={getattr(model, 'attention_positions', None)}")
    print(f"prediction_key={prediction_key}")

    rows: list[dict[str, float | int]] = []
    for mask_prob in [float(value) for value in bench_cfg["mask_probs"]]:
        print(f"running mask_prob={mask_prob}")
        mask = make_fixed_mask(
            matrix=matrix,
            mask_prob=mask_prob,
            seed=seed + int(round(mask_prob * 1000)),
            nonzero_only=bool(bench_cfg.get("nonzero_only", True)),
        )
        loader = DataLoader(
            MatrixMaskDataset(x=matrix, mask=mask),
            batch_size=int(data_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
        )
        prediction, target, used_mask = predict(
            model,
            loader,
            device,
            prediction_key=prediction_key,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        row = summarize(
            prediction,
            target,
            used_mask,
            mask_prob=mask_prob,
            zero_fill_threshold=float(bench_cfg.get("zero_fill_threshold", 0.10)),
        )
        rows.append(row)
        print(pd.DataFrame([row]).to_string(index=False))

    result = pd.DataFrame(rows)
    out_dir = output_root / "evals" / str(cfg["eval_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "overall_masked_reconstruction.csv", index=False)
    atomic_json_dump(
        {
            "eval_name": cfg["eval_name"],
            "checkpoint_path": str(checkpoint_path),
            "prediction_key": prediction_key,
            "parameter_counts": counts,
            "attention_positions": getattr(model, "attention_positions", None),
            "results": rows,
        },
        out_dir / "summary.json",
    )
    print(f"saved evaluation outputs to:\n{out_dir}")
    print("overall masked reconstruction:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
