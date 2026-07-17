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
import yaml
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset


class MatrixMaskDataset(Dataset):
    def __init__(self, x: np.ndarray, mask: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.mask = mask.astype(bool)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        target = torch.from_numpy(self.x[idx]).float()
        mask = torch.from_numpy(self.mask[idx]).bool()
        corrupted = target.clone()
        corrupted[mask] = 0.0

        return {
            "x": corrupted,
            "target": target,
            "mask": mask,
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked sc reconstruction benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    return obj


def get_root(paths_cfg: dict[str, Any], key: str, env_key: str) -> Path:
    if "paths" in paths_cfg and isinstance(paths_cfg["paths"], dict):
        if key in paths_cfg["paths"]:
            return Path(paths_cfg["paths"][key])

    if key in paths_cfg:
        return Path(paths_cfg[key])

    env_value = os.environ.get(env_key)
    if env_value:
        return Path(env_value)

    raise KeyError(
        f"Could not find {key}. Expected paths.{key}, top-level {key}, "
        f"or environment variable {env_key}."
    )


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep=None, engine="python")


def dense_matrix(adata: ad.AnnData) -> np.ndarray:
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def subset_adata_by_manifest(
    adata: ad.AnnData,
    manifest_path: Path | None,
    split: str | None,
    split_col: str = "split",
) -> ad.AnnData:
    if manifest_path is None:
        print("split_manifest_path not provided; using all cells.")
        return adata

    if not manifest_path.exists():
        print(f"split_manifest_path does not exist: {manifest_path}; using all cells.")
        return adata

    manifest = read_table(manifest_path)

    if split is not None and split_col in manifest.columns:
        manifest = manifest.loc[manifest[split_col].astype(str) == str(split)].copy()
        print(f"using split={split}; manifest rows after filtering={len(manifest)}")
    elif split is not None:
        print(f"split_col={split_col} not found in manifest; using all manifest rows.")

    if len(manifest) == 0:
        raise ValueError("Manifest split filter produced zero rows.")

    integer_index_cols = [
        "obs_index",
        "cell_index",
        "row_index",
        "idx",
        "index",
        "h5ad_index",
    ]

    for col in integer_index_cols:
        if col in manifest.columns:
            idx = manifest[col].dropna().astype(int).to_numpy()
            idx = idx[(idx >= 0) & (idx < adata.n_obs)]
            if len(idx) > 0:
                print(f"subsetting AnnData using integer column={col}; n={len(idx)}")
                return adata[idx].copy()

    name_cols = [
        "obs_name",
        "cell_name",
        "cell_id",
        "barcode",
        "sample_id",
    ]

    obs_names = pd.Index(adata.obs_names.astype(str))

    for col in name_cols:
        if col in manifest.columns:
            names = manifest[col].dropna().astype(str)
            names = names[names.isin(obs_names)]
            if len(names) > 0:
                print(f"subsetting AnnData using name column={col}; n={len(names)}")
                return adata[names.to_list()].copy()

    if len(manifest) == adata.n_obs and split is not None and split_col in manifest.columns:
        mask = manifest[split_col].astype(str).to_numpy() == str(split)
        print(f"subsetting AnnData using row-aligned split mask; n={int(mask.sum())}")
        return adata[mask].copy()

    print(
        "Could not infer cell indices from manifest. "
        "Using all cells. For a stricter benchmark, add obs_index or obs_name to the manifest."
    )
    return adata


def subset_genes(
    adata: ad.AnnData,
    shared_gene_table_path: Path | None,
    n_genes: int,
) -> ad.AnnData:
    if shared_gene_table_path is None or not shared_gene_table_path.exists():
        print("shared_gene_table_path not provided/found; using first n_genes.")
        return adata[:, :n_genes].copy()

    table = read_table(shared_gene_table_path)
    var_names = pd.Index(adata.var_names.astype(str))

    preferred_cols = [
        "gene",
        "gene_name",
        "gene_symbol",
        "symbol",
        "hgnc_symbol",
        "feature_name",
        "ensembl_id",
        "gene_id",
    ]

    candidate_cols = [c for c in preferred_cols if c in table.columns]
    candidate_cols += [c for c in table.columns if c not in candidate_cols]

    best_col = None
    best_values: list[str] = []
    best_matches = 0

    for col in candidate_cols:
        values = table[col].dropna().astype(str).tolist()
        matches = sum(v in var_names for v in values)
        if matches > best_matches:
            best_col = col
            best_values = values
            best_matches = matches

    if best_col is not None and best_matches >= min(n_genes, 100):
        genes = [g for g in best_values if g in var_names][:n_genes]
        print(
            f"subsetting genes using shared table column={best_col}; "
            f"matched={len(genes)}"
        )
        return adata[:, genes].copy()

    print(
        "Could not match shared_gene_table to adata.var_names. "
        "Using first n_genes."
    )
    return adata[:, :n_genes].copy()


def make_fixed_mask(
    matrix: np.ndarray,
    mask_prob: float,
    seed: int,
    nonzero_only: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(matrix.shape) < float(mask_prob)

    if nonzero_only:
        mask = mask & (matrix > 0)

    return mask.astype(bool)


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.size < 3:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)

    raise ValueError(f"Unknown correlation method: {method}")


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(x) - np.asarray(y)) ** 2))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(x) - np.asarray(y))))


def samplewise_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for i in range(pred.shape[0]):
        m = mask[i]
        p = pred[i][m]
        t = target[i][m]

        if p.size == 0:
            rows.append(
                {
                    "sample_index": i,
                    "n_eval": 0,
                    "mse": float("nan"),
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "pearson": float("nan"),
                    "spearman": float("nan"),
                }
            )
            continue

        mse_i = mse(p, t)
        rows.append(
            {
                "sample_index": i,
                "n_eval": int(p.size),
                "mse": mse_i,
                "rmse": float(np.sqrt(mse_i)),
                "mae": mae(p, t),
                "pearson": safe_corr(p, t, "pearson"),
                "spearman": safe_corr(p, t, "spearman"),
            }
        )

    return pd.DataFrame(rows)


def genewise_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    gene_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for j in range(pred.shape[1]):
        m = mask[:, j]
        p = pred[:, j][m]
        t = target[:, j][m]

        if p.size == 0:
            rows.append(
                {
                    "gene_index": j,
                    "gene": gene_names[j],
                    "n_eval": 0,
                    "mse": float("nan"),
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "pearson": float("nan"),
                    "spearman": float("nan"),
                }
            )
            continue

        mse_j = mse(p, t)
        rows.append(
            {
                "gene_index": j,
                "gene": gene_names[j],
                "n_eval": int(p.size),
                "mse": mse_j,
                "rmse": float(np.sqrt(mse_j)),
                "mae": mae(p, t),
                "pearson": safe_corr(p, t, "pearson"),
                "spearman": safe_corr(p, t, "spearman"),
            }
        )

    return pd.DataFrame(rows)


def false_positive_zero_fill_rate(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> float:
    true_zero = target <= 0
    if int(true_zero.sum()) == 0:
        return float("nan")
    filled = pred[true_zero] > float(threshold)
    return float(filled.mean())


def build_model(model_cfg: dict[str, Any], n_genes: int) -> tuple[str, nn.Module]:
    kind = str(model_cfg["kind"]).strip().lower()

    if kind in {"sc2_striped_mini", "striped_mini"}:
        from sc2.models.striped.sc2_striped_mini import SC2StripedMini

        model = SC2StripedMini(
            n_genes=n_genes,
            d_model=int(model_cfg.get("d_model", 128)),
            n_mamba_blocks=int(model_cfg.get("n_mamba_blocks", 6)),
            n_attention_checkpoints=int(model_cfg.get("n_attention_checkpoints", 2)),
            d_state=int(model_cfg.get("d_state", 8)),
            d_conv=int(model_cfg.get("d_conv", 5)),
            expand=int(model_cfg.get("expand", 1)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            top_k=int(model_cfg.get("top_k", 256)),
            use_rank_bins=bool(model_cfg.get("use_rank_bins", False)),
            n_rank_bins=int(model_cfg.get("n_rank_bins", 16)),
        )
        return kind, model

    if kind in {"native_mamba_bridge", "sc2_native_mamba_bridge", "native_like_mamba_bridge"}:
        from sc2.models.sc2_native_mamba_bridge import SC2NativeMambaBridge

        model = SC2NativeMambaBridge(
            n_genes=n_genes,
            d_model=int(model_cfg["d_model"]),
            n_layers=int(model_cfg["n_layers"]),
            d_state=int(model_cfg["d_state"]),
            d_conv=int(model_cfg.get("d_conv", 4)),
            expand=int(model_cfg.get("expand", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            mixer_type=str(model_cfg.get("mixer_type", "mamba1")),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            merge_mode=str(model_cfg.get("merge_mode", "sum")),
            smart_flip=bool(model_cfg.get("smart_flip", False)),
            rank_input=bool(model_cfg.get("rank_input", False)),
            preserve_prefix_tokens=int(model_cfg.get("preserve_prefix_tokens", 0)),
            norm_type=str(model_cfg.get("norm_type", "rmsnorm")),
        )
        return kind, model

    if kind == "sc2_mamba_bridge":
        from sc2.models.sc2_mamba_bridge import SC2MambaBridge

        model = SC2MambaBridge(
            n_genes=n_genes,
            d_model=int(model_cfg["d_model"]),
            n_layers=int(model_cfg["n_layers"]),
            d_state=int(model_cfg["d_state"]),
            d_conv=int(model_cfg.get("d_conv", 4)),
            expand=int(model_cfg.get("expand", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        return kind, model

    if kind in {"sc2_striped_medium", "striped_medium", "sc2_medium"}:
        from sc2.models.striped.sc2_striped_medium import build_sc2_striped_medium_from_config
        model = build_sc2_striped_medium_from_config(model_cfg, n_genes=n_genes)
        return "sc2_striped_medium", model

    raise ValueError(
        "Unsupported model kind for masked reconstruction benchmark: "
        f"{kind}"
    )


@torch.no_grad()
def run_prediction(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    pred_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)

        pred = model(x, modality=modality)

        pred_chunks.append(pred.detach().cpu().numpy())
        target_chunks.append(batch["target"].numpy())
        mask_chunks.append(batch["mask"].numpy())

    return (
        np.concatenate(pred_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
        np.concatenate(mask_chunks, axis=0).astype(bool),
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths_cfg = load_yaml(args.paths)

    data_root = get_root(paths_cfg, "data_root", "SC2_DATA_ROOT")
    output_root = get_root(paths_cfg, "output_root", "SC2_OUTPUT_ROOT")

    eval_name = str(cfg["eval_name"])
    seed = int(cfg.get("seed", 42))

    device_cfg = str(cfg.get("device", "auto")).lower()
    if device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = output_root / "evals" / eval_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"device={device}")
    print(f"eval_name={eval_name}")

    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    benchmark_cfg = cfg["benchmark"]

    h5ad_path = resolve_path(data_root, data_cfg["h5ad_path"])
    split_manifest_path = resolve_path(data_root, data_cfg.get("split_manifest_path"))
    shared_gene_table_path = resolve_path(data_root, data_cfg.get("shared_gene_table_path"))
    checkpoint_path = resolve_path(output_root, eval_cfg["checkpoint_path"])

    if h5ad_path is None:
        raise ValueError("data.h5ad_path is required.")
    if checkpoint_path is None:
        raise ValueError("eval.checkpoint_path is required.")

    print(f"h5ad_path={h5ad_path}")
    print(f"split_manifest_path={split_manifest_path}")
    print(f"shared_gene_table_path={shared_gene_table_path}")
    print(f"checkpoint_path={checkpoint_path}")

    adata = ad.read_h5ad(h5ad_path)
    print(f"loaded adata shape={adata.shape}")

    adata = subset_adata_by_manifest(
        adata=adata,
        manifest_path=split_manifest_path,
        split=data_cfg.get("split", None),
        split_col=str(data_cfg.get("split_col", "split")),
    )

    adata = subset_genes(
        adata=adata,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg.get("n_genes", adata.n_vars)),
    )

    x = dense_matrix(adata)

    if bool(data_cfg.get("log1p_input", False)):
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)

    n_genes = x.shape[1]
    gene_names = [str(g) for g in adata.var_names]

    print(f"benchmark matrix shape={x.shape}")
    print(f"n_genes={n_genes}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    ckpt_cfg = checkpoint.get("config", {})
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    ckpt_n_genes = int(ckpt_cfg.get("data", {}).get("n_genes", n_genes))

    if ckpt_n_genes != n_genes:
        print(
            f"checkpoint n_genes={ckpt_n_genes} but benchmark matrix n_genes={n_genes}. "
            "Using checkpoint n_genes for model construction."
        )

    model_kind, model = build_model(model_cfg, n_genes=ckpt_n_genes)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    params = count_parameters(model)
    print(f"model_kind={model_kind}")
    print(f"parameters_total={params['total']}")
    print(f"parameters_trainable={params['trainable']}")

    if ckpt_n_genes != n_genes:
        raise ValueError(
            f"Model expects {ckpt_n_genes} genes but benchmark matrix has {n_genes}. "
            "Fix gene subsetting/shared_gene_table_path."
        )

    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    modality = str(benchmark_cfg.get("modality", "sc"))
    nonzero_only = bool(benchmark_cfg.get("nonzero_only", True))
    zero_fill_threshold = float(benchmark_cfg.get("zero_fill_threshold", 0.1))
    mask_probs = list(benchmark_cfg["mask_probs"])

    overall_rows: list[dict[str, float | int | str]] = []

    for mask_prob in mask_probs:
        mask_prob_f = float(mask_prob)
        print(f"running mask_prob={mask_prob_f}")

        mask = make_fixed_mask(
            matrix=x,
            mask_prob=mask_prob_f,
            seed=seed + int(round(mask_prob_f * 1000)),
            nonzero_only=nonzero_only,
        )

        ds = MatrixMaskDataset(x=x, mask=mask)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        pred, target, used_mask = run_prediction(
            model=model,
            loader=loader,
            device=device,
            modality=modality,
        )

        sample_df = samplewise_metrics(pred, target, used_mask)
        gene_df = genewise_metrics(pred, target, used_mask, gene_names=gene_names)

        sample_df["mask_prob"] = mask_prob_f
        gene_df["mask_prob"] = mask_prob_f

        fp_rate = false_positive_zero_fill_rate(
            pred=pred,
            target=target,
            threshold=zero_fill_threshold,
        )

        row = {
            "mask_prob": mask_prob_f,
            "n_cells": int(x.shape[0]),
            "n_genes": int(x.shape[1]),
            "n_masked_entries": int(used_mask.sum()),
            "sample_mse_mean": float(sample_df["mse"].mean()),
            "sample_rmse_mean": float(sample_df["rmse"].mean()),
            "sample_mae_mean": float(sample_df["mae"].mean()),
            "sample_pearson_mean": float(sample_df["pearson"].mean()),
            "sample_spearman_mean": float(sample_df["spearman"].mean()),
            "gene_mse_mean": float(gene_df["mse"].mean()),
            "gene_rmse_mean": float(gene_df["rmse"].mean()),
            "gene_mae_mean": float(gene_df["mae"].mean()),
            "gene_pearson_mean": float(gene_df["pearson"].mean()),
            "gene_spearman_mean": float(gene_df["spearman"].mean()),
            "false_positive_zero_fill_rate": float(fp_rate),
        }
        overall_rows.append(row)

        prob_label = str(mask_prob_f).replace(".", "p")
        sample_df.to_csv(out_dir / f"sample_metrics_mask_{prob_label}.csv", index=False)
        gene_df.to_csv(out_dir / f"gene_metrics_mask_{prob_label}.csv", index=False)
        np.savez_compressed(out_dir / f"mask_{prob_label}.npz", mask=used_mask)

        print(pd.DataFrame([row]))

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "overall_masked_reconstruction.csv", index=False)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": overall_rows,
                "paths": {
                    "overall": str(out_dir / "overall_masked_reconstruction.csv"),
                },
            },
            f,
            indent=2,
        )

    print("saved evaluation outputs to:")
    print(out_dir)
    print("overall masked reconstruction:")
    print(overall_df)


if __name__ == "__main__":
    main()