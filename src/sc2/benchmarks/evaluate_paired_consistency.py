from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml

from sc2.benchmarks.metrics import pseudobulk_consistency_metrics
from sc2.models.factory import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--paths", required=True)
    return p.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def dense_matrix(adata: ad.AnnData) -> np.ndarray:
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


@torch.no_grad()
def impute_cells(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    modality: str = "sc",
) -> np.ndarray:
    model.eval()
    outs = []

    for start in range(0, x.shape[0], batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        pred = model(batch, modality=modality)
        outs.append(pred.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = load_yaml(args.paths)

    data_root = Path(paths["paths"]["data_root"])
    output_root = Path(paths["paths"]["output_root"])
    out_dir = output_root / "evals" / cfg["eval_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")

    sc_path = resolve(data_root, cfg["data"]["sc_h5ad_path"])
    bulk_path = resolve(data_root, cfg["data"]["bulk_h5ad_path"])
    pair_manifest_path = resolve(data_root, cfg["data"]["pair_manifest_path"])
    checkpoint_path = resolve(output_root, cfg["eval"]["checkpoint_path"])

    print(f"device={device}")
    print(f"sc_h5ad_path={sc_path}")
    print(f"bulk_h5ad_path={bulk_path}")
    print(f"pair_manifest_path={pair_manifest_path}")
    print(f"checkpoint_path={checkpoint_path}")

    sc_adata = ad.read_h5ad(sc_path)
    bulk_adata = ad.read_h5ad(bulk_path)
    pairs = pd.read_parquet(pair_manifest_path)

    sc_x = dense_matrix(sc_adata)
    bulk_x = dense_matrix(bulk_adata)

    n_genes = int(cfg["data"].get("n_genes", sc_x.shape[1]))
    sc_x = sc_x[:, :n_genes]
    bulk_x = bulk_x[:, :n_genes]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = checkpoint.get("config", {})
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    n_genes = int(ckpt_cfg.get("data", {}).get("n_genes", n_genes))

    model_kind, model = build_model(model_cfg, n_genes=n_genes)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows = []

    donor_col = cfg["data"].get("donor_col", "donor_id")
    sc_index_col = cfg["data"].get("sc_index_col", "sc_index")
    bulk_index_col = cfg["data"].get("bulk_index_col", "bulk_index")
    split_col = cfg["data"].get("split_col", "split")
    eval_split = cfg["benchmark"].get("split", "test")

    pairs_eval = pairs[pairs[split_col] == eval_split].copy()

    for donor_id, df in pairs_eval.groupby(donor_col):
        sc_idx = df[sc_index_col].astype(int).to_numpy()
        bulk_idx = int(df[bulk_index_col].iloc[0])

        cells = sc_x[sc_idx]
        real_bulk = bulk_x[bulk_idx]

        pred_cells = impute_cells(
            model=model,
            x=cells,
            device=device,
            batch_size=int(cfg["data"].get("batch_size", 64)),
            modality="sc",
        )

        metrics = pseudobulk_consistency_metrics(pred_cells, real_bulk)
        metrics["donor_id"] = donor_id
        metrics["n_cells"] = int(len(sc_idx))
        metrics["split"] = eval_split
        rows.append(metrics)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "paired_consistency_by_donor.csv", index=False)

    overall = out.drop(columns=["donor_id", "split"], errors="ignore").mean(numeric_only=True).to_dict()
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"overall": overall}, f, indent=2)

    print("saved evaluation outputs to:")
    print(out_dir)
    print(pd.DataFrame([overall]))


if __name__ == "__main__":
    main()