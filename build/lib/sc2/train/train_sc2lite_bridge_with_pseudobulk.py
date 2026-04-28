from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sc2.config import load_yaml, merge_train_and_paths
from sc2.data.archs4_denoise_datasets import ARCHS4DenoiseDataset
from sc2.data.census_shared_datasets import CensusSharedDataset
from sc2.data.mixed_loaders import infinite_loader
from sc2.data.pseudobulk_datasets import PseudobulkSharedDataset
from sc2.models.sc2lite_bridge_denoiser import SC2LiteBridgeDenoiser
from sc2.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SC2Lite bridge denoiser with pseudobulk alignment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg_device: str) -> torch.device:
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_data_path(base_data_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_data_root / path


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, modality: str, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x, modality=modality)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.shape[0]
        total_count += x.shape[0]
    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml(args.config)
    path_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(train_cfg, path_cfg)

    seed = int(cfg["seed"])
    seed_everything(seed)

    output_root = Path(cfg["paths"]["output_root"])
    data_root = Path(cfg["paths"]["data_root"])
    run_dir = output_root / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = get_device(cfg["device"])
    print(f"device={device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_section = cfg["train"]

    bulk_h5_path = resolve_data_path(data_root, data_cfg["bulk_h5_path"])
    bulk_manifest_path = resolve_data_path(data_root, data_cfg["bulk_manifest_path"])
    sc_h5ad_path = resolve_data_path(data_root, data_cfg["sc_h5ad_path"])
    sc_split_manifest_path = resolve_data_path(data_root, data_cfg["sc_split_manifest_path"])
    pb_h5ad_path = resolve_data_path(data_root, data_cfg["pseudobulk_h5ad_path"])
    shared_gene_table_path = resolve_data_path(data_root, data_cfg["shared_gene_table_path"])

    bulk_train = ARCHS4DenoiseDataset(
        split="train", h5_path=bulk_h5_path, sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path, n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]), mask_prob=float(data_cfg["bulk_mask_prob"]),
        noise_std=float(data_cfg["bulk_noise_std"]), seed=seed
    )
    bulk_val = ARCHS4DenoiseDataset(
        split="val", h5_path=bulk_h5_path, sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path, n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]), mask_prob=float(data_cfg["bulk_mask_prob"]),
        noise_std=float(data_cfg["bulk_noise_std"]), seed=seed
    )
    bulk_test = ARCHS4DenoiseDataset(
        split="test", h5_path=bulk_h5_path, sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path, n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]), mask_prob=float(data_cfg["bulk_mask_prob"]),
        noise_std=float(data_cfg["bulk_noise_std"]), seed=seed
    )

    sc_train = CensusSharedDataset(
        split="train", h5ad_path=sc_h5ad_path, split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path, n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]), mask_prob=float(data_cfg["sc_mask_prob"]),
        noise_std=float(data_cfg["sc_noise_std"]), seed=seed
    )
    sc_val = CensusSharedDataset(
        split="val", h5ad_path=sc_h5ad_path, split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path, n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]), mask_prob=float(data_cfg["sc_mask_prob"]),
        noise_std=float(data_cfg["sc_noise_std"]), seed=seed
    )
    sc_test = CensusSharedDataset(
        split="test", h5ad_path=sc_h5ad_path, split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path, n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]), mask_prob=float(data_cfg["sc_mask_prob"]),
        noise_std=float(data_cfg["sc_noise_std"]), seed=seed
    )

    pb_train = PseudobulkSharedDataset(
        split="train", h5ad_path=pb_h5ad_path, shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]), log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["pb_mask_prob"]), noise_std=float(data_cfg["pb_noise_std"]), seed=seed
    )
    pb_val = PseudobulkSharedDataset(
        split="val", h5ad_path=pb_h5ad_path, shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]), log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["pb_mask_prob"]), noise_std=float(data_cfg["pb_noise_std"]), seed=seed
    )
    pb_test = PseudobulkSharedDataset(
        split="test", h5ad_path=pb_h5ad_path, shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]), log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["pb_mask_prob"]), noise_std=float(data_cfg["pb_noise_std"]), seed=seed
    )

    print(f"bulk_train={len(bulk_train)} bulk_val={len(bulk_val)} bulk_test={len(bulk_test)}")
    print(f"sc_train={len(sc_train)} sc_val={len(sc_val)} sc_test={len(sc_test)}")
    print(f"pb_train={len(pb_train)} pb_val={len(pb_val)} pb_test={len(pb_test)}")
    print(f"input_dim={bulk_train.n_features}")

    bulk_train_loader = DataLoader(bulk_train, batch_size=int(data_cfg["bulk_batch_size"]), shuffle=True, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    bulk_val_loader = DataLoader(bulk_val, batch_size=int(data_cfg["bulk_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    bulk_test_loader = DataLoader(bulk_test, batch_size=int(data_cfg["bulk_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())

    sc_train_loader = DataLoader(sc_train, batch_size=int(data_cfg["sc_batch_size"]), shuffle=True, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    sc_val_loader = DataLoader(sc_val, batch_size=int(data_cfg["sc_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    sc_test_loader = DataLoader(sc_test, batch_size=int(data_cfg["sc_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())

    pb_train_loader = DataLoader(pb_train, batch_size=int(data_cfg["pb_batch_size"]), shuffle=True, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    pb_val_loader = DataLoader(pb_val, batch_size=int(data_cfg["pb_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())
    pb_test_loader = DataLoader(pb_test, batch_size=int(data_cfg["pb_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=torch.cuda.is_available())

    model = SC2LiteBridgeDenoiser(
        input_dim=int(data_cfg["n_genes"]),
        adapter_dim=int(model_cfg["adapter_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_section["lr"]), weight_decay=float(train_section["weight_decay"]))

    bulk_w = float(train_section["bulk_loss_weight"])
    sc_w = float(train_section["sc_loss_weight"])
    pb_w = float(train_section["pb_loss_weight"])
    align_w = float(train_section["align_loss_weight"])

    best_val_total = float("inf")
    history = []

    for epoch in range(1, int(train_section["epochs"]) + 1):
        model.train()
        bulk_iter = infinite_loader(bulk_train_loader)
        sc_iter = infinite_loader(sc_train_loader)
        pb_iter = infinite_loader(pb_train_loader)
        n_steps = max(len(bulk_train_loader), len(sc_train_loader), len(pb_train_loader))

        train_bulk_loss = 0.0
        train_sc_loss = 0.0
        train_pb_loss = 0.0
        train_align_loss = 0.0

        for _ in range(n_steps):
            bb = next(bulk_iter)
            sb = next(sc_iter)
            pb = next(pb_iter)

            xb, yb = bb["x"].to(device), bb["y"].to(device)
            xs, ys = sb["x"].to(device), sb["y"].to(device)
            xp, yp = pb["x"].to(device), pb["y"].to(device)

            optimizer.zero_grad(set_to_none=True)

            pred_b = model(xb, modality="bulk")
            pred_s = model(xs, modality="sc")
            pred_p = model(xp, modality="pseudobulk")

            loss_b = criterion(pred_b, yb)
            loss_s = criterion(pred_s, ys)
            loss_p = criterion(pred_p, yp)

            z_b = model.encode(xb, modality="bulk")
            z_p = model.encode(xp, modality="pseudobulk")
            mean_b = F.normalize(z_b.mean(dim=0, keepdim=True), dim=1)
            mean_p = F.normalize(z_p.mean(dim=0, keepdim=True), dim=1)
            loss_align = F.mse_loss(mean_b, mean_p)

            loss = bulk_w * loss_b + sc_w * loss_s + pb_w * loss_p + align_w * loss_align
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=float(train_section["grad_clip_norm"]))
            optimizer.step()

            train_bulk_loss += loss_b.item()
            train_sc_loss += loss_s.item()
            train_pb_loss += loss_p.item()
            train_align_loss += loss_align.item()

        train_bulk_loss /= n_steps
        train_sc_loss /= n_steps
        train_pb_loss /= n_steps
        train_align_loss /= n_steps

        val_bulk_loss = eval_loader(model, bulk_val_loader, "bulk", device)
        val_sc_loss = eval_loader(model, sc_val_loader, "sc", device)
        val_pb_loss = eval_loader(model, pb_val_loader, "pseudobulk", device)
        val_total = bulk_w * val_bulk_loss + sc_w * val_sc_loss + pb_w * val_pb_loss

        row = {
            "epoch": epoch,
            "train_bulk_loss": train_bulk_loss,
            "train_sc_loss": train_sc_loss,
            "train_pb_loss": train_pb_loss,
            "train_align_loss": train_align_loss,
            "val_bulk_loss": val_bulk_loss,
            "val_sc_loss": val_sc_loss,
            "val_pb_loss": val_pb_loss,
            "val_total": val_total,
        }
        history.append(row)

        print(
            f"epoch={epoch} "
            f"train_bulk_loss={train_bulk_loss:.6f} "
            f"train_sc_loss={train_sc_loss:.6f} "
            f"train_pb_loss={train_pb_loss:.6f} "
            f"train_align_loss={train_align_loss:.6f} "
            f"val_bulk_loss={val_bulk_loss:.6f} "
            f"val_sc_loss={val_sc_loss:.6f} "
            f"val_pb_loss={val_pb_loss:.6f} "
            f"val_total={val_total:.6f}"
        )

        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": cfg},
            ckpt_dir / "last.pt",
        )

        if val_total < best_val_total:
            best_val_total = val_total
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": cfg},
                ckpt_dir / "best.pt",
            )

    test_bulk_loss = eval_loader(model, bulk_test_loader, "bulk", device)
    test_sc_loss = eval_loader(model, sc_test_loader, "sc", device)
    test_pb_loss = eval_loader(model, pb_test_loader, "pseudobulk", device)
    test_total = bulk_w * test_bulk_loss + sc_w * test_sc_loss + pb_w * test_pb_loss

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_total": best_val_total,
                "test_bulk_loss": test_bulk_loss,
                "test_sc_loss": test_sc_loss,
                "test_pb_loss": test_pb_loss,
                "test_total": test_total,
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"best_val_total={best_val_total:.6f}")
    print(f"test_bulk_loss={test_bulk_loss:.6f}")
    print(f"test_sc_loss={test_sc_loss:.6f}")
    print(f"test_pb_loss={test_pb_loss:.6f}")
    print(f"test_total={test_total:.6f}")
    print(f"saved outputs to {run_dir}")


if __name__ == "__main__":
    main()