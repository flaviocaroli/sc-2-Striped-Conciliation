from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sc2.config import load_yaml, merge_train_and_paths
from sc2.data.archs4_denoise_datasets import ARCHS4DenoiseDataset
from sc2.data.census_shared_datasets import CensusSharedDataset
from sc2.data.mixed_loaders import infinite_loader
from sc2.data.pseudobulk_datasets import PseudobulkSharedDataset
from sc2.losses.bridge_v2_losses import bridge_alignment_loss, weighted_masked_mse
from sc2.models.sc2lite_bridge_denoiser import SC2LiteBridgeDenoiser
from sc2.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SC2Lite bridge v2 with weighted masked loss and stronger alignment.")
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
def eval_recon_loader(
    model: SC2LiteBridgeDenoiser,
    loader: DataLoader,
    modality: str,
    device: torch.device,
    masked_position_weight: float,
) -> tuple[float, float, float]:
    """
    Returns:
      weighted_loss_mean, all_loss_mean, masked_loss_mean
    """
    model.eval()

    total_weighted = 0.0
    total_all = 0.0
    total_masked = 0.0
    total_n = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = model(x, modality=modality)
        loss, stats = weighted_masked_mse(
            pred=pred,
            y_clean=y,
            x_corrupt=x,
            masked_position_weight=masked_position_weight,
        )

        bsz = x.shape[0]
        total_weighted += float(loss.detach().cpu().item()) * bsz
        total_all += stats["loss_all"] * bsz
        total_masked += stats["loss_masked"] * bsz
        total_n += bsz

    denom = max(total_n, 1)
    return total_weighted / denom, total_all / denom, total_masked / denom


@torch.no_grad()
def eval_alignment(
    model: SC2LiteBridgeDenoiser,
    bulk_loader: DataLoader,
    pb_loader: DataLoader,
    device: torch.device,
    align_mean_weight: float,
    align_coral_weight: float,
) -> tuple[float, float, float]:
    model.eval()

    bulk_iter = infinite_loader(bulk_loader)
    pb_iter = infinite_loader(pb_loader)
    n_steps = max(len(bulk_loader), len(pb_loader))

    total = 0.0
    total_mean = 0.0
    total_coral = 0.0

    for _ in range(n_steps):
        bb = next(bulk_iter)
        pb = next(pb_iter)

        xb = bb["x"].to(device)
        xp = pb["x"].to(device)

        zb = model.encode(xb, modality="bulk")
        zp = model.encode(xp, modality="pseudobulk")

        loss, stats = bridge_alignment_loss(
            z_bulk=zb,
            z_pseudobulk=zp,
            mean_weight=align_mean_weight,
            coral_weight=align_coral_weight,
        )

        total += float(loss.detach().cpu().item())
        total_mean += stats["loss_mean"]
        total_coral += stats["loss_coral"]

    denom = max(n_steps, 1)
    return total / denom, total_mean / denom, total_coral / denom


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
        split="train",
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["bulk_mask_prob"]),
        noise_std=float(data_cfg["bulk_noise_std"]),
        seed=seed,
    )
    bulk_val = ARCHS4DenoiseDataset(
        split="val",
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["bulk_mask_prob"]),
        noise_std=float(data_cfg["bulk_noise_std"]),
        seed=seed,
    )
    bulk_test = ARCHS4DenoiseDataset(
        split="test",
        h5_path=bulk_h5_path,
        sample_manifest_path=bulk_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["bulk_mask_prob"]),
        noise_std=float(data_cfg["bulk_noise_std"]),
        seed=seed,
    )

    sc_train = CensusSharedDataset(
        split="train",
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["sc_mask_prob"]),
        noise_std=float(data_cfg["sc_noise_std"]),
        seed=seed,
    )
    sc_val = CensusSharedDataset(
        split="val",
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["sc_mask_prob"]),
        noise_std=float(data_cfg["sc_noise_std"]),
        seed=seed,
    )
    sc_test = CensusSharedDataset(
        split="test",
        h5ad_path=sc_h5ad_path,
        split_manifest_path=sc_split_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["sc_mask_prob"]),
        noise_std=float(data_cfg["sc_noise_std"]),
        seed=seed,
    )

    pb_train = PseudobulkSharedDataset(
        split="train",
        h5ad_path=pb_h5ad_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["pb_mask_prob"]),
        noise_std=float(data_cfg["pb_noise_std"]),
        seed=seed,
    )
    pb_val = PseudobulkSharedDataset(
        split="val",
        h5ad_path=pb_h5ad_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["pb_mask_prob"]),
        noise_std=float(data_cfg["pb_noise_std"]),
        seed=seed,
    )
    pb_test = PseudobulkSharedDataset(
        split="test",
        h5ad_path=pb_h5ad_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
        mask_prob=float(data_cfg["pb_mask_prob"]),
        noise_std=float(data_cfg["pb_noise_std"]),
        seed=seed,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_section["lr"]),
        weight_decay=float(train_section["weight_decay"]),
    )

    bulk_loss_weight = float(train_section["bulk_loss_weight"])
    sc_loss_weight = float(train_section["sc_loss_weight"])
    pb_loss_weight = float(train_section["pb_loss_weight"])
    align_loss_weight = float(train_section["align_loss_weight"])

    bulk_masked_position_weight = float(train_section["bulk_masked_position_weight"])
    sc_masked_position_weight = float(train_section["sc_masked_position_weight"])
    pb_masked_position_weight = float(train_section["pb_masked_position_weight"])

    align_mean_weight = float(train_section["align_mean_weight"])
    align_coral_weight = float(train_section["align_coral_weight"])

    best_val_total = float("inf")
    history: list[dict[str, float | int]] = []

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
        train_align_mean = 0.0
        train_align_coral = 0.0

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

            loss_b, _ = weighted_masked_mse(
                pred=pred_b,
                y_clean=yb,
                x_corrupt=xb,
                masked_position_weight=bulk_masked_position_weight,
            )
            loss_s, _ = weighted_masked_mse(
                pred=pred_s,
                y_clean=ys,
                x_corrupt=xs,
                masked_position_weight=sc_masked_position_weight,
            )
            loss_p, _ = weighted_masked_mse(
                pred=pred_p,
                y_clean=yp,
                x_corrupt=xp,
                masked_position_weight=pb_masked_position_weight,
            )

            z_b = model.encode(xb, modality="bulk")
            z_p = model.encode(xp, modality="pseudobulk")
            loss_align, align_stats = bridge_alignment_loss(
                z_bulk=z_b,
                z_pseudobulk=z_p,
                mean_weight=align_mean_weight,
                coral_weight=align_coral_weight,
            )

            loss = (
                bulk_loss_weight * loss_b
                + sc_loss_weight * loss_s
                + pb_loss_weight * loss_p
                + align_loss_weight * loss_align
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=float(train_section["grad_clip_norm"]))
            optimizer.step()

            train_bulk_loss += float(loss_b.detach().cpu().item())
            train_sc_loss += float(loss_s.detach().cpu().item())
            train_pb_loss += float(loss_p.detach().cpu().item())
            train_align_loss += float(loss_align.detach().cpu().item())
            train_align_mean += align_stats["loss_mean"]
            train_align_coral += align_stats["loss_coral"]

        train_bulk_loss /= n_steps
        train_sc_loss /= n_steps
        train_pb_loss /= n_steps
        train_align_loss /= n_steps
        train_align_mean /= n_steps
        train_align_coral /= n_steps

        val_bulk_loss, val_bulk_all, val_bulk_masked = eval_recon_loader(
            model, bulk_val_loader, "bulk", device, bulk_masked_position_weight
        )
        val_sc_loss, val_sc_all, val_sc_masked = eval_recon_loader(
            model, sc_val_loader, "sc", device, sc_masked_position_weight
        )
        val_pb_loss, val_pb_all, val_pb_masked = eval_recon_loader(
            model, pb_val_loader, "pseudobulk", device, pb_masked_position_weight
        )
        val_align_loss, val_align_mean, val_align_coral = eval_alignment(
            model, bulk_val_loader, pb_val_loader, device, align_mean_weight, align_coral_weight
        )

        val_total = (
            bulk_loss_weight * val_bulk_loss
            + sc_loss_weight * val_sc_loss
            + pb_loss_weight * val_pb_loss
            + align_loss_weight * val_align_loss
        )

        row = {
            "epoch": epoch,
            "train_bulk_loss": train_bulk_loss,
            "train_sc_loss": train_sc_loss,
            "train_pb_loss": train_pb_loss,
            "train_align_loss": train_align_loss,
            "train_align_mean": train_align_mean,
            "train_align_coral": train_align_coral,
            "val_bulk_loss": val_bulk_loss,
            "val_sc_loss": val_sc_loss,
            "val_pb_loss": val_pb_loss,
            "val_align_loss": val_align_loss,
            "val_align_mean": val_align_mean,
            "val_align_coral": val_align_coral,
            "val_total": val_total,
            "val_bulk_all": val_bulk_all,
            "val_bulk_masked": val_bulk_masked,
            "val_sc_all": val_sc_all,
            "val_sc_masked": val_sc_masked,
            "val_pb_all": val_pb_all,
            "val_pb_masked": val_pb_masked,
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
            f"val_align_loss={val_align_loss:.6f} "
            f"val_total={val_total:.6f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            },
            ckpt_dir / "last.pt",
        )

        if val_total < best_val_total:
            best_val_total = val_total
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )

    test_bulk_loss, _, _ = eval_recon_loader(model, bulk_test_loader, "bulk", device, bulk_masked_position_weight)
    test_sc_loss, _, _ = eval_recon_loader(model, sc_test_loader, "sc", device, sc_masked_position_weight)
    test_pb_loss, _, _ = eval_recon_loader(model, pb_test_loader, "pseudobulk", device, pb_masked_position_weight)
    test_align_loss, test_align_mean, test_align_coral = eval_alignment(
        model, bulk_test_loader, pb_test_loader, device, align_mean_weight, align_coral_weight
    )

    test_total = (
        bulk_loss_weight * test_bulk_loss
        + sc_loss_weight * test_sc_loss
        + pb_loss_weight * test_pb_loss
        + align_loss_weight * test_align_loss
    )

    summary = {
        "best_val_total": best_val_total,
        "test_bulk_loss": test_bulk_loss,
        "test_sc_loss": test_sc_loss,
        "test_pb_loss": test_pb_loss,
        "test_align_loss": test_align_loss,
        "test_align_mean": test_align_mean,
        "test_align_coral": test_align_coral,
        "test_total": test_total,
        "history": history,
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"best_val_total={best_val_total:.6f}")
    print(f"test_bulk_loss={test_bulk_loss:.6f}")
    print(f"test_sc_loss={test_sc_loss:.6f}")
    print(f"test_pb_loss={test_pb_loss:.6f}")
    print(f"test_align_loss={test_align_loss:.6f}")
    print(f"test_total={test_total:.6f}")
    print(f"saved outputs to {run_dir}")


if __name__ == "__main__":
    main()