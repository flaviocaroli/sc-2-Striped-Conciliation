from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sc2.config import load_yaml, merge_train_and_paths
from sc2.data.archs4_datasets import ARCHS4SubsetDataset
from sc2.models.bulk_autoencoder import BulkAutoencoder
from sc2.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a first bulk autoencoder on ARCHS4 pilot data.")
    parser.add_argument("--config", required=True, help="Path to training config yaml")
    parser.add_argument("--paths", required=True, help="Path to path config yaml")
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = criterion(pred, x)

        if training:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

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

    h5_path = resolve_data_path(data_root, data_cfg["h5_path"])
    sample_manifest_path = resolve_data_path(data_root, data_cfg["sample_manifest_path"])
    shared_gene_table_path = resolve_data_path(data_root, data_cfg["shared_gene_table_path"])

    print(f"h5_path={h5_path}")
    print(f"sample_manifest_path={sample_manifest_path}")
    print(f"shared_gene_table_path={shared_gene_table_path}")

    train_ds = ARCHS4SubsetDataset(
        split="train",
        h5_path=h5_path,
        sample_manifest_path=sample_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
    )
    val_ds = ARCHS4SubsetDataset(
        split="val",
        h5_path=h5_path,
        sample_manifest_path=sample_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
    )
    test_ds = ARCHS4SubsetDataset(
        split="test",
        h5_path=h5_path,
        sample_manifest_path=sample_manifest_path,
        shared_gene_table_path=shared_gene_table_path,
        n_genes=int(data_cfg["n_genes"]),
        log1p_input=bool(data_cfg["log1p_input"]),
    )

    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"input_dim={train_ds.n_features}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    model = BulkAutoencoder(
        input_dim=train_ds.n_features,
        hidden_dims=model_cfg["hidden_dims"],
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_section["lr"]),
        weight_decay=float(train_section["weight_decay"]),
    )

    best_val = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(train_section["epochs"]) + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=float(train_section["grad_clip_norm"]),
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            grad_clip_norm=float(train_section["grad_clip_norm"]),
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)

        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            },
            ckpt_dir / "last.pt",
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )

    test_loss = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        grad_clip_norm=float(train_section["grad_clip_norm"]),
    )

    summary = {
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "history": history,
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"best_val_loss={best_val:.6f}")
    print(f"test_loss={test_loss:.6f}")
    print(f"saved outputs to {run_dir}")


if __name__ == "__main__":
    main()