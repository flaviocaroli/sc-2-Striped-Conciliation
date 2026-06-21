from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def build_model(model_cfg: dict[str, Any], n_genes: int) -> tuple[str, torch.nn.Module]:
    kind = str(model_cfg["kind"]).strip().lower()

    if kind == "bulk_autoencoder":
        from sc2.models.bulk_autoencoder import BulkAutoencoder

        model = BulkAutoencoder(
            input_dim=n_genes,
            hidden_dims=model_cfg["hidden_dims"],
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        return kind, model

    if kind == "sc2lite_denoiser":
        from sc2.models.sc2lite_denoiser import SC2LiteDenoiser

        model = SC2LiteDenoiser(
            input_dim=n_genes,
            adapter_dim=int(model_cfg["adapter_dim"]),
            latent_dim=int(model_cfg["latent_dim"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        return kind, model

    if kind == "sc2lite_bridge":
        from sc2.models.sc2lite_bridge_denoiser import SC2LiteBridgeDenoiser

        model = SC2LiteBridgeDenoiser(
            input_dim=n_genes,
            adapter_dim=int(model_cfg["adapter_dim"]),
            latent_dim=int(model_cfg["latent_dim"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        return kind, model

    if kind == "sc2_mamba_bridge":
        try:
            from sc2.models.sc2_mamba_bridge import SC2MambaBridge
        except ImportError as exc:
            raise ImportError(
                "model.kind='sc2_mamba_bridge' requires mamba-ssm. "
                "Use model.kind='native_mamba_bridge' for the native implementation."
            ) from exc

        model = SC2MambaBridge(
            n_genes=n_genes,
            d_model=int(model_cfg["d_model"]),
            n_layers=int(model_cfg["n_layers"]),
            d_state=int(model_cfg["d_state"]),
            d_conv=int(model_cfg["d_conv"]),
            expand=int(model_cfg["expand"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
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
            dropout=float(model_cfg.get("dropout", 0.0)),
            mixer_type=str(model_cfg.get("mixer_type", "mamba1")),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            merge_mode=str(model_cfg.get("merge_mode", "sum")),
            smart_flip=bool(model_cfg.get("smart_flip", False)),
            rank_input=bool(model_cfg.get("rank_input", False)),
            preserve_prefix_tokens=int(model_cfg.get("preserve_prefix_tokens", 0)),
            norm_type=str(model_cfg.get("norm_type", "rmsnorm")),
        )
        return kind, model

    if kind in {"sc2_striped_mini", "striped_mini"}:
        from sc2.models.striped.sc2_striped_mini import SC2StripedMini

        model = SC2StripedMini(
            n_genes=n_genes,
            d_model=int(model_cfg.get("d_model", 128)),
            n_mamba_blocks=int(model_cfg.get("n_mamba_blocks", 6)),
            n_attention_checkpoints=int(model_cfg.get("n_attention_checkpoints", 2)),
            d_state=int(model_cfg.get("d_state", 16)),
            d_conv=int(model_cfg.get("d_conv", 4)),
            expand=int(model_cfg.get("expand", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            top_k=int(model_cfg.get("top_k", 256)),
            use_rank_bins=bool(model_cfg.get("use_rank_bins", False)),
            n_rank_bins=int(model_cfg.get("n_rank_bins", 16)),
        )
        return kind, model

    if kind in {"sc2_striped_medium", "striped_medium", "sc2_medium"}:
        from sc2.models.striped.sc2_striped_medium import build_sc2_striped_medium_from_config
        model = build_sc2_striped_medium_from_config(model_cfg, n_genes=n_genes)
        return "sc2_striped_medium", model

    raise ValueError(f"Unsupported model kind: {kind}")


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    fallback_model_cfg: dict[str, Any],
    fallback_n_genes: int,
    map_location: str | torch.device = "cpu",
) -> tuple[str, torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", fallback_model_cfg)
    n_genes = int(cfg.get("data", {}).get("n_genes", fallback_n_genes))

    kind, model = build_model(model_cfg, n_genes=n_genes)
    model.load_state_dict(checkpoint["model_state_dict"])

    return kind, model, checkpoint