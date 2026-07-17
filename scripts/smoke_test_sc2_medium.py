#!/usr/bin/env python
from __future__ import annotations

import torch

from sc2.models.striped.sc2_striped_medium import SC2StripedMedium


def main() -> None:
    model = SC2StripedMedium(
        n_genes=128,
        d_model=64,
        n_mamba_blocks=2,
        n_attention_checkpoints=1,
        d_state=8,
        d_conv=5,
        expand=1,
        top_k=32,
        n_heads=4,
        use_rank_bins=True,
        n_rank_bins=16,
        dropout_head=True,
    )
    x = torch.rand(3, 128)
    out = model(x, modality="sc", return_dict=True)
    assert out["reconstruction"].shape == (3, 128)
    assert out["dropout_logits"].shape == (3, 128)
    print("ok", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
