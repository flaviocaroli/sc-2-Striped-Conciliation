from __future__ import annotations

import torch

from sc2.losses.striped_full_losses import compute_sc_objective
from sc2.models.striped.sc2_striped_full import SC2StripedFull


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SC2StripedFull(
        n_genes=128,
        d_model=32,
        n_mamba_blocks=4,
        n_attention_checkpoints=1,
        d_state=4,
        expand=1,
        top_k=32,
        n_heads=4,
        gradient_checkpointing=True,
        positive_activation="softplus",
        reconstruction_mode="preserve_observed",
    ).to(device)
    model.train()
    target = torch.rand(2, 128, device=device)
    target[target < 0.70] = 0.0
    observed = target.clone()
    positive = target > 0.0
    random_mask = (torch.rand_like(target) < 0.15) & positive
    observed[random_mask] = 0.0
    output = model(observed, modality="sc", return_dict=True)
    assert isinstance(output, dict)
    objective = compute_sc_objective(
        output,
        observed,
        target,
        {
            "name": "hurdle",
            "negative_ratio": 3.0,
            "structure_weight": 0.2,
            "variance_weight": 0.1,
        },
    )
    objective.loss.backward()
    print("device", device)
    print("loss", float(objective.loss.item()))
    print("masked_positive", objective.counts["masked_positive"])
    print("attention_positions", model.attention_positions)
    print("smoke_test=ok")


if __name__ == "__main__":
    main()
