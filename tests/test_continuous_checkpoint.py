from pathlib import Path

import torch

from sc2.train.checkpointing import atomic_torch_save, load_checkpoint


def test_manifest_mismatch_refuses_resume(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({
        "format": "sc2-continuous-checkpoint-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "manifest_sha256": "a",
        "global_step": 0,
        "next_sample_index": 0,
    }, path)
    try:
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            device=torch.device("cpu"),
            expected_manifest_sha256="b",
        )
    except ValueError as error:
        assert "Manifest changed" in str(error)
    else:
        raise AssertionError("Manifest mismatch should fail")
