import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "configs"
    / "extension_2026_p3"
    / "p3_16k_scaling_protocol_v1.json"
)


def load_protocol():
    assert PROTOCOL.is_file(), PROTOCOL
    with PROTOCOL.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_protocol_contract():
    p = load_protocol()

    assert p["protocol_version"] == "v1"

    data = p["data_contract"]
    assert data["target_gene_count"] == 16384
    assert data["vocabulary_source"] == "train_split_only"
    assert data["train_cells"] == 200000
    assert data["validation_cells"] == 25000
    assert data["test_cells"] == 25000
    assert data["held_out_feature_selection"] is False

    models = p["candidate_models"]
    assert set(models) == {
        "P3_16K_Small",
        "P3_16K_Large",
    }

    small = models["P3_16K_Small"]
    assert small["n_genes"] == 16384
    assert small["n_mamba_blocks"] == 20
    assert small["n_attention_checkpoints"] == 4
    assert small["d_model"] == 128
    assert small["d_state"] == 4
    assert small["top_k"] == 256
    assert small["n_heads"] == 4
    assert small["bidirectional_mamba"] is True

    large = models["P3_16K_Large"]
    assert large["n_genes"] == 16384
    assert large["n_mamba_blocks"] == 25
    assert large["n_attention_checkpoints"] == 5
    assert large["d_model"] == 192
    assert large["d_state"] == 4
    assert large["top_k"] == 256
    assert large["n_heads"] == 4
    assert large["bidirectional_mamba"] is True

    train = p["training_contract"]
    assert train["total_steps"] == 5000
    assert train["scientific_checkpoint_step"] == 5000
    assert train["effective_batch_size"] == 32
    assert train["seeds"] == [
        20260728,
        20260729,
        20260730,
        20260731,
        20260801,
    ]

    validation = p["validation_contract"]
    assert validation["architecture_selection_uses"] == "internal_validation_only"
    assert validation["internal_test_locked_until_model_and_threshold_freeze"] is True
    assert validation["baron_locked_until_model_and_threshold_freeze"] is True
    assert validation["zheng68k_locked_until_model_and_threshold_freeze"] is True
    assert validation["mask_rates"] == [0.15, 0.30, 0.50]

    threshold = validation["repair_threshold_rule"]
    assert (
        threshold["eligibility"]
        == "originally_zero_fill_le_0.02_in_every_validation_seed"
    )

    hpc = p["hpc_contract"]
    assert hpc["maximum_submitted_job_elements_at_one_time"] == 29

    paper = p["paper_contract"]
    assert paper["P2_results_remain_frozen"] is True
    assert paper["main_paper_requires_exact_measured_parameter_counts"] is True


if __name__ == "__main__":
    test_protocol_contract()
    print("P3_16K_PROTOCOL_CONTRACT=PASS")
