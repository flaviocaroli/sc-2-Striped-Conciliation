from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sc2.data.csr_shard import shard_directory_sha256, write_csr
from sc2.data.shard_manifest import load_manifest
from sc2.data.sharded_expression_dataset import CounterBasedExpressionStream


def test_counter_stream_exact(tmp_path: Path) -> None:
    shard = tmp_path / "s"
    matrix = sparse.csr_matrix(np.array([[0, 1], [2, 0]], dtype=np.float32))
    write_csr(shard, "counts", matrix)
    write_csr(shard, "log1p", sparse.csr_matrix(np.log1p(matrix.toarray())))
    pd.DataFrame({"cell_id": ["a", "b"], "dataset_id": ["d", "d"], "donor_id": ["x", "y"], "tissue": ["t", "t"], "cell_type": ["c", "c"], "split": ["train", "train"]}).to_csv(shard / "obs.csv", index=False)
    (shard / "meta.json").write_text('{"schema_version":"sc2-csr-shard-v1","n_rows":2,"n_genes":2,"gene_vocab_sha256":"v"}')
    pd.DataFrame([{"shard_id": "s", "path": str(shard), "split": "train", "modality": "sc", "n_rows": 2, "n_genes": 2, "sha256": shard_directory_sha256(shard), "gene_vocab_sha256": "v", "census_release": "test"}]).to_csv(tmp_path / "m.csv", index=False)
    records, _ = load_manifest(tmp_path / "m.csv")
    stream = CounterBasedExpressionStream(records, seed=9, mask_rates=[0.5], mask_probabilities=[1.0])
    a = stream.sample_at(123)
    b = stream.sample_at(123)
    assert np.array_equal(a["x"].numpy(), b["x"].numpy())
    assert a["row"] == b["row"]


def test_missing_requested_modality_fails(tmp_path: Path) -> None:
    shard = tmp_path / "s"
    matrix = sparse.csr_matrix(np.array([[0, 1]], dtype=np.float32))
    write_csr(shard, "counts", matrix)
    write_csr(shard, "log1p", sparse.csr_matrix(np.log1p(matrix.toarray())))
    pd.DataFrame({"cell_id": ["a"], "dataset_id": ["d"], "donor_id": ["x"], "tissue": ["t"], "cell_type": ["c"], "split": ["train"]}).to_csv(shard / "obs.csv", index=False)
    (shard / "meta.json").write_text('{"schema_version":"sc2-csr-shard-v1","n_rows":1,"n_genes":2,"gene_vocab_sha256":"v"}')
    pd.DataFrame([{"shard_id": "s", "path": str(shard), "split": "train", "modality": "sc", "n_rows": 1, "n_genes": 2, "sha256": shard_directory_sha256(shard), "gene_vocab_sha256": "v", "census_release": "test"}]).to_csv(tmp_path / "m.csv", index=False)
    records, _ = load_manifest(tmp_path / "m.csv")
    try:
        CounterBasedExpressionStream(records, seed=1, modality_weights={"sc": 1.0, "bulk": 1.0})
    except ValueError as error:
        assert "missing shard modalities" in str(error)
    else:
        raise AssertionError("Missing requested modality should fail")
