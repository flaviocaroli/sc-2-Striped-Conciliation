#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-bundle",
        required=True,
    )

    p.add_argument(
        "--checkpoint",
        required=True,
    )

    p.add_argument(
        "--source",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    args = p.parse_args()

    import anndata as ad
    import pandas as pd
    import torch

    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer
    from scipy import sparse
    from transformers import AutoModel

    bundle = Path(
        args.input_bundle
    ).resolve()

    checkpoint = Path(
        args.checkpoint
    ).resolve()

    source = Path(
        args.source
    ).resolve()

    output = Path(
        args.output_dir
    ).resolve()

    assert not output.exists()
    assert torch.cuda.is_available()

    median = Path(
        os.environ[
            "GENEFORMER_GENE_MEDIAN_FILE"
        ]
    )

    tokens = Path(
        os.environ[
            "GENEFORMER_TOKEN_DICTIONARY_FILE"
        ]
    )

    mapping = Path(
        os.environ[
            "GENEFORMER_GENE_MAPPING_FILE"
        ]
    )

    for path in (
        bundle,
        checkpoint / "model.safetensors",
        median,
        tokens,
        mapping,
    ):
        assert path.is_file()
        assert path.stat().st_size > 100

    with np.load(
        bundle,
        allow_pickle=False,
    ) as z:

        counts = np.asarray(
            z["counts"],
            dtype=np.int32,
        )

        genes = np.asarray(
            z["ensembl_id"]
        ).astype(str)

        rows = np.asarray(
            z["source_row"],
            dtype=np.int64,
        )

    assert counts.shape == (
        16,
        16384,
    )

    assert genes.shape == (
        16384,
    )

    embeddings = []
    lengths = []

    with tempfile.TemporaryDirectory(
        prefix="sc2-geneformer-final-"
    ) as td:

        root = Path(td)
        inp = root / "input"
        tok = root / "tokens"

        inp.mkdir()
        tok.mkdir()

        obs = pd.DataFrame(
            index=[
                f"p3_validation_{int(x)}"
                for x in rows
            ]
        )

        obs[
            "n_counts"
        ] = counts.sum(
            axis=1
        ).astype(np.int64)

        var = pd.DataFrame(
            index=genes
        )

        var[
            "ensembl_id"
        ] = genes

        adata = ad.AnnData(
            X=sparse.csr_matrix(
                counts
            ),
            obs=obs,
            var=var,
        )

        adata.write_h5ad(
            inp / "cells.h5ad"
        )

        tokenizer = TranscriptomeTokenizer(
            nproc=1,
            chunk_size=16,
            model_input_size=4096,
            special_token=True,
            collapse_gene_ids=True,
            use_h5ad_index=False,
            keep_counts=False,
            model_version="V2",
            gene_median_file=median,
            token_dictionary_file=tokens,
            gene_mapping_file=mapping,
        )

        tokenizer.tokenize_data(
            inp,
            tok,
            "p3_final",
            file_format="h5ad",
        )

        roots = sorted(
            {
                p.parent
                for p in tok.rglob(
                    "dataset_info.json"
                )
            }
        )

        assert len(roots) == 1, roots

        ds = load_from_disk(
            str(roots[0])
        )

        assert len(ds) == 16

        device = torch.device(
            "cuda:0"
        )

        model = AutoModel.from_pretrained(
            checkpoint,
            local_files_only=True,
        )

        model = model.to(
            device
        )

        model.eval()

        torch.cuda.reset_peak_memory_stats(
            device
        )

        with torch.inference_mode():

            for i in range(16):

                seq = list(
                    map(
                        int,
                        ds[i][
                            "input_ids"
                        ],
                    )
                )

                assert (
                    0
                    < len(seq)
                    <= 4096
                )

                lengths.append(
                    len(seq)
                )

                ids = torch.tensor(
                    [seq],
                    dtype=torch.long,
                    device=device,
                )

                mask = torch.ones_like(
                    ids
                )

                hidden = model(
                    input_ids=ids,
                    attention_mask=mask,
                ).last_hidden_state

                assert hidden.shape[
                    0
                ] == 1

                assert hidden.shape[
                    1
                ] == len(seq)

                assert torch.isfinite(
                    hidden
                ).all()

                emb = hidden.mean(
                    dim=1
                )

                embeddings.append(
                    emb.squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )

    embedding = np.stack(
        embeddings,
        axis=0,
    )

    assert embedding.shape[0] == 16
    assert np.isfinite(
        embedding
    ).all()

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024.0
        / 1024.0
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    np.save(
        output / "cell_embeddings.npy",
        embedding,
    )

    source_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    receipt = {
        "schema":
            "sc2-p3-foundation-geneformer-native-smoke-final-v1",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "PASS",

        "model":
            "Geneformer",

        "role":
            "representation_smoke",

        "scientific_result":
            False,

        "source_feature_space_genes":
            16384,

        "cells":
            16,

        "native_context_limit":
            4096,

        "embedding_shape":
            list(
                embedding.shape
            ),

        "sequence_length_min":
            int(min(lengths)),

        "sequence_length_max":
            int(max(lengths)),

        "input_bundle":
            str(bundle),

        "input_bundle_sha256":
            sha256(bundle),

        "checkpoint":
            str(checkpoint),

        "checkpoint_sha256":
            sha256(
                checkpoint
                / "model.safetensors"
            ),

        "source":
            str(source),

        "source_commit":
            source_commit,

        "gene_median_sha256":
            sha256(median),

        "token_dictionary_sha256":
            sha256(tokens),

        "gene_mapping_sha256":
            sha256(mapping),

        "pooling":
            "mean_last_hidden_state",

        "custom_decoder":
            False,

        "ad_hoc_inverse":
            False,

        "peak_gpu_memory_mib":
            peak,
    }

    (
        output
        / "smoke_receipt.json"
    ).write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            receipt,
            sort_keys=True,
        )
    )

    print(
        "GENEFORMER_NATIVE_GPU_FORWARD_SMOKE=PASS"
    )


if __name__ == "__main__":
    main()
