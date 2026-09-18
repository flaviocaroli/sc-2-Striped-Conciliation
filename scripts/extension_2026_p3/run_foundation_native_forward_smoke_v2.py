#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_bundle(path: Path):
    with np.load(
        path,
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

        feature_space = int(
            np.asarray(
                z["source_feature_space"]
            ).item()
        )

    assert counts.shape == (16, 16384)
    assert genes.shape == (16384,)
    assert rows.shape == (16,)
    assert feature_space == 16384
    assert np.all(counts >= 0)

    return counts, genes, rows


def finite_array(x):
    a = np.asarray(x)

    if (
        np.issubdtype(
            a.dtype,
            np.number,
        )
        and a.size
    ):
        assert np.isfinite(a).all()

    return a


def run_geneformer(
    input_path: Path,
    checkpoint: Path,
    output: Path,
):
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    import torch

    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer
    from transformers import AutoModel

    counts, genes, rows = load_bundle(
        input_path
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    with tempfile.TemporaryDirectory(
        prefix="sc2-geneformer-"
    ) as td:

        root = Path(td)

        input_dir = root / "input"
        token_dir = root / "tokens"

        input_dir.mkdir()
        token_dir.mkdir()

        obs = pd.DataFrame(
            index=[
                f"p3_validation_{int(x)}"
                for x in rows
            ]
        )

        obs["n_counts"] = counts.sum(
            axis=1
        ).astype(np.int64)

        var = pd.DataFrame(
            index=genes
        )

        var["ensembl_id"] = genes

        adata = ad.AnnData(
            X=sparse.csr_matrix(
                counts
            ),
            obs=obs,
            var=var,
        )

        adata.write_h5ad(
            input_dir / "p3_first16.h5ad"
        )

        gf_median = Path(
            os.environ["GENEFORMER_GENE_MEDIAN_FILE"]
        )
        gf_tokens = Path(
            os.environ["GENEFORMER_TOKEN_DICTIONARY_FILE"]
        )
        gf_mapping = Path(
            os.environ["GENEFORMER_GENE_MAPPING_FILE"]
        )

        for asset in (
            gf_median,
            gf_tokens,
            gf_mapping,
        ):
            assert asset.is_file()
            assert asset.stat().st_size > 100

        tokenizer = TranscriptomeTokenizer(
            nproc=1,
            chunk_size=16,
            model_input_size=4096,
            special_token=True,
            collapse_gene_ids=True,
            use_h5ad_index=False,
            keep_counts=False,
            model_version="V2",
            gene_median_file=gf_median,
            token_dictionary_file=gf_tokens,
            gene_mapping_file=gf_mapping,
        )

        tokenizer.tokenize_data(
            input_dir,
            token_dir,
            "p3_16k_first16",
            file_format="h5ad",
        )

        preferred = (
            token_dir
            / "p3_16k_first16.dataset"
        )

        candidates = []

        if preferred.exists():
            candidates.append(
                preferred
            )

        for p in token_dir.iterdir():
            if (
                p.is_dir()
                and (
                    p / "dataset_info.json"
                ).exists()
                and p not in candidates
            ):
                candidates.append(p)

        assert len(candidates) == 1, [
            str(x)
            for x in candidates
        ]

        ds = load_from_disk(
            str(candidates[0])
        )

        assert len(ds) == 16

        sequences = [
            list(
                map(
                    int,
                    ds[i]["input_ids"],
                )
            )
            for i in range(len(ds))
        ]

    lengths = np.asarray(
        [len(x) for x in sequences],
        dtype=np.int32,
    )

    assert np.all(lengths > 0)
    assert np.all(lengths <= 4096)

    device = torch.device("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModel.from_pretrained(
        checkpoint,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    cell_embs = []

    with torch.inference_mode():

        for sequence in sequences:

            ids = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            out = model(
                input_ids=ids,
            )

            hidden = out.last_hidden_state

            assert hidden.ndim == 3
            assert hidden.shape[0] == 1
            assert hidden.shape[1] == len(
                sequence
            )

            emb = (
                hidden[0]
                .float()
                .mean(dim=0)
                .cpu()
                .numpy()
            )

            finite_array(emb)

            cell_embs.append(emb)

    embeddings = np.stack(
        cell_embs,
        axis=0,
    ).astype(np.float32)

    finite_array(embeddings)

    np.save(
        output / "cell_embeddings.npy",
        embeddings,
    )

    np.save(
        output / "sequence_lengths.npy",
        lengths,
    )

    return {
        "role": "representation_smoke",
        "source_feature_space_genes": 16384,
        "native_context_limit": 4096,
        "cells": 16,
        "sequence_length_min":
            int(lengths.min()),
        "sequence_length_max":
            int(lengths.max()),
        "embedding_shape":
            list(embeddings.shape),
        "peak_gpu_memory_mib":
            int(
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            ),
    }


def run_genemamba(
    input_path: Path,
    checkpoint: Path,
    output: Path,
):
    import torch

    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
    )

    counts, genes, rows = load_bundle(
        input_path
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    config = AutoConfig.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )

    vocab = tokenizer.get_vocab()

    exact_token_ids = np.full(
        16384,
        -1,
        dtype=np.int64,
    )

    for j, gene in enumerate(genes):

        if gene in vocab:
            exact_token_ids[j] = int(
                vocab[gene]
            )

    mapped = exact_token_ids >= 0
    mapped_count = int(mapped.sum())

    print(
        "GENEMAMBA_EXACT_P3_VOCAB_OVERLAP=",
        mapped_count,
    )

    assert mapped_count > 0

    context = int(
        config.max_position_embeddings
    )

    assert context == 2048

    sequences = []

    for i in range(16):

        candidates = np.flatnonzero(
            (counts[i] > 0)
            & mapped
        )

        assert candidates.size > 0

        order = np.argsort(
            -counts[i, candidates],
            kind="stable",
        )

        selected = candidates[
            order[:context]
        ]

        seq = exact_token_ids[
            selected
        ].astype(np.int64)

        assert seq.size > 0
        assert seq.size <= context
        assert np.all(seq >= 0)

        sequences.append(seq)

    lengths = np.asarray(
        [len(x) for x in sequences],
        dtype=np.int32,
    )

    device = torch.device("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    cell_embs = []

    with torch.inference_mode():

        for sequence in sequences:

            ids = torch.from_numpy(
                sequence
            ).to(
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)

            attention = torch.ones_like(
                ids,
                dtype=torch.long,
            )

            out = model(
                input_ids=ids,
                attention_mask=attention,
                output_hidden_states=True,
            )

            hidden = getattr(
                out,
                "last_hidden_state",
                None,
            )

            if hidden is None:
                states = getattr(
                    out,
                    "hidden_states",
                    None,
                )

                assert states is not None
                assert len(states) > 0

                hidden = states[-1]

            assert hidden.ndim == 3
            assert hidden.shape[0] == 1
            assert hidden.shape[1] == len(
                sequence
            )

            emb = (
                hidden[0]
                .float()
                .mean(dim=0)
                .cpu()
                .numpy()
            )

            finite_array(emb)

            cell_embs.append(emb)

    embeddings = np.stack(
        cell_embs,
        axis=0,
    ).astype(np.float32)

    finite_array(embeddings)

    np.save(
        output / "cell_embeddings.npy",
        embeddings,
    )

    np.save(
        output / "sequence_lengths.npy",
        lengths,
    )

    return {
        "role": "representation_smoke",
        "source_feature_space_genes": 16384,
        "native_vocab_size":
            int(config.vocab_size),
        "native_context_limit":
            context,
        "exact_p3_vocab_overlap":
            mapped_count,
        "cells": 16,
        "sequence_length_min":
            int(lengths.min()),
        "sequence_length_max":
            int(lengths.max()),
        "embedding_shape":
            list(embeddings.shape),
        "peak_gpu_memory_mib":
            int(
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            ),
    }


def describe_result(value):
    import torch

    if isinstance(value, torch.Tensor):

        a = (
            value.detach()
            .float()
            .cpu()
            .numpy()
        )

        finite_array(a)

        return {
            "type": "torch.Tensor",
            "shape": list(a.shape),
            "dtype": str(a.dtype),
        }

    if isinstance(value, np.ndarray):

        finite_array(value)

        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, (list, tuple)):

        return {
            "type":
                type(value).__name__,
            "length":
                len(value),
            "items": [
                describe_result(x)
                for x in value[:12]
            ],
        }

    if isinstance(value, dict):

        return {
            "type": "dict",
            "keys":
                list(value.keys())[:30],
            "items": {
                str(k):
                    describe_result(v)
                for k, v in list(
                    value.items()
                )[:12]
            },
        }

    if hasattr(value, "shape"):

        return {
            "type":
                type(value).__name__,
            "shape":
                list(value.shape),
        }

    return {
        "type":
            type(value).__name__,
        "repr":
            repr(value)[:500],
    }


def run_scprint(
    input_path: Path,
    checkpoint: Path,
    output: Path,
):
    import anndata as ad
    import pandas as pd
    from scipy import sparse
    from scdataloader import Preprocessor

    import torch

    import scprint
    from scprint import scPrint
    from scprint.tasks import Denoiser

    counts, genes, rows = load_bundle(
        input_path
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    obs = pd.DataFrame(
        index=[
            f"p3_validation_{int(x)}"
            for x in rows
        ]
    )

    obs[
        "organism_ontology_term_id"
    ] = "NCBITaxon:9606"

    var = pd.DataFrame(
        index=genes
    )

    var["ensembl_id"] = genes
    var["feature_id"] = genes

    adata = ad.AnnData(
        X=sparse.csr_matrix(
            counts
        ),
        obs=obs,
        var=var,
    )

    assert adata.n_obs == 16
    assert adata.n_vars == 16384

    preprocessor = Preprocessor(
        do_postp=False,
    )

    adata = preprocessor(
        adata
    )

    assert adata.n_obs == 16
    assert adata.n_vars > 0

    print(
        "SCPRINT_POST_PREPROCESS_GENES=",
        adata.n_vars,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = scPrint.load_from_checkpoint(
        str(checkpoint),
        precpt_gene_emb=None,
        transformer="normal",
        map_location="cpu",
    )

    model = model.cuda()
    model.eval()

    denoiser = Denoiser(
        batch_size=2,
        num_workers=0,
        max_len=5000,
        precision="16-mixed",
        how="most var",
        max_cells=16,
        doplot=False,
        dtype=torch.float16,
    )

    result = denoiser(
        model,
        adata,
    )

    description = describe_result(
        result
    )

    (output / "result_structure.json").write_text(
        json.dumps(
            description,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return {
        "role": "official_denoiser_feasibility_smoke",
        "source_feature_space_genes": 16384,
        "native_max_len": 5000,
        "cells": 16,
        "scprint_version":
            getattr(
                scprint,
                "__version__",
                None,
            ),
        "result_structure":
            description,
        "peak_gpu_memory_mib":
            int(
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "Geneformer",
            "GeneMamba",
            "scPRINT",
        ],
    )

    parser.add_argument(
        "--input-bundle",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    input_path = Path(
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

    assert input_path.is_file()
    assert source.exists()
    assert not output.exists()

    if args.model == "Geneformer":

        assert checkpoint.is_dir()

        metrics = run_geneformer(
            input_path,
            checkpoint,
            output,
        )

        checkpoint_file = (
            checkpoint
            / "model.safetensors"
        )

    elif args.model == "GeneMamba":

        assert checkpoint.is_dir()

        metrics = run_genemamba(
            input_path,
            checkpoint,
            output,
        )

        checkpoint_file = (
            checkpoint
            / "model.safetensors"
        )

    else:

        assert checkpoint.is_file()

        metrics = run_scprint(
            input_path,
            checkpoint,
            output,
        )

        checkpoint_file = checkpoint

    assert checkpoint_file.is_file()

    payload = {
        "schema":
            "sc2-p3-foundation-native-forward-smoke-v1",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model":
            args.model,

        "input_bundle":
            str(input_path),

        "input_bundle_sha256":
            sha256(input_path),

        "source":
            str(source),

        "checkpoint":
            str(checkpoint),

        "checkpoint_file":
            str(checkpoint_file),

        "checkpoint_sha256":
            sha256(checkpoint_file),

        "scientific_result":
            False,

        "custom_decoder":
            False,

        "ad_hoc_inverse":
            False,

        "status":
            "PASS",

        **metrics,
    }

    receipt = (
        output
        / "smoke_receipt.json"
    )

    receipt.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            payload,
            sort_keys=True,
        )
    )

    print(
        f"{args.model.upper()}_NATIVE_GPU_FORWARD_SMOKE=PASS"
    )


if __name__ == "__main__":
    main()
