#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_bundle(path: Path):
    with np.load(path, allow_pickle=False) as z:
        required = {
            "counts",
            "ensembl_id",
            "source_row",
            "cell_id",
            "cell_type",
            "donor",
            "available_gene_mask",
        }

        if not required <= set(z.files):
            raise RuntimeError(
                f"bundle missing {sorted(required-set(z.files))}"
            )

        counts = np.asarray(z["counts"], dtype=np.int32)
        genes = np.asarray(z["ensembl_id"]).astype(str)
        rows = np.asarray(z["source_row"], dtype=np.int64)
        cell_id = np.asarray(z["cell_id"]).astype(str)
        cell_type = np.asarray(z["cell_type"]).astype(str)
        donor = np.asarray(z["donor"]).astype(str)
        available = np.asarray(
            z["available_gene_mask"],
            dtype=bool,
        )

    assert counts.shape == (5000, 16384)
    assert genes.shape == (16384,)
    assert rows.shape == (5000,)
    assert cell_id.shape == (5000,)
    assert cell_type.shape == (5000,)
    assert donor.shape == (5000,)
    assert available.shape == (16384,)

    assert np.all(counts >= 0)
    assert len(np.unique(cell_id)) == 5000

    assert set(donor) == {
        "human1",
        "human2",
        "human3",
        "human4",
    }

    return (
        counts,
        genes,
        rows,
        cell_id,
        cell_type,
        donor,
        available,
    )


def run_geneformer(
    counts,
    genes,
    rows,
    cell_id,
    cell_type,
    donor,
    cfg,
):
    import anndata as ad
    import pandas as pd
    import torch

    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer
    from scipy import sparse
    from transformers import AutoModel

    checkpoint = Path(cfg["checkpoint"])

    aux = cfg["auxiliary"]

    median = Path(aux["gene_median"]["path"])
    tokens = Path(aux["token_dictionary"]["path"])
    mapping = Path(aux["gene_mapping"]["path"])

    assert sha(median) == aux["gene_median"]["sha256"]
    assert sha(tokens) == aux["token_dictionary"]["sha256"]
    assert sha(mapping) == aux["gene_mapping"]["sha256"]

    with tempfile.TemporaryDirectory(
        prefix="sc2-geneformer-baron-"
    ) as td:

        root = Path(td)
        inp = root / "input"
        tok = root / "tokens"

        inp.mkdir()
        tok.mkdir()

        obs = pd.DataFrame(
            {
                "n_counts":
                    counts.sum(axis=1).astype(np.int64),
                "source_row": rows,
                "cell_id": cell_id,
                "donor": donor,
                "cell_type": cell_type,
            },
            index=[
                f"sc2_baron_{i:05d}"
                for i in range(5000)
            ],
        )

        var = pd.DataFrame(index=genes)
        var["ensembl_id"] = genes

        adata = ad.AnnData(
            X=sparse.csr_matrix(counts),
            obs=obs,
            var=var,
        )

        adata.write_h5ad(inp / "baron5000.h5ad")

        attrs = {
            "source_row": "source_row",
            "cell_id": "cell_id",
            "donor": "donor",
            "cell_type": "cell_type",
        }

        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict=attrs,
            nproc=1,
            chunk_size=256,
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
            "baron5000",
            file_format="h5ad",
        )

        roots = sorted({
            p.parent
            for p in tok.rglob("dataset_info.json")
        })

        if len(roots) != 1:
            raise RuntimeError(
                f"unexpected tokenized datasets {roots}"
            )

        ds = load_from_disk(str(roots[0]))

        assert len(ds) == 5000

        ds_rows = np.asarray(
            ds["source_row"],
            dtype=np.int64,
        )

        ds_cell = np.asarray(
            ds["cell_id"]
        ).astype(str)

        ds_donor = np.asarray(
            ds["donor"]
        ).astype(str)

        ds_type = np.asarray(
            ds["cell_type"]
        ).astype(str)

        if not np.array_equal(ds_rows, rows):
            raise RuntimeError(
                "Geneformer source_row order changed"
            )

        if not np.array_equal(ds_cell, cell_id):
            raise RuntimeError(
                "Geneformer cell_id order changed"
            )

        if not np.array_equal(ds_donor, donor):
            raise RuntimeError(
                "Geneformer donor order changed"
            )

        if not np.array_equal(ds_type, cell_type):
            raise RuntimeError(
                "Geneformer cell_type order changed"
            )

        sequences = [
            list(map(int, ds[i]["input_ids"]))
            for i in range(5000)
        ]

    lengths = np.asarray(
        [len(x) for x in sequences],
        dtype=np.int32,
    )

    assert np.all(lengths > 0)
    assert np.all(lengths <= 4096)

    device = torch.device("cuda:0")

    model = AutoModel.from_pretrained(
        checkpoint,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)

    result = []

    with torch.inference_mode():

        for i, seq in enumerate(sequences):

            ids = torch.tensor(
                [seq],
                dtype=torch.long,
                device=device,
            )

            attention = torch.ones_like(ids)

            hidden = model(
                input_ids=ids,
                attention_mask=attention,
            ).last_hidden_state

            assert hidden.shape[:2] == (
                1,
                len(seq),
            )

            emb = (
                hidden[0]
                .float()
                .mean(dim=0)
                .cpu()
                .numpy()
            )

            if not np.isfinite(emb).all():
                raise RuntimeError(
                    "nonfinite Geneformer embedding"
                )

            result.append(emb)

            if (
                (i + 1) % 250 == 0
                or i + 1 == 5000
            ):
                print(
                    f"GENEFORMER_EMBEDDED={i+1}/5000",
                    flush=True,
                )

    embedding = np.stack(result).astype(np.float32)

    assert embedding.shape == (5000, 768)

    return (
        embedding,
        lengths,
        {
            "native_context_limit": 4096,
            "pooling":
                "mean_last_hidden_state",
            "tokenized_row_identity_exact": True,
            "tokenized_cell_id_identity_exact": True,
            "peak_gpu_memory_mib":
                float(
                    torch.cuda.max_memory_allocated(device)
                    / 1024**2
                ),
        },
    )


def run_genemamba(
    counts,
    genes,
    cfg,
):
    import torch

    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
    )

    checkpoint = Path(cfg["checkpoint"])

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

    token_ids = np.full(
        16384,
        -1,
        dtype=np.int64,
    )

    for j, gene in enumerate(genes):
        if gene in vocab:
            token_ids[j] = int(vocab[gene])

    mapped = token_ids >= 0
    mapped_count = int(mapped.sum())

    expected_overlap = int(
        cfg["exact_p3_vocab_overlap"]
    )

    if mapped_count != expected_overlap:
        raise RuntimeError(
            f"GeneMamba overlap changed: "
            f"{mapped_count} != {expected_overlap}"
        )

    context = int(
        config.max_position_embeddings
    )

    assert context == int(
        cfg["native_context_limit"]
    )

    device = torch.device("cuda:0")

    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)

    result = []
    lengths = np.zeros(5000, dtype=np.int32)

    with torch.inference_mode():

        for i in range(5000):

            candidates = np.flatnonzero(
                (counts[i] > 0) & mapped
            )

            if candidates.size == 0:
                raise RuntimeError(
                    f"GeneMamba empty cell {i}"
                )

            order = np.argsort(
                -counts[i, candidates],
                kind="stable",
            )

            selected = candidates[
                order[:context]
            ]

            seq = token_ids[selected]

            lengths[i] = len(seq)

            ids = torch.from_numpy(
                seq.astype(np.int64)
            ).to(
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)

            attention = torch.ones_like(ids)

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

                if not states:
                    raise RuntimeError(
                        "GeneMamba hidden states missing"
                    )

                hidden = states[-1]

            emb = (
                hidden[0]
                .float()
                .mean(dim=0)
                .cpu()
                .numpy()
            )

            if not np.isfinite(emb).all():
                raise RuntimeError(
                    "nonfinite GeneMamba embedding"
                )

            result.append(emb)

            if (
                (i + 1) % 250 == 0
                or i + 1 == 5000
            ):
                print(
                    f"GENEMAMBA_EMBEDDED={i+1}/5000",
                    flush=True,
                )

    embedding = np.stack(result).astype(np.float32)

    assert embedding.shape == (5000, 512)
    assert np.all(lengths > 0)
    assert np.all(lengths <= context)

    return (
        embedding,
        lengths,
        {
            "native_vocab_size":
                int(config.vocab_size),
            "native_context_limit":
                context,
            "exact_p3_vocab_overlap":
                mapped_count,
            "pooling":
                "mean_last_hidden_state",
            "peak_gpu_memory_mib":
                float(
                    torch.cuda.max_memory_allocated(device)
                    / 1024**2
                ),
        },
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--protocol",
        required=True,
    )

    ap.add_argument(
        "--model",
        required=True,
        choices=[
            "Geneformer",
            "GeneMamba",
        ],
    )

    ap.add_argument(
        "--check-only",
        action="store_true",
    )

    args = ap.parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    p = json.loads(
        protocol_path.read_text()
    )

    expected = os.environ.get(
        "SC2_FOUNDATION_REP_PROTOCOL_SHA"
    )

    if expected and sha(protocol_path) != expected:
        raise RuntimeError(
            "representation protocol SHA mismatch"
        )

    model_cfg = p["models"][args.model]

    bundle = Path(
        p["input_bundle"]["path"]
    )

    if sha(bundle) != \
            p["input_bundle"]["sha256"]:
        raise RuntimeError(
            "input bundle SHA mismatch"
        )

    source = Path(
        model_cfg["source"]
    )

    checkpoint = Path(
        model_cfg["checkpoint"]
    )

    checkpoint_file = Path(
        model_cfg["checkpoint_file"]
    )

    if git_head(source) != \
            model_cfg["source_commit"]:
        raise RuntimeError(
            f"{args.model} source commit mismatch"
        )

    if sha(checkpoint_file) != \
            model_cfg["checkpoint_sha256"]:
        raise RuntimeError(
            f"{args.model} checkpoint SHA mismatch"
        )

    (
        counts,
        genes,
        rows,
        cell_id,
        cell_type,
        donor,
        available,
    ) = load_bundle(bundle)

    assert int(available.sum()) == 13762

    if args.check_only:
        print(
            f"{args.model.upper()}_BARON_PRECHECK=PASS"
        )
        return

    output = Path(
        model_cfg["output_dir"]
    )

    if output.exists():
        raise RuntimeError(
            f"output already exists: {output}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = output.parent / (
        f".tmp_{output.name}_{os.getpid()}"
    )

    if temp.exists():
        raise RuntimeError(
            f"temp output collision: {temp}"
        )

    temp.mkdir()

    try:

        if args.model == "Geneformer":

            (
                embedding,
                lengths,
                extra,
            ) = run_geneformer(
                counts,
                genes,
                rows,
                cell_id,
                cell_type,
                donor,
                model_cfg,
            )

        else:

            (
                embedding,
                lengths,
                extra,
            ) = run_genemamba(
                counts,
                genes,
                model_cfg,
            )

        if not np.isfinite(embedding).all():
            raise RuntimeError(
                "nonfinite final embeddings"
            )

        expected_dim = int(
            model_cfg["embedding_dim"]
        )

        assert embedding.shape == (
            5000,
            expected_dim,
        )

        np.savez(
            temp / "embedding.npz",
            embedding=embedding,
            sequence_length=lengths,
            cell_id=cell_id,
            source_row=rows,
            donor=donor,
            cell_type=cell_type,
            model=np.asarray(args.model),
            dataset=np.asarray(
                "baron_pancreas"
            ),
        )

        receipt = {
            "schema":
                "sc2-p3-foundation-baron-embedding-v1",

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "status":
                "PASS",

            "scientific_result":
                True,

            "role":
                "representation_downstream",

            "model":
                args.model,

            "dataset":
                "baron_pancreas",

            "cells":
                5000,

            "source_feature_space_genes":
                16384,

            "embedding_shape":
                list(embedding.shape),

            "embedding_dim":
                expected_dim,

            "sequence_length_min":
                int(lengths.min()),

            "sequence_length_max":
                int(lengths.max()),

            "input_bundle":
                str(bundle),

            "input_bundle_sha256":
                sha(bundle),

            "checkpoint":
                str(checkpoint),

            "checkpoint_sha256":
                sha(checkpoint_file),

            "source":
                str(source),

            "source_commit":
                git_head(source),

            "cell_id_exact":
                True,

            "source_row_exact":
                True,

            "donor_exact":
                True,

            "cell_type_exact":
                True,

            "custom_decoder":
                False,

            "ad_hoc_inverse":
                False,

            **extra,
        }

        receipt_path = (
            temp / "receipt.json"
        )

        receipt_path.write_text(
            json.dumps(
                receipt,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n"
        )

        files = [
            temp / "embedding.npz",
            receipt_path,
        ]

        (
            temp / "SHA256SUMS.txt"
        ).write_text(
            "".join(
                f"{sha(x)}  {x.name}\n"
                for x in files
            )
        )

        os.replace(
            temp,
            output,
        )

    except Exception:
        shutil.rmtree(
            temp,
            ignore_errors=True,
        )
        raise

    print(
        json.dumps(
            receipt,
            sort_keys=True,
        )
    )

    print(
        f"{args.model.upper()}_BARON_EMBEDDING=PASS"
    )


if __name__ == "__main__":
    main()
