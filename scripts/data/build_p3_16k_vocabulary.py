#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def semantic_hash_candidates(ids):
    payloads = {
        "newline_no_trailing":
            "\n".join(ids).encode(),
        "newline_trailing":
            ("\n".join(ids) + "\n").encode(),
        "concat":
            "".join(ids).encode(),
        "json_compact":
            json.dumps(ids, separators=(",", ":")).encode(),
        "json_default":
            json.dumps(ids).encode(),
        "index_tab_newline":
            "".join(
                f"{i}\t{g}\n"
                for i, g in enumerate(ids)
            ).encode(),
    }

    return {
        name: hashlib.sha256(data).hexdigest()
        for name, data in payloads.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--p2-vocabulary", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--receipt", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    stats_path = Path(args.stats)
    p2_vocab_path = Path(args.p2_vocabulary)
    out_path = Path(args.output)
    receipt_path = Path(args.receipt)

    cfg = yaml.safe_load(
        cfg_path.read_text(encoding="utf-8")
    )

    assert cfg["target_gene_count"] == 16384
    assert cfg["vocabulary_source"] == "train_split_only"

    stats = pd.read_parquet(stats_path)
    p2 = pd.read_parquet(p2_vocab_path)

    if len(p2) != 4096:
        raise SystemExit(
            f"P2 vocabulary has {len(p2)} rows, expected 4096"
        )

    # --------------------------------------------------------
    # Resolve exact columns without inventing names.
    # --------------------------------------------------------

    gene_candidates = [
        "ensembl_id",
        "feature_id",
        "gene_id",
    ]

    gene_col = next(
        (c for c in gene_candidates if c in stats.columns),
        None,
    )

    if gene_col is None:
        raise SystemExit(
            f"Cannot resolve gene ID column: {list(stats.columns)}"
        )

    detection_exact = [
        "detected_train_cells",
        "detected_cells",
        "n_detected",
        "n_cells_detected",
        "detection_count",
    ]

    detection_cols = [
        c for c in detection_exact
        if c in stats.columns
    ]

    if not detection_cols:
        detection_cols = [
            c for c in stats.columns
            if "detect" in c.lower()
            and pd.api.types.is_numeric_dtype(stats[c])
        ]

    if len(detection_cols) != 1:
        raise SystemExit(
            f"Ambiguous detection columns: {detection_cols}"
        )

    detection_col = detection_cols[0]

    variance_exact = [
        "variance_log1p",
        "var_log1p",
        "log1p_variance",
        "log1p_var",
    ]

    variance_cols = [
        c for c in variance_exact
        if c in stats.columns
    ]

    if not variance_cols:
        variance_cols = [
            c for c in stats.columns
            if "var" in c.lower()
            and "log" in c.lower()
            and pd.api.types.is_numeric_dtype(stats[c])
        ]

    if len(variance_cols) != 1:
        raise SystemExit(
            f"Ambiguous log1p variance columns: {variance_cols}"
        )

    variance_col = variance_cols[0]

    # Frozen P2 eligibility rule.
    min_detected = 100

    eligible = stats.loc[
        stats[detection_col] >= min_detected
    ].copy()

    if len(eligible) < 16384:
        raise SystemExit(
            "Only "
            f"{len(eligible)} eligible genes; "
            "cannot construct 16,384-gene vocabulary."
        )

    eligible[gene_col] = eligible[gene_col].astype(str)

    if eligible[gene_col].duplicated().any():
        raise SystemExit(
            "Eligible training statistics contain duplicate gene IDs."
        )

    p2_ids = p2["ensembl_id"].astype(str).tolist()

    # --------------------------------------------------------
    # Determine tie behavior empirically.
    #
    # Every candidate uses exactly the frozen scientific rule:
    #   detected >=100
    #   descending log1p variance
    #
    # Candidates differ only in deterministic tie handling.
    # We accept a rule ONLY if its first 4096 genes reproduce
    # the frozen P2 vocabulary exactly.
    # --------------------------------------------------------

    specs = [
        (
            "variance_desc_stable",
            [variance_col],
            [False],
        ),
        (
            "variance_desc_gene_asc",
            [variance_col, gene_col],
            [False, True],
        ),
        (
            "variance_desc_detection_desc_stable",
            [variance_col, detection_col],
            [False, False],
        ),
        (
            "variance_desc_detection_desc_gene_asc",
            [variance_col, detection_col, gene_col],
            [False, False, True],
        ),
    ]

    matches = []

    for name, cols, ascending in specs:
        ordered = eligible.sort_values(
            cols,
            ascending=ascending,
            kind="mergesort",
        )

        ids = ordered[gene_col].tolist()

        if ids[:4096] == p2_ids:
            matches.append(
                (name, ids, ordered)
            )

    if not matches:
        # Useful diagnostic only; do not produce P3 output.
        best = 0
        best_name = None

        for name, cols, ascending in specs:
            ids = (
                eligible
                .sort_values(
                    cols,
                    ascending=ascending,
                    kind="mergesort",
                )[gene_col]
                .tolist()
            )

            prefix = 0
            for a, b in zip(ids, p2_ids):
                if a != b:
                    break
                prefix += 1

            if prefix > best:
                best = prefix
                best_name = name

        raise SystemExit(
            "No candidate reproduces frozen P2 vocabulary exactly. "
            f"Best={best_name}, exact_prefix={best}/4096"
        )

    # If multiple tie rules reproduce 4K, they must also produce
    # the same 16K ordering or we refuse to choose arbitrarily.
    sixteen_lists = [
        x[1][:16384]
        for x in matches
    ]

    first = sixteen_lists[0]

    for other in sixteen_lists[1:]:
        if other != first:
            raise SystemExit(
                "Ambiguous tie handling beyond the frozen 4096 cutoff."
            )

    rule_name = matches[0][0]
    ids16 = first

    # --------------------------------------------------------
    # Recover P2 semantic vocabulary-hash convention.
    # --------------------------------------------------------

    p2_hashes = p2["vocabulary_sha256"].astype(str).unique()

    if len(p2_hashes) != 1:
        raise SystemExit(
            "Frozen P2 vocabulary has multiple semantic hashes."
        )

    expected_p2_semantic = p2_hashes[0]

    hash_candidates = semantic_hash_candidates(p2_ids)

    matching_hash_rules = [
        name
        for name, value in hash_candidates.items()
        if value == expected_p2_semantic
    ]

    if len(matching_hash_rules) != 1:
        raise SystemExit(
            "Could not uniquely reproduce P2 vocabulary semantic hash. "
            f"matches={matching_hash_rules}"
        )

    hash_rule = matching_hash_rules[0]

    p3_semantic = semantic_hash_candidates(ids16)[hash_rule]

    out = pd.DataFrame({
        "gene_index": range(16384),
        "ensembl_id": ids16,
        "vocabulary_sha256": p3_semantic,
    })

    if len(out) != 16384:
        raise AssertionError

    if out["ensembl_id"].duplicated().any():
        raise AssertionError(
            "P3 vocabulary contains duplicated genes"
        )

    # Strongest possible nesting check.
    if out.iloc[:4096]["ensembl_id"].tolist() != p2_ids:
        raise AssertionError(
            "Frozen P2 vocabulary is not exact prefix of P3 vocabulary"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = out_path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(out_path)

    p3_file_sha = sha256_file(out_path)

    receipt = {
        "schema_version": 1,
        "phase": "P3-2",
        "description": (
            "Expanded 16,384-gene vocabulary derived exclusively "
            "from the frozen P2 training-cell gene statistics."
        ),
        "target_gene_count": 16384,
        "vocabulary_source": "train_split_only",
        "selection": {
            "minimum_detected_training_cells": min_detected,
            "primary_ranking": (
                f"{variance_col} descending"
            ),
            "resolved_sort_rule": rule_name,
            "gene_id_column": gene_col,
            "detection_column": detection_col,
            "variance_column": variance_col,
            "eligible_gene_count": int(len(eligible)),
            "p2_exact_reproduction": True,
            "p2_exact_prefix_in_p3": True,
        },
        "hashing": {
            "semantic_hash_rule": hash_rule,
            "p2_semantic_sha256": expected_p2_semantic,
            "p3_semantic_sha256": p3_semantic,
            "p2_vocabulary_file_sha256":
                sha256_file(p2_vocab_path),
            "p3_vocabulary_file_sha256":
                p3_file_sha,
            "training_gene_stats_sha256":
                sha256_file(stats_path),
        },
        "artifacts": {
            "training_gene_stats": str(stats_path),
            "p2_vocabulary": str(p2_vocab_path),
            "p3_vocabulary": str(out_path),
            "config": str(cfg_path),
        },
        "heldout_feature_selection": False,
        "validation_used_for_feature_selection": False,
        "test_used_for_feature_selection": False,
        "external_data_used_for_feature_selection": False,
    }

    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    receipt_path.write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("========================================")
    print("P3_16K_VOCABULARY_BUILD=PASS")
    print("STATS_ROWS =", len(stats))
    print("ELIGIBLE_GENES =", len(eligible))
    print("GENE_ID_COLUMN =", gene_col)
    print("DETECTION_COLUMN =", detection_col)
    print("VARIANCE_COLUMN =", variance_col)
    print("MIN_DETECTED =", min_detected)
    print("SORT_RULE =", rule_name)
    print("P2_EXACT_REPRODUCTION=true")
    print("P2_EXACT_PREFIX_IN_P3=true")
    print("HASH_RULE =", hash_rule)
    print("P2_SEMANTIC_SHA256 =", expected_p2_semantic)
    print("P3_SEMANTIC_SHA256 =", p3_semantic)
    print("P3_FILE_SHA256 =", p3_file_sha)
    print("P3_ROWS =", len(out))
    print("========================================")


if __name__ == "__main__":
    main()
