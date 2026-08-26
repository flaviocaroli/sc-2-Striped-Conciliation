#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp


DATASETS = (
    "internal_test",
    "baron_pancreas",
    "zheng68k",
)

QS = (0.85, 0.70, 0.50)

METHODS = (
    "positive_train_mean",
    "positive_train_median",
    "truncated_low_rank",
    "knn",
    "alra",
    "scvi",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)

    order = np.argsort(p)
    ps = p[order]

    adj_sorted = np.empty(m, dtype=float)
    running = 0.0

    for i, value in enumerate(ps):
        candidate = (m - i) * value
        running = max(running, candidate)
        adj_sorted[i] = min(1.0, running)

    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out


def mean_ci(values, level=0.95):
    x = np.asarray(values, dtype=float)
    n = x.size

    assert n >= 2
    assert np.isfinite(x).all()

    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)

    tcrit = float(
        student_t.ppf(
            1.0 - (1.0 - level) / 2.0,
            df=n - 1,
        )
    )

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - tcrit * se,
        "ci95_high": mean + tcrit * se,
    }


def one_sample_t(values):
    x = np.asarray(values, dtype=float)

    assert x.size >= 2
    assert np.isfinite(x).all()

    if np.allclose(x, x[0], rtol=0, atol=0):
        p = 1.0 if x[0] == 0 else 0.0
    else:
        p = float(
            ttest_1samp(
                x,
                popmean=0.0,
                alternative="two-sided",
            ).pvalue
        )

    return p


def exact_sign_flip_p(values):
    x = np.asarray(values, dtype=float)
    n = len(x)

    assert n == 4
    assert np.isfinite(x).all()

    observed = abs(float(x.mean()))

    stats = []

    for signs in itertools.product((-1.0, 1.0), repeat=n):
        s = np.asarray(signs)
        stats.append(
            abs(float(np.mean(s * x)))
        )

    stats = np.asarray(stats)

    return float(
        np.mean(
            stats >= observed - 1e-15
        )
    )


def variance_components_crossed(df, value_col):
    """
    Crossed random effects:
        y_sr = mu + seed_s + replicate_r + residual_sr

    3 seeds x 5 thinning replicates, one observation per cell.
    Residual therefore includes seed-by-replicate interaction.
    """
    tab = df.pivot(
        index="model_seed",
        columns="replicate",
        values=value_col,
    )

    assert tab.shape == (3, 5)
    assert not tab.isna().any().any()

    y = tab.to_numpy(dtype=float)

    a, b = y.shape

    grand = y.mean()
    seed_mean = y.mean(axis=1)
    rep_mean = y.mean(axis=0)

    ss_seed = b * np.sum((seed_mean - grand) ** 2)
    ss_rep = a * np.sum((rep_mean - grand) ** 2)
    ss_total = np.sum((y - grand) ** 2)

    ss_res = max(
        0.0,
        float(ss_total - ss_seed - ss_rep),
    )

    ms_seed = ss_seed / (a - 1)
    ms_rep = ss_rep / (b - 1)
    ms_res = ss_res / ((a - 1) * (b - 1))

    var_seed = max(
        0.0,
        float((ms_seed - ms_res) / b),
    )

    var_rep = max(
        0.0,
        float((ms_rep - ms_res) / a),
    )

    var_res = max(
        0.0,
        float(ms_res),
    )

    total = var_seed + var_rep + var_res

    return {
        "variance_model_seed": var_seed,
        "variance_thinning_replicate": var_rep,
        "variance_residual_interaction": var_res,
        "variance_total": total,
        "fraction_model_seed":
            var_seed / total if total else 0.0,
        "fraction_thinning_replicate":
            var_rep / total if total else 0.0,
        "fraction_residual_interaction":
            var_res / total if total else 0.0,
    }


def require_columns(df, columns, name):
    missing = set(columns) - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{name} missing columns: {sorted(missing)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--analysis-commit", required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    protocol_path = Path(args.protocol).resolve()
    p = json.loads(protocol_path.read_text())

    assert p["schema_version"] == \
        "sc2-statistical-integration-protocol-v1"

    sources = {
        k: Path(v["path"])
        for k, v in p["sources"].items()
    }

    for k, path in sources.items():
        if not path.is_file():
            raise RuntimeError(f"missing source {k}: {path}")

        if sha(path) != p["sources"][k]["sha256"]:
            raise RuntimeError(f"source hash changed: {k}")

    sc2 = pd.read_csv(
        sources["sc2_thinning_runs"]
    )

    cmp = pd.read_csv(
        sources["comparator_replicate_results"]
    )

    abl = pd.read_csv(
        sources["ablation_paired_seed_effects"]
    )

    baron = pd.read_csv(
        sources["baron_donor_summary"]
    )

    sf = pd.read_csv(
        sources["scfoundation_headline"]
    )

    scgpt = json.loads(
        sources["scgpt_audit"].read_text()
    )

    require_columns(
        sc2,
        [
            "model_seed",
            "dataset",
            "replicate",
            "q",
            "recovery_index",
            "gene_spearman",
            "threshold_true_zero_fill",
        ],
        "SC2 thinning",
    )

    require_columns(
        cmp,
        [
            "dataset",
            "q",
            "method",
            "thinning_replicate",
            "candidate_value_recovery_index",
            "recovery_index",
            "gene_spearman",
            "true_zero_fill",
        ],
        "comparator thinning",
    )

    require_columns(
        abl,
        [
            "variant",
            "seed",
            "mask_percent",
            "delta_recovery_index",
            "delta_gene_spearman",
        ],
        "ablation",
    )

    require_columns(
        baron,
        [
            "q",
            "heldout_donor",
            "sc2_macro_f1",
            "corrupted_macro_f1",
            "delta_macro_f1",
        ],
        "Baron",
    )

    require_columns(
        sf,
        [
            "dataset",
            "mask_percent",
            "recovery_index",
            "gene_spearman",
            "true_zero_changed_fraction",
            "evaluated_genes",
        ],
        "scFoundation",
    )

    if args.check_only:
        print("PHASE7_SCHEMA_CHECK=PASS")
        return

    out_root = Path(p["output_root"])
    receipt_path = Path(args.receipt)

    if out_root.exists():
        raise RuntimeError("Phase7 output already exists")

    if receipt_path.exists():
        raise RuntimeError("Phase7 receipt already exists")

    # ========================================================
    # Count thinning — apples-to-apples direct value recovery
    # ========================================================

    sc2["q"] = sc2["q"].astype(float).round(2)
    cmp["q"] = cmp["q"].astype(float).round(2)

    assert len(sc2) == 135

    sc2_rep = (
        sc2.groupby(
            ["dataset", "q", "replicate"],
            as_index=False,
        )
        .agg(
            sc2_recovery_index=("recovery_index", "mean"),
            sc2_gene_spearman=("gene_spearman", "mean"),
            sc2_true_zero_fill=("threshold_true_zero_fill", "mean"),
            sc2_seed_sd_recovery=("recovery_index", "std"),
            sc2_seed_sd_gene_spearman=("gene_spearman", "std"),
            model_seed_count=("model_seed", "nunique"),
        )
        .rename(
            columns={
                "replicate": "thinning_replicate",
            }
        )
    )

    assert len(sc2_rep) == 45
    assert (sc2_rep["model_seed_count"] == 3).all()

    direct_rows = []

    for dataset in DATASETS:
        for q in QS:
            for method in METHODS:

                a = (
                    sc2_rep[
                        (sc2_rep["dataset"] == dataset)
                        & (sc2_rep["q"] == q)
                    ]
                    .sort_values("thinning_replicate")
                    .reset_index(drop=True)
                )

                b = (
                    cmp[
                        (cmp["dataset"] == dataset)
                        & (cmp["q"] == q)
                        & (cmp["method"] == method)
                    ]
                    .sort_values("thinning_replicate")
                    .reset_index(drop=True)
                )

                assert len(a) == 5
                assert len(b) == 5

                assert np.array_equal(
                    a["thinning_replicate"].to_numpy(),
                    b["thinning_replicate"].to_numpy(),
                )

                sc2_r = a["sc2_recovery_index"].to_numpy(float)

                comparator_r = b[
                    "candidate_value_recovery_index"
                ].to_numpy(float)

                delta = sc2_r - comparator_r
                ci = mean_ci(delta)

                direct_rows.append({
                    "dataset": dataset,
                    "q": q,
                    "comparator": method,
                    "n_thinning_replicates": 5,

                    "sc2_recovery_mean":
                        float(sc2_r.mean()),

                    "comparator_candidate_recovery_mean":
                        float(comparator_r.mean()),

                    "delta_sc2_minus_comparator_mean":
                        ci["mean"],

                    "delta_sd":
                        ci["sd"],

                    "delta_ci95_low":
                        ci["ci95_low"],

                    "delta_ci95_high":
                        ci["ci95_high"],

                    "sc2_better_replicates":
                        int(np.sum(delta > 0)),

                    "sc2_better_all_5":
                        bool(np.all(delta > 0)),

                    "population_p_value":
                        np.nan,

                    "p_value_reason":
                        (
                            "not computed: thinning replicates are "
                            "technical corruption realizations rather "
                            "than biological replication"
                        ),
                })

    thinning_direct = pd.DataFrame(direct_rows)

    assert len(thinning_direct) == 54

    # Best comparator by the correctly aligned candidate-value R.
    best_rows = []

    for dataset in DATASETS:
        for q in QS:

            g = thinning_direct[
                (thinning_direct["dataset"] == dataset)
                & (thinning_direct["q"] == q)
            ]

            row = g.loc[
                g["comparator_candidate_recovery_mean"].idxmax()
            ].copy()

            best_rows.append(row)

    thinning_best = pd.DataFrame(best_rows)

    assert len(thinning_best) == 9

    # Deployment metrics remain useful, but are explicitly distinct.
    deployment = (
        cmp.groupby(
            ["dataset", "q", "method"],
            as_index=False,
        )
        .agg(
            deployment_recovery_mean=("recovery_index", "mean"),
            deployment_gene_spearman_mean=("gene_spearman", "mean"),
            deployment_true_zero_fill_mean=("true_zero_fill", "mean"),
            deployment_true_zero_fill_max=("true_zero_fill", "max"),
            candidate_value_recovery_mean=(
                "candidate_value_recovery_index",
                "mean",
            ),
        )
    )

    assert len(deployment) == 54

    structural_note = pd.DataFrame([
        {
            "sc2_gene_spearman_scope":
                "expected_repair on synthetically/lost-positive targets",
            "comparator_gene_spearman_scope":
                "thresholded deployed reconstruction",
            "direct_head_to_head_delta_computed":
                False,
            "reason":
                (
                    "These stored structural metrics use different "
                    "prediction objects; Phase7 does not treat them "
                    "as the same estimand."
                ),
        }
    ])

    # ========================================================
    # SC2 3x5 variance decomposition
    # ========================================================

    variance_rows = []

    for dataset in DATASETS:
        for q in QS:

            g = sc2[
                (sc2["dataset"] == dataset)
                & (sc2["q"] == q)
            ]

            assert len(g) == 15
            assert g["model_seed"].nunique() == 3
            assert g["replicate"].nunique() == 5

            for metric in (
                "recovery_index",
                "gene_spearman",
            ):

                vc = variance_components_crossed(
                    g,
                    metric,
                )

                variance_rows.append({
                    "dataset": dataset,
                    "q": q,
                    "metric": metric,
                    **vc,
                })

    variance = pd.DataFrame(variance_rows)
    assert len(variance) == 18

    # ========================================================
    # Five-seed ablation inference
    # ========================================================

    ablation_rows = []

    metric_map = {
        "recovery_index":
            "delta_recovery_index",
        "gene_spearman":
            "delta_gene_spearman",
    }

    variants = sorted(
        abl["variant"].unique()
    )

    assert len(variants) == 5

    for variant in variants:
        for mask in (15, 30, 50):
            for metric_name, column in metric_map.items():

                g = (
                    abl[
                        (abl["variant"] == variant)
                        & (abl["mask_percent"] == mask)
                    ]
                    .sort_values("seed")
                )

                assert len(g) == 5
                assert g["seed"].nunique() == 5

                values = g[column].to_numpy(float)

                ci = mean_ci(values)
                p_raw = one_sample_t(values)

                ablation_rows.append({
                    "variant": variant,
                    "mask_percent": mask,
                    "metric": metric_name,
                    "n_paired_seeds": 5,
                    "mean_delta_variant_minus_A0":
                        ci["mean"],
                    "sd_delta":
                        ci["sd"],
                    "ci95_low":
                        ci["ci95_low"],
                    "ci95_high":
                        ci["ci95_high"],
                    "p_raw":
                        p_raw,
                    "positive_seed_deltas":
                        int(np.sum(values > 0)),
                    "negative_seed_deltas":
                        int(np.sum(values < 0)),
                })

    ablation_tests = pd.DataFrame(
        ablation_rows
    )

    assert len(ablation_tests) == 30

    ablation_tests["p_holm_30"] = holm_adjust(
        ablation_tests["p_raw"].to_numpy(float)
    )

    ablation_tests["holm_significant_0_05"] = (
        ablation_tests["p_holm_30"] < 0.05
    )

    # ========================================================
    # Baron donor-level downstream inference
    # ========================================================

    baron_rows = []

    for q in QS:

        g = (
            baron[
                np.isclose(
                    baron["q"].astype(float),
                    q,
                    rtol=0,
                    atol=1e-8,
                )
            ]
            .sort_values("heldout_donor")
        )

        assert len(g) == 4
        assert g["heldout_donor"].nunique() == 4

        values = g["delta_macro_f1"].to_numpy(float)

        ci = mean_ci(values)

        baron_rows.append({
            "q": q,
            "n_donors": 4,
            "mean_delta_macro_f1":
                ci["mean"],
            "sd_delta_macro_f1":
                ci["sd"],
            "ci95_low":
                ci["ci95_low"],
            "ci95_high":
                ci["ci95_high"],
            "positive_donors":
                int(np.sum(values > 0)),
            "negative_donors":
                int(np.sum(values < 0)),
            "p_exact_sign_flip":
                exact_sign_flip_p(values),
        })

    baron_tests = pd.DataFrame(
        baron_rows
    )

    assert len(baron_tests) == 3

    baron_tests["p_holm_3"] = holm_adjust(
        baron_tests["p_exact_sign_flip"].to_numpy(float)
    )

    baron_tests["holm_significant_0_05"] = (
        baron_tests["p_holm_3"] < 0.05
    )

    # ========================================================
    # Foundation-model descriptive integration
    # ========================================================

    sf_out = sf.copy()

    sf_out.insert(
        0,
        "model",
        "scFoundation",
    )

    sf_out["compatibility_status"] = \
        "DIRECT_DECODER_EVALUATED"

    sf_out["inferential_test"] = False

    scgpt_row = pd.DataFrame([{
        "model": "scGPT",
        "dataset": "not_evaluated",
        "mask_percent": np.nan,
        "recovery_index": np.nan,
        "gene_spearman": np.nan,
        "true_zero_changed_fraction": np.nan,
        "evaluated_genes": np.nan,
        "compatibility_status":
            scgpt["compatibility_decision"],
        "inferential_test": False,
        "coverage_fraction":
            float(scgpt["coverage_fraction"]),
        "exact_symbol_overlap":
            int(scgpt["exact_symbol_overlap"]),
        "reason":
            scgpt["reason"],
    }])

    sf_out["coverage_fraction"] = np.nan
    sf_out["exact_symbol_overlap"] = np.nan
    sf_out["reason"] = (
        "official continuous decoder evaluated; "
        "pretraining overlap cannot be excluded"
    )

    foundation = pd.concat(
        [
            sf_out,
            scgpt_row,
        ],
        ignore_index=True,
        sort=False,
    )

    # ========================================================
    # Atomic output bundle
    # ========================================================

    tmp = (
        out_root.parent
        / f".tmp_{out_root.name}_{os.getpid()}"
    )

    assert not tmp.exists()
    assert not out_root.exists()

    tmp.mkdir(parents=True)

    outputs = {
        "thinning_direct_recovery_paired_effects.csv":
            thinning_direct,

        "thinning_best_comparator_candidate_recovery.csv":
            thinning_best,

        "thinning_deployment_metrics.csv":
            deployment,

        "thinning_structural_metric_alignment.csv":
            structural_note,

        "sc2_thinning_variance_components.csv":
            variance,

        "ablation_paired_seed_tests.csv":
            ablation_tests,

        "baron_donor_tests.csv":
            baron_tests,

        "foundation_model_summary.csv":
            foundation,
    }

    for name, df in outputs.items():
        df.to_csv(
            tmp / name,
            index=False,
        )

    headline = {
        "status": "PASS",

        "metric_alignment": {
            "direct_recovery_comparator_field":
                "candidate_value_recovery_index",

            "thresholded_comparator_recovery_field":
                "recovery_index",

            "direct_gene_spearman_head_to_head_performed":
                False,
        },

        "counts": {
            "thinning_direct_comparisons":
                int(len(thinning_direct)),

            "thinning_best_condition_rows":
                int(len(thinning_best)),

            "variance_component_rows":
                int(len(variance)),

            "ablation_tests":
                int(len(ablation_tests)),

            "baron_tests":
                int(len(baron_tests)),

            "foundation_rows":
                int(len(foundation)),
        },

        "ablation_holm_significant_count":
            int(
                ablation_tests[
                    "holm_significant_0_05"
                ].sum()
            ),

        "baron_holm_significant_count":
            int(
                baron_tests[
                    "holm_significant_0_05"
                ].sum()
            ),

        "scgpt_compatibility":
            scgpt["compatibility_decision"],

        "population_pvalues_for_thinning":
            False,
    }

    headline_path = (
        tmp
        / "phase7_headline.json"
    )

    headline_path.write_text(
        json.dumps(
            headline,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    artifact_paths = [
        tmp / name
        for name in outputs
    ] + [
        headline_path
    ]

    (
        tmp
        / "SHA256SUMS.txt"
    ).write_text(
        "".join(
            f"{sha(path)}  {path.name}\n"
            for path in artifact_paths
        )
    )

    os.replace(
        tmp,
        out_root,
    )

    # ========================================================
    # Receipt
    # ========================================================

    receipt = {
        "schema_version":
            "sc2-statistical-integration-results-v1",

        "status":
            "PASS",

        "analysis_commit":
            args.analysis_commit,

        "protocol": {
            "path":
                str(protocol_path),

            "sha256":
                sha(protocol_path),
        },

        "source_receipts":
            p["receipts"],

        "metric_alignment": {
            "sc2_direct_recovery":
                "recovery_index from expected_repair",

            "comparator_direct_recovery":
                "candidate_value_recovery_index",

            "comparator_deployment_recovery":
                "recovery_index",

            "direct_comparator_gene_spearman_inference":
                False,

            "scientific_output_rerun":
                False,

            "model_inference_rerun":
                False,
        },

        "statistical_units": {
            "thinning":
                "technical thinning replicate; effect sizes/CIs only",

            "ablation":
                "predeclared model seed",

            "baron":
                "held-out donor",

            "foundation":
                "descriptive only",
        },

        "multiplicity": {
            "ablation":
                "Holm across 30 paired-seed tests",

            "baron":
                "Holm across three donor-level q tests",
        },

        "artifacts": {
            name: {
                "path":
                    str(out_root / name),

                "sha256":
                    sha(out_root / name),
            }
            for name in outputs
        },

        "headline": {
            "path":
                str(
                    out_root
                    / "phase7_headline.json"
                ),

            "sha256":
                sha(
                    out_root
                    / "phase7_headline.json"
                ),
        },

        "manifest": {
            "path":
                str(
                    out_root
                    / "SHA256SUMS.txt"
                ),

            "sha256":
                sha(
                    out_root
                    / "SHA256SUMS.txt"
                ),
        },

        "next_step":
            "Phase 8 manuscript, figure and table integration",
    }

    receipt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    receipt_path.write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    print()
    print("============================================================")
    print("PHASE 7 — STATISTICAL INTEGRATION")
    print("============================================================")
    print("METRIC_ALIGNMENT=PASS")
    print("THINNING_DIRECT_RECOVERY_FIELD=candidate_value_recovery_index")
    print("THINNING_POPULATION_PVALUES=false")
    print("SC2_VARIANCE_DECOMPOSITION=PASS")
    print("ABLATION_PAIRED_TESTS=30")
    print("BARON_DONOR_TESTS=3")
    print("FOUNDATION_INTEGRATION=PASS")

    print()
    print("BEST DIRECT-VALUE COMPARATOR BY CONDITION")
    print(
        thinning_best[
            [
                "dataset",
                "q",
                "comparator",
                "sc2_recovery_mean",
                "comparator_candidate_recovery_mean",
                "delta_sc2_minus_comparator_mean",
                "delta_ci95_low",
                "delta_ci95_high",
                "sc2_better_replicates",
            ]
        ].to_string(index=False)
    )

    print()
    print("HOLM-SIGNIFICANT ABLATION EFFECTS")
    sig = ablation_tests[
        ablation_tests[
            "holm_significant_0_05"
        ]
    ]

    if len(sig):
        print(
            sig[
                [
                    "variant",
                    "mask_percent",
                    "metric",
                    "mean_delta_variant_minus_A0",
                    "ci95_low",
                    "ci95_high",
                    "p_raw",
                    "p_holm_30",
                ]
            ].to_string(index=False)
        )
    else:
        print("NONE")

    print()
    print("BARON DONOR-LEVEL EFFECTS")
    print(
        baron_tests.to_string(
            index=False
        )
    )

    print()
    print("SCGPT_COMPATIBILITY="
          + scgpt["compatibility_decision"])

    print("PHASE7_STATISTICAL_INTEGRATION=PASS")


if __name__ == "__main__":
    main()
