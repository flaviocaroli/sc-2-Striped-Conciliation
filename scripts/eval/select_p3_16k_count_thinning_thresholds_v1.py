#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)
    return h.hexdigest()


def qlabel(q: float) -> str:
    return f"q{int(round(q * 100)):03d}"


def verify_manifest(root: Path) -> None:
    m = root / "SHA256SUMS.txt"

    if not m.is_file():
        raise RuntimeError(
            f"missing manifest: {m}"
        )

    for line in m.read_text().splitlines():

        digest, rel = line.split(
            None,
            1,
        )

        p = root / rel.strip()

        if not p.is_file():
            raise RuntimeError(
                f"missing hashed file {p}"
            )

        if sha(p) != digest:
            raise RuntimeError(
                f"SHA mismatch {p}"
            )


def aggregate(
    runs: pd.DataFrame,
    max_fill: float,
) -> pd.DataFrame:

    rows = []

    expected_runs = int(
        runs["run_id"].nunique()
    )

    if expected_runs != 25:
        raise RuntimeError(
            f"expected 25 runs, got "
            f"{expected_runs}"
        )

    for threshold, g in runs.groupby(
        "threshold",
        sort=True,
    ):

        if g["run_id"].nunique() != 25:
            raise RuntimeError(
                "incomplete threshold grid"
            )

        rows.append({
            "threshold":
                float(threshold),

            "n_runs":
                25,

            "mean_recall":
                float(g["recall"].mean()),

            "mean_precision":
                float(g["precision"].mean()),

            "mean_true_zero_fill":
                float(
                    g[
                        "true_zero_fill"
                    ].mean()
                ),

            "max_true_zero_fill":
                float(
                    g[
                        "true_zero_fill"
                    ].max()
                ),
        })

    result = pd.DataFrame(rows)

    if len(result) != 201:
        raise RuntimeError(
            "aggregate threshold grid != 201"
        )

    result["eligible"] = (
        result["max_true_zero_fill"]
        <= max_fill
    )

    return result


def choose(
    aggregate_df: pd.DataFrame,
) -> pd.Series:

    eligible = aggregate_df[
        aggregate_df["eligible"]
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "no eligible threshold"
        )

    return (
        eligible
        .sort_values(
            [
                "mean_recall",
                "mean_precision",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="mergesort",
        )
        .iloc[0]
    )


def write_manifest(root: Path) -> None:

    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file()
        and p.name != "SHA256SUMS.txt"
    )

    (
        root / "SHA256SUMS.txt"
    ).write_text(
        "".join(
            f"{sha(p)}  {p.name}\n"
            for p in files
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--protocol",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    p = json.loads(
        protocol_path.read_text()
    )

    if p["protocol_id"] != \
            "sc2-p3-16k-count-thinning-validation-inference-v1":
        raise RuntimeError(
            "wrong protocol"
        )

    tasks = p[
        "task_layout"
    ]["tasks"]

    if len(tasks) != 25:
        raise RuntimeError(
            "selector expected 25 tasks"
        )

    root = Path(
        p["output"]["root"]
    )

    out = Path(
        args.output_dir
    )

    if out.exists():
        raise RuntimeError(
            f"selection output exists: {out}"
        )

    sweep_rows = []
    value_rows = []

    for task in tasks:

        tid = int(task["task_id"])
        seed = int(task["model_seed"])
        rep = int(task["replicate"])

        task_dir = (
            root
            / (
                f"task_{tid:02d}"
                f"_model{seed}"
                f"_rep{rep:02d}"
            )
        )

        if not task_dir.is_dir():
            raise RuntimeError(
                f"missing task {task_dir}"
            )

        verify_manifest(
            task_dir
        )

        manifest = json.loads(
            (
                task_dir
                / "task_manifest.json"
            ).read_text()
        )

        assert manifest["status"] == "PASS"
        assert manifest[
            "per_panel_threshold_selection_used"
        ] is False

        assert manifest[
            "scientific_threshold_selection_deferred"
        ] is True

        for q in p[
            "task_layout"
        ]["q_order"]:

            q = float(q)

            qdir = (
                task_dir
                / qlabel(q)
            )

            summary = json.loads(
                (
                    qdir
                    / "summary.json"
                ).read_text()
            )

            sweep = pd.read_csv(
                qdir
                / "threshold_sweep.csv"
            )

            if len(sweep) != 201:
                raise RuntimeError(
                    "threshold sweep != 201"
                )

            expected_grid = np.linspace(
                0.0,
                1.0,
                201,
            )

            if not np.allclose(
                sweep[
                    "threshold"
                ].to_numpy(
                    dtype=np.float64
                ),
                expected_grid,
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "threshold grid changed"
                )

            run_id = (
                f"q={q:.2f}"
                f"|model={seed}"
                f"|rep={rep}"
            )

            local = sweep[
                [
                    "threshold",
                    "recall",
                    "precision",
                    "true_zero_fill",
                ]
            ].copy()

            local["run_id"] = run_id
            local["q"] = q
            local["model_seed"] = seed
            local["replicate"] = rep

            sweep_rows.append(
                local
            )

            target_sd = float(
                summary["target_sd"]
            )

            masked_mse = float(
                summary["masked_mse"]
            )

            value_rows.append({
                "run_id":
                    run_id,

                "q":
                    q,

                "model_seed":
                    seed,

                "replicate":
                    rep,

                "masked_mse":
                    masked_mse,

                "masked_mae":
                    float(
                        summary[
                            "masked_mae"
                        ]
                    ),

                "recovery_index":
                    float(
                        1.0
                        - masked_mse
                        / (target_sd ** 2)
                    ),

                "gene_spearman":
                    float(
                        summary[
                            "gene_spearman"
                        ]
                    ),

                "sample_spearman":
                    float(
                        summary[
                            "sample_spearman"
                        ]
                    ),

                "sd_ratio":
                    float(
                        summary[
                            "sd_ratio"
                        ]
                    ),

                "gate_auroc":
                    float(
                        summary[
                            "gate_auroc"
                        ]
                    ),

                "gate_auprc":
                    float(
                        summary[
                            "gate_auprc"
                        ]
                    ),
            })

    values = pd.DataFrame(
        value_rows
    )

    if len(values) != 75:
        raise RuntimeError(
            f"expected 75 evaluations, "
            f"got {len(values)}"
        )

    sweeps = pd.concat(
        sweep_rows,
        ignore_index=True,
    )

    max_fill = float(
        p[
            "common_threshold_selection"
        ][
            "max_true_zero_fill"
        ]
    )

    selected = {}
    agg_frames = []

    for q in p[
        "task_layout"
    ]["q_order"]:

        q = float(q)

        qdata = sweeps[
            np.isclose(
                sweeps["q"],
                q,
                atol=1e-12,
                rtol=0.0,
            )
        ].copy()

        if qdata[
            "run_id"
        ].nunique() != 25:
            raise RuntimeError(
                f"q={q} does not have "
                "25 runs"
            )

        agg = aggregate(
            qdata,
            max_fill,
        )

        choice = choose(
            agg
        )

        agg["q"] = q

        agg_frames.append(
            agg
        )

        selected[
            f"{q:.2f}"
        ] = {
            "threshold":
                float(
                    choice[
                        "threshold"
                    ]
                ),

            "mean_recall":
                float(
                    choice[
                        "mean_recall"
                    ]
                ),

            "mean_precision":
                float(
                    choice[
                        "mean_precision"
                    ]
                ),

            "mean_true_zero_fill":
                float(
                    choice[
                        "mean_true_zero_fill"
                    ]
                ),

            "max_true_zero_fill":
                float(
                    choice[
                        "max_true_zero_fill"
                    ]
                ),

            "n_runs":
                25,
        }

    aggregate_df = pd.concat(
        agg_frames,
        ignore_index=True,
    )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    values.to_csv(
        out / "run_level_value_metrics.csv",
        index=False,
    )

    aggregate_df.to_csv(
        out / "aggregate_threshold_sweep.csv",
        index=False,
    )

    payload = {
        "schema":
            "sc2-p3-16k-count-thinning-selected-thresholds-v1",

        "status":
            "PASS",

        "scientific_result":
            True,

        "selection_data":
            "internal_validation_only",

        "selected_model":
            "P3-16K-Small",

        "validation_evaluations":
            75,

        "runs_per_q":
            25,

        "model_seeds":
            5,

        "thinning_replicates":
            5,

        "selection_unit":
            "one common threshold per q",

        "eligibility":
            (
                "true_zero_fill <= 0.02 "
                "in every one of the 25 "
                "validation runs for that q"
            ),

        "objective_order": [
            "maximize mean recall",
            "maximize mean precision",
            "choose stricter higher threshold",
        ],

        "selected_thresholds":
            selected,

        "runtime_placeholder_threshold":
            1.0,

        "runtime_placeholder_used_for_selection":
            False,

        "full_threshold_sweep_used":
            True,

        "internal_test_used":
            False,

        "baron_used":
            False,

        "zheng68k_used":
            False,

        "protocol":
            str(protocol_path),

        "protocol_sha256":
            sha(protocol_path),
    }

    (
        out / "selected_thresholds.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    )

    write_manifest(
        out
    )

    print(
        json.dumps(
            selected,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "P3_16K_THINNING_THRESHOLD_SELECTION=PASS"
    )


if __name__ == "__main__":
    main()
