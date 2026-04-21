from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc2.config import load_yaml, merge_train_and_paths
from sc2.utils.paths import ensure_dir
from sc2.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SC2 Stage 1 smoke test")
    parser.add_argument("--config", required=True, help="Path to train config yaml")
    parser.add_argument("--paths", required=True, help="Path to path config yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_cfg = load_yaml(args.config)
    path_cfg = load_yaml(args.paths)
    cfg = merge_train_and_paths(train_cfg, path_cfg)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    output_root = Path(cfg["paths"]["output_root"])
    run_name = cfg.get("run_name", "stage1_smoke")
    run_dir = output_root / run_name
    ckpt_dir = run_dir / "checkpoints"

    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    resolved_config_path = run_dir / "resolved_config.json"
    with resolved_config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    metrics = {
        "status": "ok",
        "message": "Stage 1 smoke test completed.",
        "run_name": run_name,
        "seed": seed,
        "data_root": cfg["paths"]["data_root"],
        "output_root": cfg["paths"]["output_root"],
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (ckpt_dir / "placeholder.txt").open("w", encoding="utf-8") as f:
        f.write("This is a placeholder checkpoint artifact.\n")

    print("SC2 Stage 1 smoke test completed successfully.")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()