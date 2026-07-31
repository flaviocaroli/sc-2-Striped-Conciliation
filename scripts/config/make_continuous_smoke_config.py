#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a short single-cell-only SC2 continuous-training smoke config")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=10)
    parser.add_argument("--diagnostic-every-steps", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Source YAML must contain a mapping")

    payload["run_name"] = args.run_name
    train = payload.setdefault("train", {})
    train["total_steps"] = int(args.steps)
    train["batch_size"] = int(args.batch_size)
    train["grad_accum_steps"] = int(args.grad_accum_steps)
    train["save_every_steps"] = int(args.save_every_steps)
    train["log_every_steps"] = 1
    train["gradient_diagnostic_every_steps"] = int(args.diagnostic_every_steps)
    train["num_workers"] = 0
    train["modality_weights"] = {"sc": 1.0, "bulk": 0.0, "pseudobulk": 0.0}

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"smoke_config={destination} run_name={args.run_name} steps={args.steps}")


if __name__ == "__main__":
    main()
