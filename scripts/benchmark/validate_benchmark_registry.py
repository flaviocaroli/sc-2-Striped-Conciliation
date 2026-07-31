#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from sc2.data.census_pipeline import file_sha256, validate_registry_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that the SC2 benchmark registry is review-frozen")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--census-release", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON validation report")
    args = parser.parse_args()

    path = Path(args.registry)
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Registry YAML must contain a mapping")
    errors = validate_registry_payload(registry, expected_release=args.census_release)
    report = {
        "registry": str(path.resolve()),
        "registry_sha256": file_sha256(path),
        "census_release": args.census_release,
        "benchmarks": len(registry.get("benchmarks", [])),
        "errors": errors,
        "valid": not errors,
    }
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        print("benchmark_registry=INCOMPLETE")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(2)
    print(f"benchmark_registry=frozen sha256={report['registry_sha256']}")


if __name__ == "__main__":
    main()
