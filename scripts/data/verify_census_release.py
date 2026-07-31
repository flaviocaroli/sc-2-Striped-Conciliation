#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a pinned CELLxGENE Census release opens")
    parser.add_argument("--census-release", required=True)
    args = parser.parse_args()
    try:
        import cellxgene_census
    except ImportError as error:
        raise SystemExit("cellxgene-census is not installed in the active environment") from error
    with cellxgene_census.open_soma(census_version=args.census_release):
        print(f"census_open=ok release={args.census_release}")


if __name__ == "__main__":
    main()
