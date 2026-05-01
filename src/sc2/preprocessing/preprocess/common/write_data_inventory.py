from __future__ import annotations

import json
import os
from pathlib import Path


def file_info(path: Path) -> dict:
    return {
        "exists": path.exists(),
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def main() -> None:
    data_root = Path(os.environ.get("SC2_DATA_ROOT", "/home/3159436/sc2/data"))
    artifact_root = data_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    inventory = {
        "archs4_human_gene": file_info(data_root / "raw" / "archs4" / "human_gene_v2.5.h5"),
        "census_pilot": file_info(data_root / "raw" / "census" / "census_pilot.h5ad"),
    }

    out_path = artifact_root / "data_inventory.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print(f"wrote inventory to: {out_path}")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()