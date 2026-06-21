#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "src/sc2/train/train_sc2_mamba_bridge.py",
    ROOT / "src/sc2/eval/evaluate_sc2_mamba_bridge.py",
    ROOT / "src/sc2/eval/benchmarks/evaluate_masked_reconstruction.py",
    ROOT / "src/sc2/models/factory.py",
]


def patch_build_model(path: Path) -> bool:
    if not path.exists():
        print(f"skip missing {path}")
        return False
    text = path.read_text(encoding="utf-8")
    if "sc2_striped_medium" in text:
        print(f"already patched {path}")
        return False
    lines = text.splitlines(keepends=True)
    mini_idx = None
    for idx, line in enumerate(lines):
        if "sc2_striped_mini" in line and "kind" in line:
            mini_idx = idx
            break
    if mini_idx is None:
        print(f"skip no sc2_striped_mini build block in {path}")
        return False
    raise_idx = None
    for idx in range(mini_idx + 1, min(len(lines), mini_idx + 120)):
        if lines[idx].lstrip().startswith("raise ValueError"):
            raise_idx = idx
            break
    if raise_idx is None:
        print(f"skip no local raise ValueError after mini block in {path}")
        return False
    indent = lines[raise_idx][: len(lines[raise_idx]) - len(lines[raise_idx].lstrip())]
    block = [
        f"{indent}if kind in {{\"sc2_striped_medium\", \"striped_medium\", \"sc2_medium\"}}:\n",
        f"{indent}    from sc2.models.striped.sc2_striped_medium import build_sc2_striped_medium_from_config\n",
        f"{indent}    model = build_sc2_striped_medium_from_config(model_cfg, n_genes=n_genes)\n",
        f"{indent}    return \"sc2_striped_medium\", model\n",
        "\n",
    ]
    lines[raise_idx:raise_idx] = block
    new_text = "".join(lines)
    new_text = new_text.replace(
        "'sc2_mamba_bridge', 'native_mamba_bridge', 'sc2_hybrid_bridge', 'sc2_striped_mini'",
        "'sc2_mamba_bridge', 'native_mamba_bridge', 'sc2_hybrid_bridge', 'sc2_striped_mini', 'sc2_striped_medium'",
    )
    new_text = new_text.replace(
        '"sc2_mamba_bridge", "native_mamba_bridge", "sc2_hybrid_bridge", "sc2_striped_mini"',
        '"sc2_mamba_bridge", "native_mamba_bridge", "sc2_hybrid_bridge", "sc2_striped_mini", "sc2_striped_medium"',
    )
    path.write_text(new_text, encoding="utf-8")
    print(f"patched {path}")
    return True


def main() -> None:
    changed = 0
    for target in TARGETS:
        changed += int(patch_build_model(target))
    print(f"changed_files={changed}")


if __name__ == "__main__":
    main()
