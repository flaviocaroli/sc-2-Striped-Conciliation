#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _compare(left: Any, right: Any, path: str, differences: list[dict[str, object]]) -> tuple[bool, float]:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            differences.append({"path": path, "reason": "tensor_shape_or_dtype"})
            return False, float("inf")
        if torch.equal(left, right):
            return True, 0.0
        maximum = float((left.float() - right.float()).abs().max().item())
        if len(differences) < 50:
            differences.append({"path": path, "max_abs_difference": maximum})
        return False, maximum
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            differences.append({"path": path, "reason": "array_shape_or_dtype"})
            return False, float("inf")
        if np.array_equal(left, right):
            return True, 0.0
        maximum = float(np.max(np.abs(left.astype(float) - right.astype(float))))
        if len(differences) < 50:
            differences.append({"path": path, "max_abs_difference": maximum})
        return False, maximum
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append({"path": path, "reason": "mapping_keys"})
            return False, float("inf")
        equal = True
        maximum = 0.0
        for key in sorted(left, key=str):
            child_equal, child_max = _compare(left[key], right[key], f"{path}.{key}", differences)
            equal &= child_equal
            maximum = max(maximum, child_max)
        return equal, maximum
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            differences.append({"path": path, "reason": "sequence_length"})
            return False, float("inf")
        equal = True
        maximum = 0.0
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            child_equal, child_max = _compare(left_item, right_item, f"{path}[{index}]", differences)
            equal &= child_equal
            maximum = max(maximum, child_max)
        return equal, maximum
    try:
        equal = bool(left == right)
    except Exception:
        equal = False
    if not equal and len(differences) < 50:
        differences.append({"path": path, "left": repr(left)[:200], "right": repr(right)[:200]})
    return equal, 0.0 if equal else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare uninterrupted and resumed SC2 continuous checkpoints")
    parser.add_argument("reference")
    parser.add_argument("resumed")
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--require-config-hash-equal", action="store_true")
    args = parser.parse_args()

    reference = torch.load(Path(args.reference), map_location="cpu", weights_only=False)
    resumed = torch.load(Path(args.resumed), map_location="cpu", weights_only=False)

    required_equal = ["format", "global_step", "next_sample_index", "manifest_sha256"]
    if args.require_config_hash_equal:
        required_equal.append("config_sha256")
    field_equal = {key: reference.get(key) == resumed.get(key) for key in required_equal}

    sections = [
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
    ]
    section_results: dict[str, object] = {}
    overall_equal = True
    overall_max = 0.0
    differences: list[dict[str, object]] = []
    for section in sections:
        if section not in reference and section not in resumed:
            section_results[section] = "absent_in_both"
            continue
        equal, maximum = _compare(reference.get(section), resumed.get(section), section, differences)
        section_results[section] = {"bitwise_equal": equal, "max_abs_difference": maximum}
        overall_equal &= equal
        overall_max = max(overall_max, maximum)

    result = {
        "field_equal": field_equal,
        "reference_config_sha256": reference.get("config_sha256"),
        "resumed_config_sha256": resumed.get("config_sha256"),
        "sections": section_results,
        "all_compared_state_bitwise_equal": overall_equal,
        "max_numeric_difference": overall_max,
        "differences_preview": differences,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if not all(field_equal.values()):
        raise SystemExit(2)
    if args.require_bitwise and not overall_equal:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
