#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

GENERATED = [
    Path("build"),
    Path("dist"),
    Path("src/sc2.egg-info"),
    Path(".pytest_cache"),
    Path(".mypy_cache"),
    Path(".ruff_cache"),
    Path("htmlcov"),
    Path(".coverage"),
]

ARCHIVE_CANDIDATES = [
    Path("slurm/bulk"),
    Path("slurm/census"),
    Path("slurm/external"),
]

PROTECTED_DEPENDENCIES = [
    Path("src/sc2/models/striped/sc2_striped_medium.py"),
    Path("src/sc2/train/train_sc2_mamba_bridge.py"),
    Path("src/sc2/train/train_sc2_striped_medium.py"),
]


def git_tracked(root: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative SC2 repository cleanup")
    parser.add_argument("--root", default=".")
    parser.add_argument("--apply-generated", action="store_true", help="Delete generated artifacts")
    parser.add_argument("--archive-legacy", action="store_true", help="Move selected legacy launchers to archive")
    parser.add_argument("--archive-dir", default="archive/lung_pilots_2026")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not (root / ".git").exists():
        raise SystemExit(f"Not a git repository root: {root}")

    print("Protected dependency files (never touched by this script):")
    for relative in PROTECTED_DEPENDENCIES:
        print(f"  KEEP {relative} exists={(root / relative).exists()}")

    print("Generated artifacts:")
    for relative in GENERATED:
        absolute = root / relative
        if not absolute.exists():
            print(f"  ABSENT {relative}")
            continue
        action = "DELETE" if args.apply_generated else "WOULD_DELETE"
        print(f"  {action} {relative} tracked={git_tracked(root, relative)}")
        if args.apply_generated:
            remove_path(absolute)

    print("Legacy archive candidates:")
    archive_root = root / args.archive_dir
    for relative in ARCHIVE_CANDIDATES:
        source = root / relative
        if not source.exists():
            print(f"  ABSENT {relative}")
            continue
        destination = archive_root / relative
        action = "ARCHIVE" if args.archive_legacy else "WOULD_ARCHIVE"
        print(f"  {action} {relative} -> {destination.relative_to(root)}")
        if args.archive_legacy:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(source), str(destination))

    print("Run `git diff --stat` and the full test suite before committing.")


if __name__ == "__main__":
    main()
