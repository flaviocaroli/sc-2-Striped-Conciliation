from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CSRMemmap:
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]

    @classmethod
    def open(cls, directory: str | Path, prefix: str) -> "CSRMemmap":
        root = Path(directory)
        with (root / f"{prefix}.shape.json").open("r", encoding="utf-8") as handle:
            raw_shape = json.load(handle)
        shape = (int(raw_shape[0]), int(raw_shape[1]))
        data = np.load(root / f"{prefix}.data.npy", mmap_mode="r")
        indices = np.load(root / f"{prefix}.indices.npy", mmap_mode="r")
        indptr = np.load(root / f"{prefix}.indptr.npy", mmap_mode="r")
        if indptr.shape != (shape[0] + 1,):
            raise ValueError(f"Invalid indptr shape for {root}/{prefix}: {indptr.shape}")
        if data.shape != indices.shape:
            raise ValueError(f"CSR data/indices length mismatch for {root}/{prefix}")
        if int(indptr[-1]) != int(data.shape[0]):
            raise ValueError(f"CSR terminal pointer mismatch for {root}/{prefix}")
        return cls(data=data, indices=indices, indptr=indptr, shape=shape)

    def dense_row(self, row: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
        if row < 0 or row >= self.shape[0]:
            raise IndexError(row)
        start = int(self.indptr[row])
        stop = int(self.indptr[row + 1])
        output = np.zeros(self.shape[1], dtype=dtype)
        output[self.indices[start:stop]] = self.data[start:stop].astype(dtype, copy=False)
        return output


def write_csr(directory: str | Path, prefix: str, matrix: object) -> None:
    """Write a scipy-compatible CSR matrix without requiring scipy at import time."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    required = ("data", "indices", "indptr", "shape")
    if any(not hasattr(matrix, name) for name in required):
        raise TypeError("matrix must expose data, indices, indptr, and shape")
    np.save(root / f"{prefix}.data.npy", np.asarray(matrix.data))
    np.save(root / f"{prefix}.indices.npy", np.asarray(matrix.indices, dtype=np.int32))
    np.save(root / f"{prefix}.indptr.npy", np.asarray(matrix.indptr, dtype=np.int64))
    with (root / f"{prefix}.shape.json").open("w", encoding="utf-8") as handle:
        json.dump([int(matrix.shape[0]), int(matrix.shape[1])], handle)


def sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def shard_directory_sha256(directory: str | Path) -> str:
    root = Path(directory)
    files = [path for path in root.iterdir() if path.is_file() and not path.name.endswith(".tmp")]
    if not files:
        raise ValueError(f"No files in shard directory: {root}")
    return sha256_files(files)
