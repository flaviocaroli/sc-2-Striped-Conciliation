from __future__ import annotations

import hashlib

import numpy as np
from scipy import sparse


def _counts_array(counts: np.ndarray) -> np.ndarray:
    array = np.asarray(counts)

    if array.ndim != 2:
        raise ValueError(
            f"counts must be two-dimensional, got shape={array.shape}"
        )

    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"counts must have integer dtype, got {array.dtype}"
        )

    if np.any(array < 0):
        raise ValueError("counts contain negative values")

    maximum = int(array.max()) if array.size else 0
    if maximum > np.iinfo(np.uint32).max:
        raise ValueError("counts exceed uint32 range")

    return np.asarray(
        array,
        dtype=np.uint32,
        order="C",
    )


def _availability(
    mask: np.ndarray | None,
    n_genes: int,
) -> np.ndarray:
    if mask is None:
        return np.ones(n_genes, dtype=np.bool_)

    available = np.asarray(mask, dtype=np.bool_)

    if available.shape != (n_genes,):
        raise ValueError(
            "available_gene_mask must have shape "
            f"({n_genes},), got {available.shape}"
        )

    return available


def cp10k_log1p_counts(
    counts: np.ndarray,
    available_gene_mask: np.ndarray | None = None,
    *,
    zero_library_rule: str = "error",
) -> np.ndarray:
    """Apply the corpus normalization exactly after gene availability.

    The operation mirrors the frozen Census materializer:
        total = sum available raw counts per cell
        CP10K = counts * 10000 / total
        output = log1p(CP10K)

    Unavailable genes are zeroed before the library-size denominator.
    """

    array = _counts_array(counts)
    available = _availability(
        available_gene_mask,
        array.shape[1],
    )

    working = array.copy()
    working[:, ~available] = 0

    matrix = sparse.csr_matrix(working)
    matrix = matrix.astype(np.float32, copy=True)

    totals = np.asarray(
        matrix.sum(axis=1)
    ).reshape(-1)

    zero_rows = np.flatnonzero(totals <= 0)

    if zero_rows.size:
        if zero_library_rule == "error":
            preview = zero_rows[:20].tolist()
            raise ValueError(
                "zero-library cells after availability/thinning: "
                f"{preview}; n={zero_rows.size}"
            )

        if zero_library_rule != "keep_zero":
            raise ValueError(
                "zero_library_rule must be 'error' or 'keep_zero'"
            )

    scales = np.divide(
        10000.0,
        totals,
        out=np.zeros_like(
            totals,
            dtype=np.float32,
        ),
        where=totals > 0,
    )

    normalized = sparse.diags(scales) @ matrix
    normalized = normalized.tocsr()
    normalized.data = np.log1p(normalized.data)

    output = np.asarray(
        normalized.toarray(),
        dtype=np.float32,
        order="C",
    )

    output[:, ~available] = 0.0

    if not np.all(np.isfinite(output)):
        raise ValueError("normalization produced non-finite values")

    return output


def binomial_thin_counts(
    counts: np.ndarray,
    *,
    q: float,
    seed: int,
    available_gene_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Independently retain each observed molecule with probability q."""

    if not 0.0 < float(q) <= 1.0:
        raise ValueError(
            f"q must be in (0, 1], got {q}"
        )

    array = _counts_array(counts)
    available = _availability(
        available_gene_mask,
        array.shape[1],
    )

    working = array.copy()
    working[:, ~available] = 0

    rng = np.random.default_rng(int(seed))

    thinned = rng.binomial(
        working.astype(np.int64, copy=False),
        float(q),
    )

    thinned = np.asarray(
        thinned,
        dtype=np.uint32,
        order="C",
    )

    thinned[:, ~available] = 0

    if np.any(thinned > working):
        raise AssertionError(
            "binomial thinning increased at least one count"
        )

    return thinned


def thinning_masks(
    full_counts: np.ndarray,
    thinned_counts: np.ndarray,
    available_gene_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lost-positive and originally-zero masks."""

    full = _counts_array(full_counts)
    thin = _counts_array(thinned_counts)

    if full.shape != thin.shape:
        raise ValueError(
            "full_counts and thinned_counts must have equal shapes"
        )

    available = _availability(
        available_gene_mask,
        full.shape[1],
    )[None, :]

    lost_positive = (
        (full > 0)
        & (thin == 0)
        & available
    )

    originally_zero = (
        (full == 0)
        & available
    )

    if np.any(lost_positive & originally_zero):
        raise AssertionError(
            "lost-positive and originally-zero masks overlap"
        )

    return (
        np.asarray(lost_positive, dtype=np.bool_),
        np.asarray(originally_zero, dtype=np.bool_),
    )


def identity_sha256(
    shard_id: np.ndarray,
    row: np.ndarray,
) -> str:
    shards = np.asarray(shard_id).astype(str).reshape(-1)
    rows = np.asarray(row, dtype=np.int64).reshape(-1)

    if len(shards) != len(rows):
        raise ValueError(
            "shard_id and row lengths differ"
        )

    payload = "\n".join(
        f"{shard}\t{int(index)}"
        for shard, index in zip(
            shards.tolist(),
            rows.tolist(),
            strict=True,
        )
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()
