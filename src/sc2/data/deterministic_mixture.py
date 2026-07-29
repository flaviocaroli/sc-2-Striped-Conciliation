from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from sc2.data.sharded_expression_dataset import CounterBasedExpressionStream, collate_expression_batch


def iter_batches(
    stream: CounterBasedExpressionStream,
    *,
    batch_size: int,
    start_sample_index: int,
) -> Iterator[dict[str, Any]]:
    """Yield exact, contiguous batches; checkpoint stores the next sample index."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    index = int(start_sample_index)
    while True:
        modality = None
        items: list[dict[str, Any]] = []
        while len(items) < batch_size:
            candidate = stream.sample_at(index)
            index += 1
            if modality is None:
                modality = candidate["modality"]
            if candidate["modality"] == modality:
                items.append(candidate)
        batch = collate_expression_batch(items)
        batch["next_sample_index"] = index
        yield batch
