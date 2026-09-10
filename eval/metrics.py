"""Pure paper-level retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _unique_at_k(ranking: Sequence[str], k: int) -> tuple[str, ...]:
    if k <= 0:
        raise ValueError("k must be positive.")
    unique = []
    seen = set()
    for paper_id in ranking:
        if paper_id not in seen:
            seen.add(paper_id)
            unique.append(paper_id)
        if len(unique) == k:
            break
    return tuple(unique)


def recall_at_k(ranking: Sequence[str], relevant: set[str], *, k: int) -> float | None:
    if not relevant:
        return None
    retrieved = set(_unique_at_k(ranking, k))
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank_at_k(ranking: Sequence[str], relevant: set[str], *, k: int) -> float | None:
    if not relevant:
        return None
    for rank, paper_id in enumerate(_unique_at_k(ranking, k), start=1):
        if paper_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], *, k: int) -> float | None:
    if not relevant:
        return None
    ranked = _unique_at_k(ranking, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, paper_id in enumerate(ranked, start=1)
        if paper_id in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg
