"""Rank fusion + diversification.

BM25 scores and cosine similarities are on different scales so they cannot
be added. Reciprocal rank fusion (Cormack, Clarke & Buettcher 2009) drops
the scores and combines ranks:

    RRF(d) = sum over rankings r of 1 / (k + rank_r(d))

Then maximal marginal relevance (Carbonell & Goldstein 1998) trades
relevance against redundancy so the context is not five paraphrases of the
same card.
"""

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Ranking = Sequence[Tuple[str, float]]


def reciprocal_rank_fusion(
    rankings: Sequence[Ranking],
    *,
    k: float = 60.0,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Fuse ranked lists by reciprocal rank, ties broken by doc id."""
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")
    fused: Dict[str, float] = {}
    for index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[index]
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def maximal_marginal_relevance(
    candidates: Ranking,
    similarity: Callable[[str, str], float],
    *,
    lambda_: float = 0.72,
    top_k: int = 6,
) -> List[Tuple[str, float]]:
    """Greedy pick of a relevant and mutually dissimilar subset.

        MMR(d) = lambda * relevance(d) - (1 - lambda) * max_similarity(d, selected)

    Relevance is rescaled to [0, 1] first so both terms are comparable.
    """
    if not candidates:
        return []
    values = [score for _, score in candidates]
    low, high = min(values), max(values)
    span = high - low
    normalised = {
        doc_id: (1.0 if span <= 1e-12 else (score - low) / span)
        for doc_id, score in candidates
    }

    remaining = [doc_id for doc_id, _ in candidates]
    selected: List[Tuple[str, float]] = []
    while remaining and len(selected) < top_k:
        best_id: Optional[str] = None
        best_value = float("-inf")
        for doc_id in remaining:
            redundancy = max(
                (similarity(doc_id, chosen) for chosen, _ in selected), default=0.0
            )
            value = lambda_ * normalised[doc_id] - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_id, best_value = doc_id, value
        if best_id is None:
            break
        selected.append((best_id, best_value))
        remaining.remove(best_id)
    return selected
