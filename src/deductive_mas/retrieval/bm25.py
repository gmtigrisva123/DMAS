"""Okapi BM25, the sparse half of the hybrid retriever. Kept because a query
with popleft or O(n log n) must match docs with those exact tokens.

    idf(q)      = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
    score(D, Q) = sum_q idf(q) * tf(q,D)(k1 + 1)
                          / (tf(q,D) + k1 (1 - b + b |D| / avgdl))

k1 = tf saturation, b = length normalisation. The +1 in the log keeps idf
non negative for terms in more than half the docs.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass
class BM25Index:
    """Inverted index with BM25 scoring."""

    k1: float = 1.35
    b: float = 0.72
    doc_ids: List[str] = field(default_factory=list)
    doc_lengths: List[int] = field(default_factory=list)
    postings: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    average_length: float = 0.0

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    @classmethod
    def build(
        cls,
        documents: Sequence[Tuple[str, Sequence[str]]],
        *,
        k1: float = 1.35,
        b: float = 0.72,
    ) -> "BM25Index":
        index = cls(k1=k1, b=b)
        for doc_id, tokens in documents:
            position = len(index.doc_ids)
            index.doc_ids.append(doc_id)
            index.doc_lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                index.postings.setdefault(term, []).append((position, count))
        total = sum(index.doc_lengths)
        index.average_length = total / len(index.doc_lengths) if index.doc_lengths else 0.0
        return index

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.size - df + 0.5) / (df + 0.5))

    def score(self, query: Sequence[str]) -> Dict[str, float]:
        """Score every doc that shares at least one term with the query."""
        if not self.doc_ids:
            return {}
        scores: Dict[int, float] = {}
        for term, query_count in Counter(query).items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            if idf <= 0.0:
                continue
            for position, count in postings:
                length = self.doc_lengths[position]
                norm = 1.0 - self.b + self.b * (length / self.average_length if self.average_length else 1.0)
                contribution = idf * (count * (self.k1 + 1.0)) / (count + self.k1 * norm)
                scores[position] = scores.get(position, 0.0) + contribution * min(1.0, query_count)
        return {self.doc_ids[position]: value for position, value in scores.items()}

    def ranked(self, query: Sequence[str], top_k: int = 10) -> List[Tuple[str, float]]:
        scores = self.score(query)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
