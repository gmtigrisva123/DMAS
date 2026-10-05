"""Latent semantic indexing, the dense half of the retriever.

No pretrained embeddings (would make the pipeline unauditable), so the
classic way: tf-idf term document matrix, truncated SVD, represent docs and
queries in the latent space. Truncating at k merges terms that co-occur,
which is what lets "cache the answers of repeated calls" find a card titled
memoisation. SVD uses the randomised range finder from util/linalg.py so
the cost is O(n_docs * n_terms * k).
"""

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..util.linalg import Matrix, Vector, cosine, randomised_svd


@dataclass
class LSAIndex:
    """Rank k latent semantic index over a token corpus."""

    doc_ids: List[str] = field(default_factory=list)
    vocabulary: Dict[str, int] = field(default_factory=dict)
    idf: List[float] = field(default_factory=list)
    # n_docs x rank document coordinates (U * Sigma)
    doc_vectors: Matrix = field(default_factory=list)
    # n_terms x rank term loadings (V), used to fold queries in
    term_vectors: Matrix = field(default_factory=list)
    singular_values: Vector = field(default_factory=list)

    @property
    def rank(self) -> int:
        return len(self.singular_values)

    @classmethod
    def build(
        cls,
        documents: Sequence[Tuple[str, Sequence[str]]],
        *,
        components: int = 24,
        power_iterations: int = 3,
        oversampling: int = 8,
        min_document_frequency: int = 1,
        seed: int = 20260909,
    ) -> "LSAIndex":
        index = cls()
        if not documents:
            return index

        frequency: Counter = Counter()
        for _, tokens in documents:
            frequency.update(set(tokens))
        terms = sorted(t for t, df in frequency.items() if df >= min_document_frequency)
        if not terms:
            return index
        index.vocabulary = {term: i for i, term in enumerate(terms)}
        n_docs = len(documents)
        index.idf = [
            math.log((1.0 + n_docs) / (1.0 + frequency[term])) + 1.0 for term in terms
        ]

        matrix: Matrix = []
        for doc_id, tokens in documents:
            index.doc_ids.append(doc_id)
            matrix.append(index._vectorise(tokens))

        rank = max(1, min(components, n_docs, len(terms)))
        u, s, v = randomised_svd(
            matrix,
            rank,
            power_iterations=power_iterations,
            oversampling=oversampling,
            rng=random.Random(seed),
        )
        index.singular_values = s
        index.doc_vectors = [[value * s[j] for j, value in enumerate(row)] for row in u]
        index.term_vectors = v
        return index

    # internals
    def _vectorise(self, tokens: Sequence[str]) -> Vector:
        """Sublinear tf times idf, L2 normalised."""
        counts = Counter(t for t in tokens if t in self.vocabulary)
        vector = [0.0] * len(self.vocabulary)
        for term, count in counts.items():
            position = self.vocabulary[term]
            vector[position] = (1.0 + math.log(count)) * self.idf[position]
        norm = math.sqrt(math.fsum(x * x for x in vector))
        if norm > 1e-12:
            vector = [x / norm for x in vector]
        return vector

    def fold_in(self, tokens: Sequence[str]) -> Vector:
        """Project a query into the latent space: q^T V."""
        if not self.term_vectors:
            return []
        sparse = self._vectorise(tokens)
        rank = len(self.term_vectors[0])
        out = [0.0] * rank
        for position, weight in enumerate(sparse):
            if weight == 0.0:
                continue
            row = self.term_vectors[position]
            for j in range(rank):
                out[j] += weight * row[j]
        return out

    # queries
    def score(self, tokens: Sequence[str]) -> Dict[str, float]:
        query = self.fold_in(tokens)
        if not query:
            return {}
        return {
            doc_id: cosine(query, vector)
            for doc_id, vector in zip(self.doc_ids, self.doc_vectors)
        }

    def ranked(self, tokens: Sequence[str], top_k: int = 10) -> List[Tuple[str, float]]:
        scores = self.score(tokens)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

    def similarity(self, a: str, b: str) -> float:
        try:
            i, j = self.doc_ids.index(a), self.doc_ids.index(b)
        except ValueError:
            return 0.0
        return cosine(self.doc_vectors[i], self.doc_vectors[j])

    def vector(self, doc_id: str) -> Vector:
        try:
            return self.doc_vectors[self.doc_ids.index(doc_id)]
        except ValueError:
            return []
