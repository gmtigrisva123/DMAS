"""Hybrid retrieval index.

BM25 cannot connect "cache the results of repeated calls" to a card called
memoisation, LSA cannot guarantee that a query with popleft returns the
card that actually says popleft. Run both, fuse the ranks (not the scores,
different scales), then MMR so six cards cover six ideas.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import RetrievalConfig
from ..errors import RetrievalError
from ..util.text import prose_tokens
from .bm25 import BM25Index
from .corpus import KnowledgeCard, cards
from .fusion import maximal_marginal_relevance, reciprocal_rank_fusion
from .lsa import LSAIndex


def _trigrams(term: str) -> set:
    """Character trigrams of a padded term (unit for fuzzy matching)."""
    padded = f"${term}$"
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


@dataclass(frozen=True)
class RetrievedCard:
    """A card plus why it was retrieved."""

    card: KnowledgeCard
    score: float
    rank: int
    lexical: float = 0.0
    semantic: float = 0.0
    concept_boost: float = 0.0

    @property
    def card_id(self) -> str:
        return self.card.card_id

    def __str__(self) -> str:
        return f"{self.card.card_id}({self.score:.3f})"


class HybridIndex:
    """BM25 + LSA over the grounding corpus."""

    def __init__(
        self,
        documents: Optional[Sequence[KnowledgeCard]] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.config = config or RetrievalConfig()
        self.cards: List[KnowledgeCard] = list(documents if documents is not None else cards())
        if not self.cards:
            raise RetrievalError("the grounding corpus is empty")
        self._by_id = {card.card_id: card for card in self.cards}
        tokenised = [(card.card_id, prose_tokens(card.searchable)) for card in self.cards]
        self.bm25 = BM25Index.build(tokenised, k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.lsa = LSAIndex.build(
            tokenised,
            components=self.config.lsa_components,
            power_iterations=self.config.lsa_power_iterations,
            oversampling=self.config.lsa_oversampling,
        )
        self._concept_index: Dict[str, List[str]] = {}
        for card in self.cards:
            for concept in card.concepts:
                self._concept_index.setdefault(concept, []).append(card.card_id)

        # trigram postings over the vocabulary, for fuzzy expansion of query terms
        # the stemmer cannot match
        self._trigrams: Dict[str, List[str]] = {}
        for term in self.lsa.vocabulary:
            for gram in _trigrams(term):
                self._trigrams.setdefault(gram, []).append(term)

    # accessors
    def __len__(self) -> int:
        return len(self.cards)

    def card(self, card_id: str) -> KnowledgeCard:
        try:
            return self._by_id[card_id]
        except KeyError as exc:
            raise RetrievalError(f"unknown card: {card_id!r}") from exc

    def cards_for_concepts(self, concepts: Iterable[str]) -> List[str]:
        out: List[str] = []
        for concept in concepts:
            out.extend(self._concept_index.get(concept, ()))
        return list(dict.fromkeys(out))

    def expand(self, tokens: Sequence[str]) -> List[str]:
        """Add a fuzzy synonym for each out of vocabulary query term.
        Suffix stripping cannot relate every form to its stem ("revisits" shares
        no token with "visited"), so an unmatched term is matched against the
        vocabulary by trigram overlap. Catches morphological near misses and
        nothing fancier.
        """
        out = list(tokens)
        for token in tokens:
            if token in self.lsa.vocabulary or len(token) < 5:
                continue
            grams = _trigrams(token)
            if not grams:
                continue
            counts: Dict[str, int] = {}
            for gram in grams:
                for candidate in self._trigrams.get(gram, ()):
                    counts[candidate] = counts.get(candidate, 0) + 1
            scored = []
            for candidate, shared in counts.items():
                union = len(grams) + len(_trigrams(candidate)) - shared
                similarity = shared / union if union else 0.0
                if similarity >= 0.45:
                    scored.append((similarity, candidate))
            # keep the two best, not one: near misses tie all the time (revisit is
            # equally far from revis and visit) and dropping one loses the useful match
            # half the time. Two extra terms cost nothing here.
            scored.sort(key=lambda pair: (-pair[0], pair[1]))
            out.extend(candidate for _, candidate in scored[:2])
        return out

    def _redundancy(self, a: str, b: str) -> float:
        """Similarity for diversification, discounted across card kinds.
        A procedure card for BFS and a pitfall card about when to mark a vertex
        share vocabulary but not information. Plain cosine would drop the second
        one, scaling it down when kinds differ keeps complementary cards and still
        removes real paraphrases.
        """
        similarity = self.lsa.similarity(a, b)
        if similarity <= 0.0:
            return 0.0
        left, right = self._by_id.get(a), self._by_id.get(b)
        if left is not None and right is not None and left.kind != right.kind:
            similarity *= 0.55
        return similarity

    # search
    def search(
        self,
        query: str,
        *,
        concepts: Sequence[str] = (),
        top_k: Optional[int] = None,
        diversify: bool = True,
    ) -> List[RetrievedCard]:
        """Retrieve cards for a query, optionally biased by concepts."""
        k = top_k or self.config.top_k
        tokens = self.expand(prose_tokens(query))
        if not tokens and not concepts:
            return []

        pool = self.config.candidate_pool
        lexical = self.bm25.ranked(tokens, top_k=pool)
        semantic = self.lsa.ranked(tokens, top_k=pool)

        rankings: List[Sequence[Tuple[str, float]]] = [lexical, semantic]
        weights = [1.0, 0.9]
        if concepts:
            # concept membership is an exact structural signal, treat it as a third
            # ranking instead of a score bonus so it fuses on equal footing
            matched = self.cards_for_concepts(concepts)
            rankings.append([(cid, 1.0) for cid in matched[:pool]])
            weights.append(1.1)

        fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k, weights=weights)
        if not fused:
            return []

        if diversify:
            selected = maximal_marginal_relevance(
                fused,
                self._redundancy,
                lambda_=self.config.mmr_lambda,
                top_k=k,
            )
        else:
            selected = fused[:k]

        lexical_map = dict(lexical)
        semantic_map = dict(semantic)
        fused_map = dict(fused)
        concept_set = set(concepts)
        out: List[RetrievedCard] = []
        for rank, (card_id, _mmr) in enumerate(selected, start=1):
            card = self._by_id[card_id]
            out.append(
                RetrievedCard(
                    card=card,
                    score=fused_map.get(card_id, 0.0),
                    rank=rank,
                    lexical=lexical_map.get(card_id, 0.0),
                    semantic=semantic_map.get(card_id, 0.0),
                    concept_boost=1.0 if concept_set & set(card.concepts) else 0.0,
                )
            )
        return out
