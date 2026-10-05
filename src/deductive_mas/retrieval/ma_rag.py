"""Training free multi agent RAG. No learned parameters anywhere, behaviour is
fully determined by the corpus, the index and this code.

Three roles, each a pure function of its input:

- Planner: turns a diagnosis into typed sub queries. Structural, not
  generative (a non_termination divergence always asks about termination
  measures), which keeps retrieval reproducible.
- Retriever: runs each sub query on the hybrid index and merges the rankings
  with RRF, so a card supporting several sub queries wins.
- Verifier: checks every assertive sentence of the generated text against the
  retrieved evidence and reports what is unsupported.

Questions and statements about the student's own code are exempt: "what is
hi after the first iteration?" claims nothing about algorithms. Grounding is
only demanded where a hallucination would do damage.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import RetrievalConfig
from ..util.linalg import cosine
from ..util.text import prose_tokens, sentences
from .fusion import reciprocal_rank_fusion
from .index import HybridIndex, RetrievedCard


# planning
@dataclass(frozen=True)
class SubQuery:
    """One typed information need derived from the diagnosis."""

    purpose: str
    text: str
    concepts: Tuple[str, ...] = ()
    weight: float = 1.0


# sub query templates keyed by the divergence kind that triggers them
_DIVERGENCE_QUERIES: Mapping[str, str] = {
    "non_termination": "what guarantees a loop or recursion terminates, decreasing measure",
    "early_termination": "boundary conditions, the last element, empty and single-element inputs",
    "control_mismatch": "loop invariant maintenance, extra or missing iteration",
    "state_mismatch": "loop invariant, correct update of bounds and accumulators",
    "exception": "index range validity, valid indices of a sequence of length n",
    "output_mismatch": "what the algorithm must return, definition and postcondition",
}


class Planner:
    """Turns a diagnosis into a few complementary sub queries."""

    def plan(
        self,
        *,
        problem_title: str = "",
        misconception_ids: Sequence[str] = (),
        misconception_texts: Sequence[str] = (),
        concepts: Sequence[str] = (),
        frontier: Sequence[str] = (),
        divergence_kind: Optional[str] = None,
        complexity_gap: Optional[Tuple[str, str]] = None,
    ) -> List[SubQuery]:
        queries: List[SubQuery] = []

        for text in misconception_texts[:2]:
            queries.append(SubQuery("misconception", text, tuple(concepts), 1.2))

        if divergence_kind and divergence_kind in _DIVERGENCE_QUERIES:
            queries.append(
                SubQuery("divergence", _DIVERGENCE_QUERIES[divergence_kind], tuple(concepts), 1.0)
            )

        if complexity_gap:
            student, reference = complexity_gap
            queries.append(
                SubQuery(
                    "complexity",
                    f"why an algorithm costs {student} instead of {reference}, "
                    "composing loop costs and hidden linear operations",
                    tuple(concepts),
                    0.9,
                )
            )

        if frontier:
            queries.append(
                SubQuery(
                    "prerequisite",
                    "definition and invariant of " + ", ".join(frontier[:3]).replace("-", " "),
                    tuple(frontier[:3]),
                    1.1,
                )
            )

        if problem_title:
            queries.append(SubQuery("problem", problem_title, tuple(concepts), 0.6))

        if not queries:
            queries.append(SubQuery("fallback", " ".join(misconception_ids) or "algorithm", tuple(concepts), 1.0))
        return queries


# retrieval
@dataclass
class GroundedContext:
    """The evidence bundle passed to the generation stage."""

    cards: Tuple[RetrievedCard, ...] = ()
    subqueries: Tuple[SubQuery, ...] = ()

    @property
    def citations(self) -> Tuple[str, ...]:
        return tuple(card.card.card_id for card in self.cards)

    def render(self, *, max_chars: int = 2600) -> str:
        """Format the evidence for a prompt, within a budget."""
        blocks: List[str] = []
        used = 0
        for retrieved in self.cards:
            card = retrieved.card
            block = f"[{card.card_id}] ({card.kind}) {card.title}\n{card.text}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def concept_coverage(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for retrieved in self.cards:
            for concept in retrieved.card.concepts:
                if concept not in seen:
                    seen.append(concept)
        return tuple(seen)


class Retriever:
    """Runs the plan on the hybrid index and fuses the per query results."""

    def __init__(self, index: HybridIndex, config: Optional[RetrievalConfig] = None):
        self.index = index
        self.config = config or index.config

    def retrieve(self, plan: Sequence[SubQuery], *, top_k: Optional[int] = None) -> GroundedContext:
        if not plan:
            return GroundedContext()
        k = top_k or self.config.top_k
        rankings: List[Sequence[Tuple[str, float]]] = []
        weights: List[float] = []
        seen: Dict[str, RetrievedCard] = {}
        for query in plan:
            hits = self.index.search(
                query.text,
                concepts=query.concepts,
                top_k=max(k, 4),
                diversify=False,
            )
            if not hits:
                continue
            rankings.append([(hit.card_id, hit.score) for hit in hits])
            weights.append(query.weight)
            for hit in hits:
                seen.setdefault(hit.card_id, hit)
        if not rankings:
            return GroundedContext(subqueries=tuple(plan))

        fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k, weights=weights)
        from .fusion import maximal_marginal_relevance

        selected = maximal_marginal_relevance(
            fused, self.index._redundancy, lambda_=self.config.mmr_lambda, top_k=k
        )
        cards = []
        fused_scores = dict(fused)
        for rank, (card_id, _value) in enumerate(selected, start=1):
            base = seen[card_id]
            cards.append(
                RetrievedCard(
                    card=base.card,
                    score=fused_scores.get(card_id, base.score),
                    rank=rank,
                    lexical=base.lexical,
                    semantic=base.semantic,
                    concept_boost=base.concept_boost,
                )
            )
        return GroundedContext(cards=tuple(cards), subqueries=tuple(plan))


# verification
_QUESTION_RE = re.compile(r"\?\s*$")
_SELF_REFERENCE_RE = re.compile(
    r"\b(your|you|line \d+|iteration|trace|walk through|what happens|notice|compare|hand)\b",
    re.IGNORECASE,
)

# reflective imperatives ("count the distinct states", "trace the loop by
# hand") only direct attention, they have no truth value, so asking a card
# to entail one makes no sense and would delete the useful content. Exempt.
#
# code action verbs ("use", "set", "replace", "change", "add",
# "initialise"...) are NOT in this list on purpose: they propose an edit so
# they stay assertions and face both the grounding check and the leak gate.
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:then\s+|now\s+|first\s+|next\s+)?"
    r"(state|count|write|check|compare|draw|list|take|trace|name|identify|derive|"
    r"revisit|try|pick|substitute|iterate|multiply|assume|predict|work|step|walk|ask|"
    r"read|explain|show|find|choose|consider|note|observe|calculate|evaluate|verify|"
    r"describe|record|follow|separate|distinguish|prove|reason|sketch|enumerate|treat|"
    r"track|hand-execute|re-read|look|examine|inspect|focus|recall|remember|imagine|"
    r"suppose|think|reflect|run|re-run|rerun|test|experiment|print)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    kind: str                 # "question" | "self-reference" | "assertion"
    score: float
    supported: bool
    evidence: Optional[str] = None


@dataclass
class GroundingReport:
    """How well a piece of generated text is supported by the cards."""

    score: float = 1.0
    checks: Tuple[ClaimCheck, ...] = ()

    @property
    def assertions(self) -> Tuple[ClaimCheck, ...]:
        return tuple(c for c in self.checks if c.kind == "assertion")

    @property
    def unsupported(self) -> Tuple[ClaimCheck, ...]:
        """Every checked claim that failed."""
        return tuple(c for c in self.checks if not c.supported)

    def passes(self, threshold: float) -> bool:
        return self.score >= threshold


class Verifier:
    """Scores generated text against the retrieved evidence."""

    def __init__(self, index: HybridIndex, config: Optional[RetrievalConfig] = None):
        self.index = index
        self.config = config or index.config

    def classify(self, claim: str) -> str:
        if _QUESTION_RE.search(claim):
            return "question"
        if _SELF_REFERENCE_RE.search(claim):
            return "self-reference"
        if _IMPERATIVE_RE.match(claim):
            return "instruction"
        return "assertion"

    def unknown_vocabulary_ratio(self, claim: str) -> float:
        """Share of a sentence's informative weight carried by unseen terms."""
        tokens = set(prose_tokens(claim))
        if not tokens:
            return 0.0
        total = sum(self._term_weight(t) for t in tokens) or 1.0
        unknown = sum(
            self._term_weight(t) for t in tokens if self.index.bm25.idf(t) <= 0.0
        )
        return unknown / total

    # how the two support channels are blended. Lexical dominates on purpose,
    # latent similarity measures topic and a made up claim about binary search
    # is on topic by construction.
    LEXICAL_WEIGHT = 0.68
    SEMANTIC_WEIGHT = 0.32

    def _term_weight(self, term: str) -> float:
        """IDF of a term, unseen terms count as maximally informative.
        This is the important bit: scoring only over terms the corpus contains
        would let a claim smuggle in invented vocabulary for free ("rebuilds the
        heap on every comparison" would be judged on the words it shares). A term
        the corpus never saw cannot be grounded in it, so it goes into the
        denominator at max weight and never into the numerator.
        """
        idf = self.index.bm25.idf(term)
        if idf > 0.0:
            return idf
        return math.log(1.0 + (self.index.bm25.size + 0.5) / 0.5)

    def support(self, claim: str, context: GroundedContext) -> Tuple[float, Optional[str]]:
        """Best support score for a claim over the retrieved cards.
        Two channels like in retrieval: idf weighted lexical overlap for the
        technical vocabulary and latent cosine for paraphrase. Blended, not max:
        max would let a merely on topic sentence pass on the semantic channel
        alone, which is exactly the failure this check is for.
        """
        tokens = prose_tokens(claim)
        if not tokens or not context.cards:
            return 0.0, None
        query_vector = self.index.lsa.fold_in(tokens)
        token_set = set(tokens)
        total = sum(self._term_weight(t) for t in token_set) or 1.0

        best_score = 0.0
        best_card: Optional[str] = None
        for retrieved in context.cards:
            card = retrieved.card
            card_tokens = set(prose_tokens(card.searchable))
            if not card_tokens:
                continue
            overlap = sum(self._term_weight(t) for t in token_set & card_tokens)
            lexical = overlap / total

            semantic = 0.0
            if query_vector:
                semantic = max(0.0, cosine(query_vector, self.index.lsa.vector(card.card_id)))
            score = self.LEXICAL_WEIGHT * lexical + self.SEMANTIC_WEIGHT * semantic
            if score > best_score:
                best_score, best_card = score, card.card_id
        return best_score, best_card

    def verify(self, text: str, context: GroundedContext) -> GroundingReport:
        claims = sentences(text)
        if not claims:
            return GroundingReport(score=1.0, checks=())
        checks: List[ClaimCheck] = []
        for claim in claims:
            kind = self.classify(claim)
            if kind != "assertion":
                checks.append(ClaimCheck(claim=claim, kind=kind, score=1.0, supported=True))
                continue
            score, evidence = self.support(claim, context)
            checks.append(
                ClaimCheck(
                    claim=claim,
                    kind=kind,
                    score=score,
                    supported=score >= self.config.min_grounding_score,
                    evidence=evidence,
                )
            )
        scored = [c for c in checks if c.kind == "assertion"]
        if not scored:
            # pure Socratic output, nothing asserted so nothing can be ungrounded.
            # Report full grounding instead of a meaningless zero.
            return GroundingReport(score=1.0, checks=tuple(checks))
        score = sum(c.score for c in scored) / len(scored)
        return GroundingReport(score=score, checks=tuple(checks))


# facade
class MultiAgentRAG:
    """Planner + Retriever + Verifier in one object."""

    def __init__(self, index: Optional[HybridIndex] = None, config: Optional[RetrievalConfig] = None):
        self.index = index or HybridIndex(config=config)
        self.config = config or self.index.config
        self.planner = Planner()
        self.retriever = Retriever(self.index, self.config)
        self.verifier = Verifier(self.index, self.config)

    def ground(self, **plan_kwargs) -> GroundedContext:
        return self.retriever.retrieve(self.planner.plan(**plan_kwargs))

    def verify(self, text: str, context: GroundedContext) -> GroundingReport:
        return self.verifier.verify(text, context)
