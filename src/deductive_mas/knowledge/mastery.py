"""Bayesian knowledge tracing on the prerequisite graph.

Standard BKT (Corbett & Anderson 1995): one latent binary skill per
concept, four params (prior, learn, slip, guess), updated from
correct/incorrect observations. Two extensions:

- soft evidence: a diagnosis comes with a belief in [0, 1], so the update is
  a convex mix between the prior and the hard evidence posterior. Weight 1
  is textbook BKT.
- prerequisite coupling: after every update the posterior is swept in
  topological order and capped by a weakest link ceiling from its
  prerequisites. You cannot be credited with tabulation while the transition
  order is shaky.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import KnowledgeConfig
from .graph import KnowledgeGraph

_EPS = 1e-9


@dataclass(frozen=True)
class BKTParameters:
    """The four BKT params plus an optional forgetting rate."""

    prior: float = 0.30
    learn: float = 0.20
    slip: float = 0.10
    guess: float = 0.20
    forget: float = 0.0

    def validated(self) -> "BKTParameters":
        if not 0.0 <= self.slip < 0.5:
            raise ValueError("slip must lie in [0, 0.5)")
        if not 0.0 <= self.guess < 0.5:
            raise ValueError("guess must lie in [0, 0.5)")
        if self.slip + self.guess >= 1.0:
            raise ValueError("slip + guess must be below 1 for the update to be informative")
        return self


@dataclass
class MasteryState:
    """Per concept posterior of mastery, with an evidence count."""

    posterior: Dict[str, float] = field(default_factory=dict)
    observations: Dict[str, int] = field(default_factory=dict)

    def copy(self) -> "MasteryState":
        return MasteryState(posterior=dict(self.posterior), observations=dict(self.observations))

    def get(self, cid: str, default: float = 0.0) -> float:
        return self.posterior.get(cid, default)

    def weakest(self, n: int = 5) -> List[Tuple[str, float]]:
        return sorted(self.posterior.items(), key=lambda kv: (kv[1], kv[0]))[:n]

    def mean(self) -> float:
        return sum(self.posterior.values()) / len(self.posterior) if self.posterior else 0.0


class MasteryTracker:
    """Keeps a MasteryState over a KnowledgeGraph."""

    def __init__(self, graph: KnowledgeGraph, config: Optional[KnowledgeConfig] = None):
        self.graph = graph
        self.config = config or KnowledgeConfig()
        self.defaults = BKTParameters(
            prior=self.config.bkt_prior,
            learn=self.config.bkt_learn,
            slip=self.config.bkt_slip,
            guess=self.config.bkt_guess,
        ).validated()

    # creation
    def parameters(self, cid: str) -> BKTParameters:
        concept = self.graph.concept(cid)
        return replace(
            self.defaults,
            prior=self.defaults.prior if concept.prior is None else concept.prior,
            learn=self.defaults.learn if concept.learn_rate is None else concept.learn_rate,
        )

    def initial(self, ability: float = 0.0) -> MasteryState:
        """Cold start priors given the difficulty being attempted.

        A flat prior makes no sense here: a student who wrote a valid binary
        search has shown they know variables, ifs and loops. So mastery starts at
        a Rasch probability of ability > difficulty, squeezed away from 0 and 1
        so evidence can still move it:

            P(mastered) = floor + (1 - floor - headroom) * sigma(k (theta - b))

        ability = problem.difficulty encodes "a learner attempting a task is
        roughly at its level".
        """
        floor, headroom, slope = 0.15, 0.05, 1.1
        span = 1.0 - floor - headroom
        posterior: Dict[str, float] = {}
        for concept in self.graph:
            prior = concept.prior
            if prior is not None:
                posterior[concept.cid] = _clamp(prior)
                continue
            posterior[concept.cid] = _clamp(
                floor + span * _logistic(slope * (ability - concept.difficulty))
            )
        return MasteryState(posterior=posterior, observations={cid: 0 for cid in self.graph.ids})

    # update
    def observe(
        self,
        state: MasteryState,
        cid: str,
        correct: bool,
        *,
        weight: float = 1.0,
        learn: bool = True,
    ) -> MasteryState:
        """Apply one (possibly soft) observation to one concept."""
        if cid not in self.graph:
            return state
        # a weight is not a probability, it must be allowed to be exactly 0 or
        # "no evidence" becomes "a little evidence"
        weight = max(0.0, min(1.0, weight))
        if weight <= _EPS:
            return state
        params = self.parameters(cid)
        prior = state.posterior.get(cid, params.prior)

        if correct:
            likely = prior * (1.0 - params.slip)
            unlikely = (1.0 - prior) * params.guess
        else:
            likely = prior * params.slip
            unlikely = (1.0 - prior) * (1.0 - params.guess)
        denominator = likely + unlikely
        conditioned = prior if denominator <= _EPS else likely / denominator

        if learn and correct:
            conditioned = conditioned + (1.0 - conditioned) * params.learn
        if params.forget > 0.0:
            conditioned *= 1.0 - params.forget

        updated = state.copy()
        updated.posterior[cid] = _clamp((1.0 - weight) * prior + weight * conditioned)
        updated.observations[cid] = updated.observations.get(cid, 0) + 1
        return updated

    def apply_blame(
        self,
        state: MasteryState,
        blame: Mapping[str, float],
        *,
        intensity: float = 1.0,
    ) -> MasteryState:
        """Fold the blame distribution in as soft negative evidence.
        Blame is a distribution over concepts so the values are tiny on a big
        graph, it is rescaled by its max first. What matters is the relative
        implication of each concept.
        """
        if not blame:
            return state
        peak = max(blame.values())
        if peak <= _EPS:
            return state
        updated = state
        for cid, mass in sorted(blame.items(), key=lambda kv: -kv[1]):
            weight = max(0.0, min(1.0, intensity * mass / peak))
            if weight < 0.02:
                continue
            updated = self.observe(updated, cid, correct=False, weight=weight)
        return self.regularise(updated)

    def apply_success(
        self,
        state: MasteryState,
        concepts: Iterable[str],
        *,
        weight: float = 1.0,
    ) -> MasteryState:
        updated = state
        for cid in concepts:
            updated = self.observe(updated, cid, correct=True, weight=weight)
        return self.regularise(updated)

    # regularisation
    def regularise(self, state: MasteryState) -> MasteryState:
        """Cap each posterior by a weakest link ceiling from its prerequisites.
        Topological order means a prerequisite is final before anything that
        depends on it, so one pass is enough.
        """
        coupling = _clamp(self.config.prerequisite_coupling)
        if coupling <= _EPS:
            return state
        updated = state.copy()
        for cid in self.graph.topological_order():
            prerequisites = self.graph.prerequisites(cid)
            if not prerequisites:
                continue
            support = min(updated.posterior.get(p, 0.0) for p in prerequisites)
            ceiling = 1.0 - coupling * (1.0 - support)
            current = updated.posterior.get(cid, 0.0)
            if current > ceiling:
                updated.posterior[cid] = _clamp(ceiling)
        return updated

    # readout
    def unmastered(self, state: MasteryState, *, threshold: Optional[float] = None) -> Tuple[str, ...]:
        limit = self.config.mastery_threshold if threshold is None else threshold
        return tuple(sorted(cid for cid, value in state.posterior.items() if value < limit))

    def frontier(
        self,
        state: MasteryState,
        *,
        candidates: Optional[Iterable[str]] = None,
        threshold: Optional[float] = None,
        background: Optional[MasteryState] = None,
    ) -> Tuple[str, ...]:
        """Root cause concepts. background = the pre diagnosis belief state."""
        limit = self.config.mastery_threshold if threshold is None else threshold
        return self.graph.deficiency_frontier(
            state.posterior,
            threshold=limit,
            candidates=candidates,
            prerequisite_mastery=background.posterior if background is not None else None,
        )


def _clamp(value: float, low: float = 0.001, high: float = 0.999) -> float:
    return max(low, min(high, value))


def _logistic(x: float) -> float:
    import math

    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
