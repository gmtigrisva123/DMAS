"""Zone of proximal development with IRT.

The ZPD is the band between what the learner can do alone and what they
cannot do even with help. With IRT that is: tasks whose success probability
at the current ability falls in a target window.

Here: 3PL response model + Fisher information, EAP ability estimate (grid
with a standard normal prior, MLE is undefined for all correct / all wrong
which is the normal case at the start of a session), and ZPD constrained
item selection (greedy on a submodular objective so the picked set is
diverse).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..config import InterventionConfig

_GRID = tuple(-4.0 + 0.1 * i for i in range(81))          # theta in [-4, 4]
_PRIOR = tuple(math.exp(-0.5 * t * t) for t in _GRID)     # standard normal


@dataclass(frozen=True)
class Item:
    """A task on the same latent ability scale as the learner."""

    iid: str
    difficulty: float = 0.0        # b
    discrimination: float = 1.2    # a
    guessing: float = 0.05         # c
    concepts: Tuple[str, ...] = ()
    title: str = ""

    def probability(self, ability: float) -> float:
        return probability_correct(ability, self)

    def information(self, ability: float) -> float:
        return fisher_information(ability, self)


def probability_correct(ability: float, item: Item) -> float:
    """3PL response function."""
    z = item.discrimination * (ability - item.difficulty)
    logistic = 1.0 / (1.0 + math.exp(-z)) if z >= -700 else 0.0
    return item.guessing + (1.0 - item.guessing) * logistic


def fisher_information(ability: float, item: Item) -> float:
    """Fisher information of an item at ability. For 3PL:

        I(theta) = a^2 * (P - c)^2 * (1 - P) / ((1 - c)^2 * P)

    Peaks a bit above the item difficulty and vanishes for items that are far
    too easy or too hard, which is why an informative item and a ZPD item are
    nearly the same thing.
    """
    p = probability_correct(ability, item)
    if p <= 1e-9 or p >= 1.0 - 1e-9:
        return 0.0
    c = item.guessing
    numerator = (item.discrimination ** 2) * ((p - c) ** 2) * (1.0 - p)
    denominator = ((1.0 - c) ** 2) * p
    return numerator / denominator if denominator > 1e-12 else 0.0


@dataclass(frozen=True)
class AbilityEstimate:
    mean: float
    sd: float
    responses: int

    def __str__(self) -> str:
        return f"theta={self.mean:+.2f}+-{self.sd:.2f} (n={self.responses})"


def estimate_ability(responses: Sequence[Tuple[Item, bool]]) -> AbilityEstimate:
    """EAP ability estimate over a grid. Always defined, with no responses it
    returns the prior.
    """
    if not responses:
        return AbilityEstimate(mean=0.0, sd=1.0, responses=0)

    posterior: List[float] = []
    for theta, prior in zip(_GRID, _PRIOR):
        likelihood = prior
        for item, correct in responses:
            p = probability_correct(theta, item)
            likelihood *= p if correct else (1.0 - p)
            if likelihood < 1e-300:
                break
        posterior.append(likelihood)

    total = math.fsum(posterior)
    if total <= 0.0:
        return AbilityEstimate(mean=0.0, sd=1.0, responses=len(responses))
    mean = math.fsum(t * w for t, w in zip(_GRID, posterior)) / total
    variance = math.fsum(((t - mean) ** 2) * w for t, w in zip(_GRID, posterior)) / total
    return AbilityEstimate(mean=mean, sd=math.sqrt(max(0.0, variance)), responses=len(responses))


def ability_from_mastery(mastery: Mapping[str, float], difficulties: Mapping[str, float]) -> float:
    """Project a per concept mastery vector onto one ability axis.
    Each concept is a Rasch item the learner "passed" with the given
    probability, ability = the value that best explains the whole vector (grid
    search on the log likelihood).
    """
    shared = [cid for cid in mastery if cid in difficulties]
    if not shared:
        return 0.0
    best_theta, best_score = 0.0, -math.inf
    for theta in _GRID:
        score = 0.0
        for cid in shared:
            p = 1.0 / (1.0 + math.exp(-(theta - difficulties[cid])))
            observed = min(0.999, max(0.001, mastery[cid]))
            score += observed * math.log(max(p, 1e-12)) + (1.0 - observed) * math.log(max(1.0 - p, 1e-12))
        if score > best_score:
            best_theta, best_score = theta, score
    return best_theta


# selection
@dataclass(frozen=True)
class ScoredItem:
    item: Item
    probability: float
    information: float
    relevance: float
    utility: float
    in_band: bool


class ZPDSelector:
    """Picks the next task: in the ZPD, informative and on target."""

    def __init__(self, config: Optional[InterventionConfig] = None):
        self.config = config or InterventionConfig()

    def band_weight(self, probability: float) -> float:
        """1 inside the ZPD, decaying smoothly outside.
        A hard window would make p = 0.44 worth nothing and p = 0.45 worth
        everything, so it tapers over the same width as the band.
        """
        low, high = self.config.zpd_low, self.config.zpd_high
        if low <= probability <= high:
            return 1.0
        width = max(1e-6, high - low)
        distance = (low - probability) if probability < low else (probability - high)
        return max(0.0, 1.0 - (distance / width) ** 2)

    def score(
        self,
        items: Sequence[Item],
        ability: float,
        relevance: Mapping[str, float] = (),
    ) -> List[ScoredItem]:
        """Score every item: ZPD fit x information x concept relevance."""
        weights = dict(relevance or {})
        peak = max(weights.values()) if weights else 0.0
        scored: List[ScoredItem] = []
        for item in items:
            probability = probability_correct(ability, item)
            information = fisher_information(ability, item)
            if peak > 0.0 and item.concepts:
                mass = sum(weights.get(c, 0.0) for c in item.concepts) / (peak * len(item.concepts))
            else:
                mass = 0.0
            band = self.band_weight(probability)
            utility = band * (0.35 + information) * (0.25 + mass)
            scored.append(
                ScoredItem(
                    item=item,
                    probability=probability,
                    information=information,
                    relevance=mass,
                    utility=utility,
                    in_band=self.config.zpd_low <= probability <= self.config.zpd_high,
                )
            )
        return sorted(scored, key=lambda s: (-s.utility, s.item.iid))

    def select(
        self,
        items: Sequence[Item],
        ability: float,
        relevance: Mapping[str, float] = (),
        *,
        k: int = 3,
    ) -> List[ScoredItem]:
        """Greedy submodular maximisation with a diversity penalty.
        The marginal value of an item is discounted by its concept overlap with
        what is already picked, so three tasks probe three parts of the frontier
        instead of the same question three times. Greedy is within 1 - 1/e of
        optimal for monotone submodular.
        """
        scored = self.score(items, ability, relevance)
        chosen: List[ScoredItem] = []
        covered: Set[str] = set()
        lam = self.config.diversity_lambda
        remaining = list(scored)
        while remaining and len(chosen) < k:
            best: Optional[ScoredItem] = None
            best_gain = -math.inf
            for candidate in remaining:
                concepts = set(candidate.item.concepts)
                overlap = (
                    len(concepts & covered) / len(concepts) if concepts else 0.0
                )
                gain = candidate.utility * (1.0 - lam * overlap)
                if gain > best_gain:
                    best, best_gain = candidate, gain
            if best is None or best_gain <= 0.0:
                break
            chosen.append(best)
            covered |= set(best.item.concepts)
            remaining.remove(best)
        return chosen


def items_from_concepts(
    concept_difficulties: Mapping[str, float],
    concepts: Iterable[str],
    *,
    discrimination: float = 1.2,
) -> List[Item]:
    """Treat each concept as a one concept probe item (for frontier drills)."""
    return [
        Item(
            iid=f"concept::{cid}",
            difficulty=concept_difficulties.get(cid, 0.0),
            discrimination=discrimination,
            concepts=(cid,),
            title=cid,
        )
        for cid in concepts
        if cid in concept_difficulties
    ]
