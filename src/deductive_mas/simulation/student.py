"""Simulated learner for validating the evaluation pipeline.

READ THIS before interpreting any number from here. These are simulated
students. The response model encodes a pedagogical assumption from the
literature (on target guidance inside the ZPD gives more durable learning
than being handed the answer) and any experiment run on it will reproduce
that assumption. Nothing here is evidence that Socratic tutoring works.

What it does establish:
- the pipeline, the stats and the reporting are exercised end to end on data
  with known ground truth, so bugs show up here first
- the effect size is driven by things the system really computes (right
  misconception? right concept? reachable?), so the experiment measures the
  system's targeting quality through an explicit response model
- the design can be power analysed before recruiting anyone

Every result carries this caveat in the output.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..domain import Submission
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.mastery import MasteryState, MasteryTracker
from ..knowledge.misconceptions import MISCONCEPTIONS
from ..knowledge.zpd import Item, probability_correct


@dataclass
class ResponseModel:
    """The explicit assumptions that turn guidance into learning.
    Every coefficient is a simulation parameter, not a finding. Values are
    consistent with the scaffolding literature: correct identification matters
    most, aiming inside the ZPD next, a handed solution gives a real but
    shallow and badly retained gain.
    """

    # learning from a Socratic turn that identified the belief correctly
    socratic_on_target: float = 0.62
    # ...that addressed a concept the learner does hold but did not diagnose
    socratic_off_target: float = 0.18
    # multiplier when the target is inside the ZPD
    zpd_bonus: float = 1.35
    # multiplier when the target is way too hard (frustration) or too easy (boredom)
    zpd_penalty: float = 0.55
    # learning from being shown a worked solution
    direct_answer: float = 0.34
    # how much of a direct answer survives to a transfer problem
    direct_transfer_retention: float = 0.35
    # how much of a Socratic gain survives to a transfer problem
    socratic_transfer_retention: float = 0.86
    # prob. a misconception is dropped after correctly targeted guidance
    repair_on_target: float = 0.58
    # ...after a direct answer, which fixes the symptom and leaves the belief
    repair_direct: float = 0.16
    # baseline forgetting between session and post test
    decay: float = 0.05


@dataclass
class StudentProfile:
    """One simulated learner."""

    sid: str
    ability: float
    learning_rate: float
    misconceptions: Set[str] = field(default_factory=set)
    mastery: Optional[MasteryState] = None
    history: List[str] = field(default_factory=list)

    def holds(self, misconception_id: str) -> bool:
        return misconception_id in self.misconceptions


class StudentPopulation:
    """Samples reproducible cohorts of simulated learners."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        tracker: MasteryTracker,
        *,
        model: Optional[ResponseModel] = None,
    ):
        self.graph = graph
        self.tracker = tracker
        self.model = model or ResponseModel()

    def sample(
        self,
        n: int,
        *,
        seed: int,
        misconception_pool: Sequence[str],
        misconceptions_per_student: Tuple[int, int] = (1, 2),
    ) -> List[StudentProfile]:
        """Draw n learners with abilities from a standard normal."""
        rng = random.Random(seed)
        pool = [m for m in misconception_pool if m in MISCONCEPTIONS]
        if not pool:
            raise ValueError("the misconception pool is empty")
        students: List[StudentProfile] = []
        low, high = misconceptions_per_student
        for index in range(n):
            ability = rng.gauss(0.0, 1.0)
            count = rng.randint(low, high)
            held = set(rng.sample(pool, min(count, len(pool))))
            profile = StudentProfile(
                sid=f"S{index:04d}",
                ability=ability,
                learning_rate=min(0.95, max(0.15, rng.gauss(0.55, 0.18))),
                misconceptions=held,
            )
            profile.mastery = self._initial_mastery(profile)
            students.append(profile)
        return students

    def _initial_mastery(self, profile: StudentProfile) -> MasteryState:
        """Ability appropriate priors, lowered on the concepts they misconceive."""
        state = self.tracker.initial(ability=profile.ability)
        for misconception_id in profile.misconceptions:
            entry = MISCONCEPTIONS.get(misconception_id)
            if entry is None:
                continue
            for concept in entry.concepts:
                if concept in self.graph:
                    state.posterior[concept] = min(state.posterior.get(concept, 0.3), 0.18)
        return self.tracker.regularise(state)

    # interaction
    def implicated_concepts(self, misconception_id: str) -> Set[str]:
        """Concepts a misconception implicates plus their prerequisites."""
        entry = MISCONCEPTIONS.get(misconception_id)
        if entry is None:
            return set()
        concepts: Set[str] = set()
        for concept in entry.concepts:
            if concept in self.graph:
                concepts.add(concept)
                concepts |= self.graph.ancestors(concept)
        return concepts

    def apply_intervention(
        self,
        profile: StudentProfile,
        *,
        misconception_id: str,
        diagnosed: bool,
        target_concept: Optional[str],
        zpd_probability: float,
        socratic: bool,
        rng: random.Random,
    ) -> Dict[str, float]:
        """Update the learner after one tutoring turn.
        Returns the per turn quantities the experiment records.
        """
        model = self.model
        implicated = self.implicated_concepts(misconception_id)
        on_target = bool(target_concept and target_concept in implicated)

        if socratic:
            base = model.socratic_on_target if (diagnosed and on_target) else model.socratic_off_target
            if 0.45 <= zpd_probability <= 0.85:
                base *= model.zpd_bonus
            elif zpd_probability < 0.25 or zpd_probability > 0.95:
                base *= model.zpd_penalty
            repair_probability = model.repair_on_target if (diagnosed and on_target) else 0.12
            transfer = model.socratic_transfer_retention
        else:
            base = model.direct_answer
            repair_probability = model.repair_direct
            transfer = model.direct_transfer_retention

        gain = min(0.95, base * profile.learning_rate)
        entry = MISCONCEPTIONS.get(misconception_id)
        touched = list(entry.concepts) if entry else []
        if target_concept and target_concept not in touched:
            touched.append(target_concept)

        assert profile.mastery is not None
        for concept in touched:
            if concept not in self.graph:
                continue
            current = profile.mastery.posterior.get(concept, 0.2)
            profile.mastery.posterior[concept] = min(0.99, current + gain * (1.0 - current))
        profile.mastery = self.tracker.regularise(profile.mastery)

        repaired = rng.random() < repair_probability
        if repaired:
            profile.misconceptions.discard(misconception_id)
        profile.history.append(
            f"{'socratic' if socratic else 'direct'}:{misconception_id}:"
            f"{'on' if on_target else 'off'}-target:{'repaired' if repaired else 'retained'}"
        )
        return {
            "gain": gain,
            "on_target": 1.0 if on_target else 0.0,
            "diagnosed": 1.0 if diagnosed else 0.0,
            "in_zpd": 1.0 if 0.45 <= zpd_probability <= 0.85 else 0.0,
            "repaired": 1.0 if repaired else 0.0,
            "transfer_retention": transfer,
        }

    # post test
    def post_test(
        self,
        profile: StudentProfile,
        items: Sequence[Item],
        *,
        transfer_retention: float,
        rng: random.Random,
    ) -> Tuple[float, float]:
        """Transfer test, returns (score, repeat_error_rate).
        Two things decide each item: is mastery of the item's concepts enough (IRT
        response), and does a retained misconception fire on an item touching it.
        Separating them lets the experiment report retention and recurring error
        rate as different outcomes.
        """
        assert profile.mastery is not None
        if not items:
            return 0.0, 0.0
        correct = 0
        repeats = 0
        exposures = 0
        model = self.model
        for item in items:
            concepts = [c for c in item.concepts if c in self.graph]
            if concepts:
                mastery = sum(profile.mastery.posterior.get(c, 0.2) for c in concepts) / len(concepts)
            else:
                mastery = 0.5
            # only the part of a gain that transfers is available on a new problem
            effective = mastery * transfer_retention + (1.0 - transfer_retention) * 0.5 * mastery
            effective *= 1.0 - model.decay
            ability = _logit(min(0.99, max(0.01, effective)))
            probability = probability_correct(ability + profile.ability * 0.25, item)

            fires = False
            for misconception_id in profile.misconceptions:
                if self.implicated_concepts(misconception_id) & set(concepts):
                    exposures += 1
                    if rng.random() < 0.72:
                        fires = True
                    break
            if fires:
                repeats += 1
                continue
            if rng.random() < probability:
                correct += 1
        score = correct / len(items)
        repeat_rate = repeats / exposures if exposures else 0.0
        return score, repeat_rate


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))
