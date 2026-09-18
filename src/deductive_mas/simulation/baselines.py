"""The tutors compared in the RQ2 experiment. Three is the minimum to attribute
the effect to the right cause:

- DirectAnswerTutor: control, explains the correct solution. No diagnosis,
  no target.
- DeductiveTutor: the full system.
- UntargetedSocraticTutor: keeps the Socratic form but drops the knowledge
  graph alignment, picks a target concept at random from the problem.
  Without this arm any effect could just be "asking questions helps" and
  the graph would never be evaluated.
"""

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from ..agents.orchestrator import DeductiveOrchestrator
from ..domain import ProblemSpec, Submission


@dataclass
class TurnOutcome:
    """What a tutor produced for one submission, in comparable terms."""

    arm: str
    diagnosed: bool
    target_concept: Optional[str]
    zpd_probability: float
    socratic: bool
    leakage: float
    grounding: float
    hints: int = 0


class Tutor:
    """Interface an experimental arm must satisfy."""

    arm = "tutor"
    socratic = True

    def teach(
        self, problem: ProblemSpec, submission: Submission, *, rng: random.Random
    ) -> TurnOutcome:
        raise NotImplementedError


class DirectAnswerTutor(Tutor):
    """Control: explain the solution, diagnose nothing."""

    arm = "control-direct"
    socratic = False

    def teach(self, problem: ProblemSpec, submission: Submission, *, rng: random.Random) -> TurnOutcome:
        return TurnOutcome(
            arm=self.arm,
            diagnosed=False,
            target_concept=None,
            zpd_probability=0.0,
            socratic=False,
            leakage=1.0,          # it IS the answer
            grounding=1.0,
            hints=0,
        )


class _MemoisingTutor(Tutor):
    """Shared plumbing: the pipeline is deterministic so cache per submission.
    A cohort of 200 draws from a bank of ~30 submissions, without this the
    experiment would run the same diagnosis hundreds of times.
    """

    # shared between tutors wrapping the same orchestrator so the design
    # analysis (hundreds of runs) diagnoses each submission once
    _shared_cache: dict = {}

    def __init__(self, orchestrator: DeductiveOrchestrator):
        self.orchestrator = orchestrator
        self._cache = self._shared_cache.setdefault(id(orchestrator), {})

    def _result(self, problem: ProblemSpec, submission: Submission):
        key = (problem.pid, submission.sid)
        if key not in self._cache:
            self._cache[key] = self.orchestrator.tutor(problem, submission)
        return self._cache[key]


class DeductiveTutor(_MemoisingTutor):
    """The full deductive system."""

    arm = "treatment-mas"
    socratic = True

    def teach(self, problem: ProblemSpec, submission: Submission, *, rng: random.Random) -> TurnOutcome:
        result = self._result(problem, submission)
        primary = result.diagnosis.primary
        diagnosed = bool(
            primary and primary.misconception_id in set(submission.gold_misconceptions)
        )
        return TurnOutcome(
            arm=self.arm,
            diagnosed=diagnosed,
            target_concept=result.intervention.target_concept,
            zpd_probability=result.intervention.zpd_probability,
            socratic=True,
            leakage=result.intervention.leakage,
            grounding=result.intervention.grounding,
            hints=len(result.intervention.hints),
        )


class UntargetedSocraticTutor(_MemoisingTutor):
    """Ablation: Socratic questions without the knowledge graph."""

    arm = "ablation-no-kg"
    socratic = True

    def teach(self, problem: ProblemSpec, submission: Submission, *, rng: random.Random) -> TurnOutcome:
        result = self._result(problem, submission)
        primary = result.diagnosis.primary
        diagnosed = bool(
            primary and primary.misconception_id in set(submission.gold_misconceptions)
        )
        # keep the diagnosis, throw away the alignment: aim at a random concept of
        # the problem instead of the located deficiency
        concepts = list(problem.concepts)
        target = rng.choice(concepts) if concepts else None
        return TurnOutcome(
            arm=self.arm,
            diagnosed=diagnosed,
            target_concept=target,
            zpd_probability=rng.uniform(0.2, 0.95),
            socratic=True,
            leakage=result.intervention.leakage,
            grounding=result.intervention.grounding,
            hints=len(result.intervention.hints),
        )
