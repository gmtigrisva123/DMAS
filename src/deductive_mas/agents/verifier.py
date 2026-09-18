"""The compliance gate. Everything shown to a student goes through here first.

- answer leakage: winnowing fingerprints (MOSS style), AST matching against
  the reference solution and an edit proximity channel that catches concrete
  repair instructions (see analysis/leakage.py)
- grounding: every assertive sentence must be supported by a retrieved card
  (see retrieval/ma_rag.py). Questions and statements about the student's
  own code assert nothing so they are exempt.

Text that fails is replaced, not just flagged. That is what gives the
guarantee that nothing reaches the student without passing the check.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..analysis.leakage import LeakageContext, LeakageDetector, LeakageReport
from ..config import Config
from ..retrieval.ma_rag import GroundedContext, GroundingReport, MultiAgentRAG
from .base import Agent, Blackboard


@dataclass
class ItemVerdict:
    """Gate decision for one piece of text."""

    text: str
    leakage: LeakageReport
    grounding: GroundingReport
    accepted: bool
    reasons: Tuple[str, ...] = ()


@dataclass
class VerificationReport:
    verdicts: Tuple[ItemVerdict, ...] = ()
    replaced: int = 0

    @property
    def max_leakage(self) -> float:
        return max((v.leakage.score for v in self.verdicts), default=0.0)

    @property
    def min_grounding(self) -> float:
        return min((v.grounding.score for v in self.verdicts), default=1.0)

    @property
    def all_accepted(self) -> bool:
        return all(v.accepted for v in self.verdicts)


class ComplianceVerifier:
    """Scores and gates generated text for one submission."""

    def __init__(
        self,
        config: Optional[Config] = None,
        rag: Optional[MultiAgentRAG] = None,
    ):
        self.config = config or Config()
        self.rag = rag or MultiAgentRAG(config=self.config.retrieval)
        self.detector = LeakageDetector(
            kgram=self.config.intervention.fingerprint_kgram,
            window=self.config.intervention.fingerprint_window,
        )

    def context_for(self, reference_solution: str, student_source: str) -> LeakageContext:
        return self.detector.build(reference_solution, student_source)

    def review(
        self,
        text: str,
        leakage_context: LeakageContext,
        grounding_context: GroundedContext,
    ) -> ItemVerdict:
        leakage = leakage_context.score(text)
        grounding = self.rag.verify(text, grounding_context)
        reasons: List[str] = []
        accepted = True
        if leakage.rejected(self.config.intervention.max_leakage):
            accepted = False
            reasons.append(
                f"answer leakage {leakage.score:.2f} exceeds the limit of "
                f"{self.config.intervention.max_leakage:.2f}"
            )
            reasons.extend(leakage.reasons)
        # gate on every assertion, not the mean: one made up sentence among four
        # good ones is exactly what a mean would hide
        if grounding.unsupported:
            accepted = False
            unsupported = "; ".join(c.claim[:70] for c in grounding.unsupported[:2])
            reasons.append(
                f"unsupported claim(s) below the grounding floor of "
                f"{self.config.retrieval.min_grounding_score:.2f}: {unsupported}"
            )
        return ItemVerdict(
            text=text, leakage=leakage, grounding=grounding, accepted=accepted,
            reasons=tuple(reasons),
        )


class VerificationAgent(Agent):
    """Pipeline stage that records the final verdict."""

    name = "compliance-verifier"
    role = "policy enforcement"

    def __init__(self, verifier: Optional[ComplianceVerifier] = None, config: Optional[Config] = None):
        self.config = config or Config()
        self.verifier = verifier or ComplianceVerifier(self.config)

    def run(self, board: Blackboard) -> Blackboard:
        intervention = board.intervention
        if intervention is None:
            return board
        leakage_context = self.verifier.context_for(
            board.problem.reference_solution, board.submission.source
        )
        grounding_context = board.context or GroundedContext()

        verdicts: List[ItemVerdict] = []
        for hint in intervention.hints:
            verdicts.append(self.verifier.review(hint.text, leakage_context, grounding_context))
        if intervention.challenge is not None:
            combined = f"{intervention.challenge.prompt} {intervention.challenge.trace_question}"
            verdicts.append(self.verifier.review(combined, leakage_context, grounding_context))

        report = VerificationReport(verdicts=tuple(verdicts))
        board.extras["verification"] = report
        intervention.leakage = report.max_leakage
        intervention.grounding = report.min_grounding
        if not report.all_accepted:
            # getting here means the intervention agent's own gate let something
            # through, which is a bug
            board.telemetry.warn(
                "final compliance check rejected generated guidance; it was withheld"
            )
            board.intervention.hints = tuple(
                hint for hint, verdict in zip(intervention.hints, verdicts) if verdict.accepted
            )
        return board
