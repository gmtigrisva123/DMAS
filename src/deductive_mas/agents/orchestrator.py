"""The orchestrator.

Runs the four stages on a shared blackboard in a fixed order (you cannot
align what you have not diagnosed etc.). Fixed order also keeps a session
reproducible. Any stage may fail: it records a warning, leaves the blackboard
alone and the later stages do what they still can.
"""

from typing import List, Optional, Sequence

from ..config import Config
from ..domain import (
    AlignmentReport,
    Diagnosis,
    Intervention,
    ProblemSpec,
    Submission,
    Telemetry,
    TutoringResult,
)
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.mastery import MasteryState, MasteryTracker
from ..knowledge.ontology import knowledge_graph
from ..llm.registry import Reasoner
from ..retrieval.ma_rag import MultiAgentRAG
from .alignment import KnowledgeGraphAlignmentAgent
from .base import Agent, Blackboard
from .evaluator import SocraticEvaluatorAgent
from .intervention import CognitiveInterventionAgent
from .verifier import ComplianceVerifier, VerificationAgent


class DeductiveOrchestrator:
    """The whole system behind one call."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        reasoner: Optional[Reasoner] = None,
        graph: Optional[KnowledgeGraph] = None,
        rag: Optional[MultiAgentRAG] = None,
    ):
        self.config = config or Config()
        self.graph = graph or knowledge_graph()
        self.reasoner = reasoner or Reasoner(self.config.llm)
        self.rag = rag or MultiAgentRAG(config=self.config.retrieval)
        self.tracker = MasteryTracker(self.graph, self.config.knowledge)
        verifier = ComplianceVerifier(self.config, self.rag)

        self.evaluator = SocraticEvaluatorAgent(self.config, self.reasoner)
        self.alignment = KnowledgeGraphAlignmentAgent(self.config, self.graph, self.tracker)
        self.intervention = CognitiveInterventionAgent(
            self.config, self.reasoner, self.rag, self.graph, verifier
        )
        self.verification = VerificationAgent(verifier, self.config)

    @property
    def stages(self) -> Sequence[Agent]:
        return (self.evaluator, self.alignment, self.intervention, self.verification)

    # public api
    def tutor(
        self,
        problem: ProblemSpec,
        submission: Submission,
        *,
        mastery: Optional[MasteryState] = None,
    ) -> TutoringResult:
        """One full diagnose -> align -> intervene cycle."""
        board = Blackboard(problem=problem, submission=submission)
        board.telemetry.backend = self.reasoner.name
        if mastery is not None:
            board.extras["mastery"] = mastery

        for stage in self.stages:
            try:
                stage(board)
            except Exception as exc:
                board.telemetry.warn(f"{stage.name} failed: {type(exc).__name__}: {exc}")
                board.note(f"{stage.name} did not complete")

        board.telemetry.cache_hits = self.reasoner.cache_hits
        updated = board.extras.get("mastery")
        return TutoringResult(
            submission=submission,
            problem=problem,
            diagnosis=board.diagnosis or Diagnosis(submission.sid, problem.pid, inconclusive=True),
            alignment=board.alignment or AlignmentReport(),
            intervention=board.intervention or Intervention(),
            telemetry=board.telemetry,
            evidence=tuple(board.evidence),
            mastery_posterior=dict(updated.posterior) if updated is not None else {},
            mastery_observations=dict(updated.observations) if updated is not None else {},
            gate_log=list(board.extras.get("gate_log", ())),
        )

    def diagnose(
        self, problem: ProblemSpec, submission: Submission, *, evaluator: Optional[Agent] = None
    ) -> Diagnosis:
        """Only the diagnosis stage (used by the RQ1 benchmark).

        evaluator can be another diagnostic agent (e.g. the model only baseline) so
        the benchmark scores every variant the same way.
        """
        board = Blackboard(problem=problem, submission=submission)
        board.telemetry.backend = self.reasoner.name
        (evaluator or self.evaluator)(board)
        return board.diagnosis or Diagnosis(submission.sid, problem.pid, inconclusive=True)

    def mastery_after(self, result: TutoringResult, base: Optional[MasteryState] = None) -> MasteryState:
        """Mastery state after a finished session."""
        state = base or self.tracker.initial()
        return self.tracker.apply_blame(state, result.alignment.blame)
