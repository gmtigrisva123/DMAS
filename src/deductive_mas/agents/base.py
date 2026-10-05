"""Blackboard + base Agent class.

The agents never call each other, they only read/write the shared
Blackboard. That is what makes it possible to swap or ablate one stage without
touching the others.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..domain import (
    AlignmentReport,
    Diagnosis,
    Evidence,
    Intervention,
    ProblemSpec,
    Submission,
    Telemetry,
)


@dataclass
class Blackboard:
    """Shared working memory for one session."""

    problem: ProblemSpec
    submission: Submission
    telemetry: Telemetry = field(default_factory=Telemetry)
    analysis: Any = None                       # AnalysisResult, untyped to avoid an import cycle
    evidence: List[Evidence] = field(default_factory=list)
    diagnosis: Optional[Diagnosis] = None
    alignment: Optional[AlignmentReport] = None
    intervention: Optional[Intervention] = None
    context: Any = None                        # GroundedContext from the retrieval layer
    notes: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def note(self, message: str):
        if message not in self.notes:
            self.notes.append(message)


class Agent:
    """A named stage that updates the blackboard in place."""

    name = "agent"
    role = ""

    def run(self, board: Blackboard) -> Blackboard:
        raise NotImplementedError

    def __call__(self, board: Blackboard) -> Blackboard:
        started = time.perf_counter()
        try:
            return self.run(board)
        finally:
            board.telemetry.record(self.name, time.perf_counter() - started)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
