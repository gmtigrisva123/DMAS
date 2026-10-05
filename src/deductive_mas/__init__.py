"""DMAS - deductive multi agent system for finding student misconceptions.

Three agents share one blackboard:
- SocraticEvaluatorAgent: runs the student code next to the reference, matches
  variables by behaviour and finds the first place where they differ. Evidence
  from rules / traces / the model is fused with Dempster-Shafer.
- KnowledgeGraphAlignmentAgent: spreads blame over the prerequisite graph
  (personalised PageRank), updates BKT mastery and picks the deficiency frontier.
- CognitiveInterventionAgent: picks a target in the ZPD (IRT), grounds hints in
  the retrieved corpus and checks them for answer leakage before showing them.

Usage:

    from deductive_mas import DeductiveOrchestrator, problem, submission
    result = DeductiveOrchestrator().tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
    print(result.diagnosis.primary.misconception_id)

No third party dependencies, everything is written from scratch in the stdlib.
"""

from .agents.orchestrator import DeductiveOrchestrator
from .config import Config
from .data.problems import all_problems, problem
from .data.submissions import all_submissions, submission
from .domain import (
    AlignmentReport,
    Diagnosis,
    Intervention,
    ProblemSpec,
    Submission,
    TutoringResult,
)
from .knowledge.ontology import knowledge_graph
from .pipeline import AnalysisEngine, AnalysisResult
from .version import CODENAME, __version__
