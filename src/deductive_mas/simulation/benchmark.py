"""RQ1: diagnostic accuracy against the gold labels.

Scores the evaluator the way a second annotator would be scored: multi
label precision / recall / F1, top-1 accuracy, chance corrected agreement.
Also compares with a plain test runner, which says THAT a submission is
wrong but never why.

Caveat: the taxonomy, the rules and the dev bank were developed together, so
numbers on the dev bank show the pipeline is sound, not that it
generalises. That is what the held-out bank is for (see
docs/EXPERIMENT_PROTOCOL.md).
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..agents.base import Agent
from ..agents.orchestrator import DeductiveOrchestrator
from ..config import Config
from ..data.problems import problem as get_problem
from ..data.submissions import all_submissions
from ..domain import Submission
from ..stats.agreement import ClassificationReport, multilabel_report

CAVEAT = (
    "The taxonomy, the detection rules and this labelled bank were developed together. "
    "These figures establish internal validity (the pipeline recovers the beliefs it is "
    "designed to name) and must not be read as generalisation to unseen student code."
)


@dataclass
class BenchmarkRow:
    submission_id: str
    problem_id: str
    gold: Tuple[str, ...]
    predicted: Tuple[str, ...]
    top1: Optional[str]
    confidence: float
    located: bool
    seconds: float
    backend: str = "offline"
    degraded: bool = False

    @property
    def correct(self) -> bool:
        return bool(self.top1 and self.top1 in self.gold)

    @property
    def recalled(self) -> bool:
        return bool(set(self.gold) & set(self.predicted))


@dataclass
class BenchmarkResult:
    rows: List[BenchmarkRow] = field(default_factory=list)
    report: ClassificationReport = field(default_factory=ClassificationReport)
    false_positive_rate: float = 0.0
    clean_submissions: int = 0
    localisation_rate: float = 0.0
    mean_seconds: float = 0.0
    baseline_top1: float = 0.0
    caveat: str = CAVEAT
    backend: str = "offline"
    # rows the live backend could not answer (offline reasoner stood in)
    degraded_rows: int = 0
    # clean submissions where something was flagged, by id
    clean_flagged_ids: Tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True if every row was answered by the configured backend."""
        return self.degraded_rows == 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "caveat": self.caveat,
            "backend": self.backend,
            "degraded_rows": self.degraded_rows,
            "n_labelled": self.report.support,
            "n_clean": self.clean_submissions,
            "top1_accuracy": round(self.report.top1_accuracy, 4),
            "recall_any": round(
                sum(1 for r in self.rows if r.recalled) / len(self.rows), 4
            ) if self.rows else 0.0,
            "micro_precision": round(self.report.micro_precision, 4),
            "micro_recall": round(self.report.micro_recall, 4),
            "micro_f1": round(self.report.micro_f1, 4),
            "macro_f1": round(self.report.macro_f1, 4),
            "cohen_kappa": round(self.report.kappa, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "clean_flagged": list(self.clean_flagged_ids),
            "divergence_localisation_rate": round(self.localisation_rate, 4),
            "mean_seconds_per_submission": round(self.mean_seconds, 4),
            "baseline_top1_accuracy": round(self.baseline_top1, 4),
            "per_label": {
                label: {
                    "precision": round(p, 3), "recall": round(r, 3),
                    "f1": round(f, 3), "support": s,
                }
                for label, (p, r, f, s) in sorted(self.report.per_label.items())
                if s
            },
            "rows": [
                {
                    "submission": r.submission_id,
                    "problem": r.problem_id,
                    "gold": list(r.gold),
                    "predicted": list(r.predicted),
                    "top1": r.top1,
                    "confidence": round(r.confidence, 4),
                    "correct": r.correct,
                    "located": r.located,
                    "backend": r.backend,
                    "degraded": r.degraded,
                }
                for r in self.rows
            ],
        }


class DiagnosticBenchmark:
    """Scores the evaluator on a labelled submission bank."""

    def __init__(
        self,
        config: Optional[Config] = None,
        orchestrator: Optional[DeductiveOrchestrator] = None,
        *,
        evaluator: Optional[Agent] = None,
    ):
        self.config = config or Config()
        self.orchestrator = orchestrator or DeductiveOrchestrator(self.config)
        # an alternative diagnostic agent (e.g. the model only baseline)
        self.evaluator = evaluator

    def run(self, submissions: Optional[Sequence[Submission]] = None) -> BenchmarkResult:
        bank = list(submissions if submissions is not None else all_submissions())
        result = BenchmarkResult(backend=self.orchestrator.reasoner.name)
        degraded = 0
        flagged_clean: List[str] = []

        predictions: List[List[str]] = []
        gold: List[List[str]] = []
        top_choices: List[str] = []
        clean_flagged = 0
        clean_total = 0
        located = 0
        elapsed = 0.0

        for submission in bank:
            spec = get_problem(submission.problem_id)
            started = time.perf_counter()
            diagnosis = self.orchestrator.diagnose(spec, submission, evaluator=self.evaluator)
            duration = time.perf_counter() - started
            elapsed += duration
            if diagnosis.degraded:
                degraded += 1

            predicted = tuple(hit.misconception_id for hit in diagnosis.hits)
            top1 = predicted[0] if predicted else None

            if submission.gold_misconceptions:
                row = BenchmarkRow(
                    submission_id=submission.sid,
                    problem_id=submission.problem_id,
                    gold=submission.gold_misconceptions,
                    predicted=predicted,
                    top1=top1,
                    confidence=diagnosis.confidence,
                    located=diagnosis.divergence is not None,
                    seconds=duration,
                    backend=diagnosis.backend,
                    degraded=diagnosis.degraded,
                )
                result.rows.append(row)
                predictions.append(list(predicted))
                gold.append(list(submission.gold_misconceptions))
                top_choices.append(top1 or "")
                if row.located:
                    located += 1
            else:
                clean_total += 1
                if predicted:
                    clean_flagged += 1
                    flagged_clean.append(submission.sid)

        result.report = multilabel_report(predictions, gold, top_choice=top_choices)
        result.clean_submissions = clean_total
        result.false_positive_rate = clean_flagged / clean_total if clean_total else 0.0
        result.localisation_rate = located / len(result.rows) if result.rows else 0.0
        result.mean_seconds = elapsed / len(bank) if bank else 0.0
        # a test runner detects failure but names no misconception, so its top-1
        # accuracy is 0 by construction. Stating it makes clear what the
        # diagnostic layer buys.
        result.baseline_top1 = 0.0
        result.degraded_rows = degraded
        result.clean_flagged_ids = tuple(flagged_clean)
        return result
