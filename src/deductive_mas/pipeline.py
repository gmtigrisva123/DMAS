"""The deterministic analysis pipeline, one pass over a submission.

It lives above analysis/ and knowledge/ because it uses both: run the program
analyses, then evaluate the misconception rules on the results. Order:

1. structural analysis: parse, normalise, features, CFG, dataflow
2. cost analysis: complexity of the submission and of the reference
3. differential testing: look for a failing input and shrink it
4. trace alignment: run both programs on that input, match variables by
   behaviour, find the first divergence
5. rule evaluation over all of the above

No model call happens here. The model only gets to interpret this evidence.
"""

import ast
import copy
import random
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config
from .util.determinism import seeded_rng
from .domain import (
    Counterexample,
    DivergenceKind,
    DivergencePoint,
    Evidence,
    EvidenceKind,
    ProblemSpec,
    SourceSpan,
    Submission,
)
from .knowledge.misconceptions import RULES, RuleContext, RuleHit
from .analysis.align import locate_divergence
from .analysis.ast_features import CodeFeatures, extract
from .analysis.cfg import ControlFlowGraph, build_cfg, synchronisation_lines
from .analysis.complexity import ComplexityEstimator, ComplexityReport, estimate
from .analysis.counterexample import CounterexampleSearch, SearchOutcome
from .analysis.dataflow import DataflowFacts, analyse as solve_dataflow
from .analysis.normalize import NormalForm, normalise
from .analysis.tracer import ExecutionResult, GuardedRunner
from .analysis.varmatch import RoleMapping, match_variables


@dataclass
class AnalysisResult:
    """Everything the deterministic layers found out about one submission."""

    submission: Submission
    problem: ProblemSpec
    features: CodeFeatures
    complexity: ComplexityReport
    reference_complexity: ComplexityReport
    entry_point: str
    cfg: Optional[ControlFlowGraph] = None
    dataflow: Optional[DataflowFacts] = None
    normal_form: Optional[NormalForm] = None
    student_execution: Optional[ExecutionResult] = None
    reference_execution: Optional[ExecutionResult] = None
    role_mapping: Optional[RoleMapping] = None
    divergence: Optional[DivergencePoint] = None
    search: SearchOutcome = field(default_factory=SearchOutcome)
    evidence: Tuple[Evidence, ...] = ()
    notes: List[str] = field(default_factory=list)

    @property
    def parsed(self) -> bool:
        return self.features.parsed

    @property
    def counterexample(self) -> Optional[Counterexample]:
        return self.search.counterexample

    @property
    def complexity_gap(self) -> bool:
        return self.complexity.total.dominates(self.reference_complexity.total)

    @property
    def passes_all_tests(self) -> bool:
        return self.search.counterexample is None and self.search.attempts > 0

    def evidence_for(self, misconception_id: str) -> Tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if e.misconception_id == misconception_id)


class AnalysisEngine:
    """Runs the pipeline for one (problem, submission) pair."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.runner = GuardedRunner(self.config.execution)
        self.searcher = CounterexampleSearch(
            GuardedRunner(
                replace(
                    self.config.execution,
                    max_steps=self.config.execution.search_max_steps,
                    max_seconds=self.config.execution.search_max_seconds,
                )
            )
        )

    # public api
    def analyse(self, problem: ProblemSpec, submission: Submission) -> AnalysisResult:
        entry = resolve_entry(submission.source, problem.entry_point)
        features = extract(submission.source, entry)
        reference_complexity = estimate(problem.reference_solution, problem.entry_point)

        result = AnalysisResult(
            submission=submission,
            problem=problem,
            features=features,
            complexity=ComplexityReport(function=entry),
            reference_complexity=reference_complexity,
            entry_point=entry,
        )

        if not features.parsed:
            # syntax error = the diagnosis, nothing later would add anything
            line = features.parse_error_line
            missing_entry = "not defined" in (features.parse_error or "")
            result.evidence = (
                Evidence(
                    misconception_id="var.uninitialised" if missing_entry else "run.type-error",
                    kind=EvidenceKind.STATIC,
                    belief=0.0,
                    rationale=f"the submission does not parse: {features.parse_error}",
                    span=features.span(line),
                ),
            )
            result.notes.append(f"parse failure: {features.parse_error}")
            return result

        result.normal_form = normalise(submission.source)
        assert features.function is not None
        result.cfg = build_cfg(features.function)
        result.dataflow = solve_dataflow(
            result.cfg,
            tuple(features.parameters) + _module_level_names(features.tree),
        )
        result.complexity = _safe_complexity(submission.source, entry)

        if problem.reference_solution.strip():
            result.search = self._search(problem, submission, entry)
            self._trace(problem, submission, entry, result)
        else:
            # no reference for this file, the structural rules / dataflow / cost
            # analysis still work but everything comparative (diff testing, trace
            # alignment, complexity gap) is skipped instead of run against nothing
            result.notes.append(
                "no reference solution for this task: reporting structural findings only "
                "(no counterexample search, no trace alignment, no complexity comparison)"
            )
        result.evidence = self._apply_rules(problem, result)
        return result

    # internals
    def _search(self, problem: ProblemSpec, submission: Submission, entry: str) -> SearchOutcome:
        rng = seeded_rng(self.config.seed, submission.sid)
        try:
            return self.searcher.search(
                problem,
                submission.source,
                student_entry=entry,
                rng=rng,
                random_attempts=self.config.execution.search_random_attempts,
            )
        except Exception as exc:
            outcome = SearchOutcome()
            outcome.source = f"search-error: {type(exc).__name__}"
            return outcome

    def _trace(
        self,
        problem: ProblemSpec,
        submission: Submission,
        entry: str,
        result: AnalysisResult,
    ):
        args = self._trace_input(problem, result)
        if args is None:
            result.notes.append("no input available for tracing")
            return

        # deep copy per run, a submission that mutates its argument (sort in place,
        # pop from the list) must not change the input for the reference or the
        # next submission
        student = self.runner.run(submission.source, entry, copy.deepcopy(args))
        reference = self.runner.run(
            problem.reference_solution, problem.entry_point, copy.deepcopy(args)
        )
        result.student_execution = student
        result.reference_execution = reference

        if not reference.ok and not reference.events:
            result.notes.append("the reference solution could not be traced on this input")
            return

        student_params = list(result.features.parameters)
        reference_params = list(problem.parameters)
        mapping = match_variables(
            student,
            reference,
            student_params=student_params,
            reference_params=reference_params,
        )
        mapping = self._trusted_roles(mapping, student_params, reference_params, result)
        result.role_mapping = mapping
        result.divergence = locate_divergence(
            student,
            reference,
            mapping,
            source_lines=submission.lines,
            student_sync=_sync_lines(submission.source, entry),
            reference_sync=_sync_lines(problem.reference_solution, problem.entry_point),
        )

    def _trusted_roles(
        self,
        mapping: RoleMapping,
        student_params: Sequence[str],
        reference_params: Sequence[str],
        result: AnalysisResult,
    ) -> RoleMapping:
        """Drop variable matches that are too weak to quote to the student.

        The Hungarian algorithm always returns some assignment, even when the
        student used a totally different algorithm. Saying "your used holds 0 but
        should be 1" based on a 0.66 match would be wrong. So weak roles are dropped,
        and if nothing survives we fall back to comparing outputs, which is always
        sound.
        """
        threshold = self.config.execution.min_role_confidence
        kept = {
            student: reference
            for student, reference in mapping.pairs.items()
            if student in mapping.forced or mapping.scores.get(student, 0.0) >= threshold
        }
        dropped = sorted(set(mapping.pairs) - set(kept))
        if not dropped:
            return mapping

        detail = ", ".join(
            "{}~{} ({:.2f})".format(name, mapping.pairs[name], mapping.scores.get(name, 0.0))
            for name in dropped[:3]
        )
        free_kept = [name for name in kept if name not in mapping.forced]
        if free_kept:
            result.notes.append(
                "variable role(s) {} matched too weakly to quote; excluded from the "
                "comparison".format(detail)
            )
        else:
            result.notes.append(
                "no variable role matched confidently ({}); the submission appears to use a "
                "different algorithm, so divergence is reported at the output level".format(detail)
            )
        return RoleMapping(
            pairs=kept,
            scores={name: mapping.scores.get(name, 1.0) for name in kept},
            forced=mapping.forced,
            unmatched_student=mapping.unmatched_student + tuple(dropped),
            unmatched_reference=mapping.unmatched_reference,
        )

    def _trace_input(self, problem: ProblemSpec, result: AnalysisResult) -> Optional[Sequence[Any]]:
        """Use the shrunk counterexample if we have one, otherwise a declared test."""
        if result.search.counterexample is not None:
            return list(result.search.counterexample.args)
        if problem.tests:
            return list(problem.tests[0].args)
        return None

    def _apply_rules(self, problem: ProblemSpec, result: AnalysisResult) -> Tuple[Evidence, ...]:
        context = RuleContext(
            problem=problem,
            has_reference=bool(problem.reference_solution.strip()),
            features=result.features,
            complexity=result.complexity,
            reference_complexity=result.reference_complexity,
            cfg=result.cfg,
            dataflow=result.dataflow,
            execution=result.student_execution,
            divergence=result.divergence,
            counterexample=result.counterexample,
        )
        evidence: List[Evidence] = []
        for rule in RULES:
            try:
                hit = rule(context)
            except Exception as exc:
                result.notes.append(
                    f"rule {rule.misconception_id} raised {type(exc).__name__}: {exc}"
                )
                continue
            if hit is None:
                continue
            evidence.append(
                Evidence(
                    misconception_id=rule.misconception_id,
                    kind=rule.kind,
                    belief=max(0.0, min(1.0, hit.belief)),
                    rationale=hit.rationale,
                    span=result.features.span(hit.line),
                    payload={"rule": rule.misconception_id},
                )
            )

        divergence = result.divergence
        if divergence is not None and divergence.student_line:
            # dynamic evidence backs up whichever symbolic hypothesis sits at the line
            # where the trace diverged. static + dynamic agreeing on the same line is a
            # lot stronger than either alone, Dempster fusion turns that into belief
            for item in list(evidence):
                if item.span is None or item.kind is not EvidenceKind.STATIC:
                    continue
                if abs(item.span.line - divergence.student_line) <= 1:
                    evidence.append(
                        Evidence(
                            misconception_id=item.misconception_id,
                            kind=EvidenceKind.DYNAMIC,
                            belief=min(0.9, 0.45 + 0.35 * item.belief),
                            rationale=(
                                f"execution first diverges at line {divergence.student_line}, the "
                                f"same line this rule flags ({divergence.kind})"
                            ),
                            span=item.span,
                            payload={"corroborates": item.misconception_id},
                        )
                    )
        return tuple(evidence)


# helpers
def resolve_entry(source: str, expected: str) -> str:
    """Function to call: the expected name, else the first one defined."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return expected
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if expected in names:
        return expected
    top_level = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    return top_level[0] if top_level else (names[0] if names else expected)


def _module_level_names(tree) -> Tuple[str, ...]:
    """Names bound at module level (functions, classes, imports, globals).
    Without these a recursive call would look like use before definition.
    """
    if tree is None:
        return ()
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.extend(alias.asname or alias.name.split(".")[0] for alias in node.names)
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
    return tuple(dict.fromkeys(names))


def _sync_lines(source: str, entry: str) -> Tuple[int, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    name = resolve_entry(source, entry)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return tuple(sorted(synchronisation_lines(node, build_cfg(node))))
    return ()


def _safe_complexity(source: str, entry: str) -> ComplexityReport:
    try:
        return estimate(source, entry)
    except (RecursionError, ValueError, TypeError) as exc:
        return ComplexityReport(function=entry, notes=[f"complexity inference failed: {exc}"])
