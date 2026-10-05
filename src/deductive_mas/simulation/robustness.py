"""RQ3: robustness to coding style and naming.

Mutation study: each labelled submission is rewritten by semantics
preserving AST transforms (obfuscated identifiers, split assignments,
inverted conditionals, dead code, expanded augmented assignments) and the
diagnosis is recomputed. A system reasoning about behaviour should not
move, one that pattern matches on surface form should degrade.

Every mutant is checked to behave identically to the original before it is
scored, otherwise a transform that changed the meaning would make an honest
diagnostic change look like a failure.
"""

import ast
import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..agents.orchestrator import DeductiveOrchestrator
from ..analysis.tracer import GuardedRunner
from ..config import Config
from ..data.problems import problem as get_problem
from ..data.submissions import labelled
from ..domain import ProblemSpec, Submission
from ..util.text import jaccard


# semantics preserving transforms
_PROTECTED = frozenset(
    """print len range enumerate sorted sum min max abs int float str list dict set tuple
    reversed zip map filter any all bool divmod pow round isinstance next iter deque
    defaultdict Counter heappush heappop bisect_left bisect_right math collections""".split()
)


class _Obfuscate(ast.NodeTransformer):
    """Rename every user identifier to an unhelpful but legal name."""

    def __init__(self, rng: random.Random, keep: Sequence[str] = ()):
        self.rng = rng
        self.keep = set(keep)
        self.mapping: Dict[str, str] = {}
        self._alphabet = ["qq", "zz", "ll", "oO", "xx1", "yy2", "aA", "bB", "cC", "dD", "eE", "fF"]

    def _rename(self, name: str) -> str:
        if name in _PROTECTED or name in self.keep or name.startswith("__"):
            return name
        if name not in self.mapping:
            index = len(self.mapping)
            base = self._alphabet[index % len(self._alphabet)]
            suffix = "" if index < len(self._alphabet) else str(index)
            self.mapping[name] = base + suffix
        return self.mapping[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._rename(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg=self._rename(node.arg), annotation=None), node)


class _SplitAssignments(ast.NodeTransformer):
    """a, b = x, y becomes two statements."""

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        target, value = node.targets[0], node.value
        if not (isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)):
            return node
        if len(target.elts) != len(value.elts):
            return node
        # only safe when no target name appears on the right hand side, otherwise
        # the simultaneous semantics of tuple assignment is lost
        targets = {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        reads = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
        if targets & reads:
            return node
        out: List[ast.stmt] = []
        for element, item in zip(target.elts, value.elts):
            statement = ast.Assign(targets=[element], value=item)
            ast.copy_location(statement, node)
            out.append(statement)
        return out


class _InvertConditionals(ast.NodeTransformer):
    """if C: A else: B becomes if not C: B else: A."""

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if not node.orelse:
            return node
        negated = ast.UnaryOp(op=ast.Not(), operand=node.test)
        ast.copy_location(negated, node.test)
        node.test, node.body, node.orelse = negated, node.orelse, node.body
        return node


class _ExpandAugmented(ast.NodeTransformer):
    """x += 1 becomes x = x + 1."""

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            return node
        load = ast.Name(id=node.target.id, ctx=ast.Load())
        ast.copy_location(load, node.target)
        binop = ast.BinOp(left=load, op=node.op, right=node.value)
        ast.copy_location(binop, node)
        statement = ast.Assign(targets=[node.target], value=binop)
        return ast.copy_location(statement, node)


class _InjectNoise(ast.NodeTransformer):
    """Add a docstring and a dead assignment to every function."""

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        docstring = ast.Expr(value=ast.Constant(value="Solution attempt. TODO: check edge cases."))
        dead = ast.Assign(
            targets=[ast.Name(id="_scratch_note", ctx=ast.Store())],
            value=ast.Constant(value=0),
        )
        for extra in (docstring, dead):
            ast.copy_location(extra, node)
        node.body = [docstring, dead] + list(node.body)
        return node


def _defined_functions(tree: ast.AST) -> List[str]:
    """Function names must survive renaming or recursive calls break."""
    return [
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


TRANSFORMS: Dict[str, Callable[[random.Random], ast.NodeTransformer]] = {
    "rename-identifiers": lambda rng: _Obfuscate(rng),
    "split-assignments": lambda rng: _SplitAssignments(),
    "invert-conditionals": lambda rng: _InvertConditionals(),
    "expand-augmented": lambda rng: _ExpandAugmented(),
    "inject-noise": lambda rng: _InjectNoise(),
}

# hardest condition: all of the above at once
COMPOSITE = "all-combined"

# findings a transform CREATES. Injecting a dead assignment makes the mutant
# really contain a dead store, so reporting it is a correct new observation
# and not a changed diagnosis. Discounted before comparing the finding sets.
INTRODUCED: Dict[str, frozenset] = {
    "inject-noise": frozenset({"var.dead-store"}),
    COMPOSITE: frozenset({"var.dead-store"}),
}


def mutate(source: str, transform: str, rng: random.Random) -> Optional[str]:
    """Apply a named transform, None if it cannot be applied."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    functions = _defined_functions(tree)
    names = [transform] if transform != COMPOSITE else list(TRANSFORMS)
    for name in names:
        factory = TRANSFORMS.get(name)
        if factory is None:
            return None
        transformer = factory(rng)
        if isinstance(transformer, _Obfuscate):
            transformer.keep |= set(functions)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None


# measurement
@dataclass
class MutantOutcome:
    submission_id: str
    transform: str
    verified: bool
    stable_top1: bool
    overlap: float
    original_top1: Optional[str]
    mutant_top1: Optional[str]


@dataclass
class RobustnessResult:
    outcomes: List[MutantOutcome] = field(default_factory=list)
    skipped: int = 0

    def by_transform(self) -> Dict[str, Dict[str, float]]:
        buckets: Dict[str, List[MutantOutcome]] = {}
        for outcome in self.outcomes:
            buckets.setdefault(outcome.transform, []).append(outcome)
        out: Dict[str, Dict[str, float]] = {}
        for name, rows in sorted(buckets.items()):
            out[name] = {
                "n": float(len(rows)),
                "top1_stability": sum(1 for r in rows if r.stable_top1) / len(rows),
                "mean_overlap": sum(r.overlap for r in rows) / len(rows),
            }
        return out

    def overall(self) -> Dict[str, float]:
        if not self.outcomes:
            return {"n": 0.0, "top1_stability": 0.0, "mean_overlap": 0.0}
        return {
            "n": float(len(self.outcomes)),
            "top1_stability": sum(1 for r in self.outcomes if r.stable_top1) / len(self.outcomes),
            "mean_overlap": sum(r.overlap for r in self.outcomes) / len(self.outcomes),
            "verified_mutants": float(sum(1 for r in self.outcomes if r.verified)),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall": {k: round(v, 4) for k, v in self.overall().items()},
            "by_transform": {
                name: {k: round(v, 4) for k, v in stats.items()}
                for name, stats in self.by_transform().items()
            },
            "unverifiable_mutants_skipped": self.skipped,
            "regressions": [
                {
                    "submission": o.submission_id,
                    "transform": o.transform,
                    "original": o.original_top1,
                    "mutant": o.mutant_top1,
                }
                for o in self.outcomes
                if not o.stable_top1
            ],
        }


class RobustnessStudy:
    """Mutation testing of the diagnosis under semantics preserving rewrites."""

    def __init__(
        self,
        config: Optional[Config] = None,
        orchestrator: Optional[DeductiveOrchestrator] = None,
    ):
        self.config = config or Config()
        self.orchestrator = orchestrator or DeductiveOrchestrator(self.config)
        self.runner = GuardedRunner(self.config.execution)

    def run(
        self,
        *,
        submissions: Optional[Sequence[Submission]] = None,
        transforms: Optional[Sequence[str]] = None,
        seed: int = 20260909,
    ) -> RobustnessResult:
        bank = list(submissions if submissions is not None else labelled())
        names = list(transforms or (list(TRANSFORMS) + [COMPOSITE]))
        result = RobustnessResult()

        for submission in bank:
            spec = get_problem(submission.problem_id)
            baseline = self.orchestrator.diagnose(spec, submission)
            original_top1 = baseline.hits[0].misconception_id if baseline.hits else None
            original_set = {hit.misconception_id for hit in baseline.hits}

            for transform in names:
                rng = random.Random(f"{seed}:{submission.sid}:{transform}")
                mutated = mutate(submission.source, transform, rng)
                if mutated is None:
                    result.skipped += 1
                    continue
                verified = self._behaviourally_identical(spec, submission.source, mutated)
                if not verified:
                    result.skipped += 1
                    continue

                mutant = Submission(
                    sid=f"{submission.sid}#{transform}",
                    problem_id=submission.problem_id,
                    source=mutated,
                    gold_misconceptions=submission.gold_misconceptions,
                    functionally_correct=submission.functionally_correct,
                )
                diagnosis = self.orchestrator.diagnose(spec, mutant)
                introduced = INTRODUCED.get(transform, frozenset()) - original_set
                mutant_hits = [h for h in diagnosis.hits if h.misconception_id not in introduced]
                mutant_top1 = mutant_hits[0].misconception_id if mutant_hits else None
                mutant_set = {hit.misconception_id for hit in mutant_hits}
                result.outcomes.append(
                    MutantOutcome(
                        submission_id=submission.sid,
                        transform=transform,
                        verified=True,
                        stable_top1=(original_top1 == mutant_top1),
                        overlap=jaccard(original_set, mutant_set) if (original_set or mutant_set) else 1.0,
                        original_top1=original_top1,
                        mutant_top1=mutant_top1,
                    )
                )
        return result

    def _behaviourally_identical(self, problem: ProblemSpec, original: str, mutant: str) -> bool:
        """Check the rewrite kept the behaviour on the problem's own tests."""
        entry = problem.entry_point
        for test in problem.tests:
            first = self.runner.run(original, entry, copy.deepcopy(list(test.args)), trace=False)
            second = self.runner.run(mutant, entry, copy.deepcopy(list(test.args)), trace=False)
            if first.ok != second.ok:
                return False
            if first.ok and first.value != second.value:
                return False
            if not first.ok and first.error_type != second.error_type:
                return False
        return True
