"""External validation on real student code: the Refactory corpus (Hu et al.,
ASE 2019), 4,225 Python submissions to five intro assignments, each marked
correct / wrong by the course's own tests. The labels are objective and
made by third parties, but they say nothing about WHY a program is wrong,
so this cannot score diagnostic accuracy. What it can score:

- false accusation rate: how often a correctness misconception is named on
  a program the course accepted
- detection: how often differential testing finds a failing input on a
  rejected program, and how often that is localised to a line and explained
  by a named belief
- cost of the analysis on programs we never tuned on

The corpus is not redistributed, point --data at an unpacked copy of its
data directory. The course reference solutions are the oracle.
"""

import ast
import contextlib
import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..agents.base import Blackboard
from ..agents.orchestrator import DeductiveOrchestrator
from ..config import Config
from ..domain import ProblemSpec, Submission, TestCase
from ..knowledge.misconceptions import is_correctness_misconception

_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


# samplers and validators for the five assignments
def _sample_search(rng: Random) -> Tuple[Any, ...]:
    n = rng.randint(0, 8)
    values = sorted(rng.randint(-8, 10) for _ in range(n))
    # the course tests pass both tuples and lists
    return (rng.randint(-10, 12), tuple(values) if rng.random() < 0.5 else list(values))


def _valid_search(args: Tuple[Any, ...]) -> bool:
    return (
        len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], (tuple, list))
        and all(isinstance(v, int) for v in args[1]) and list(args[1]) == sorted(args[1])
    )


def _sample_birthdays(rng: Random) -> Tuple[Any, ...]:
    n = rng.randint(0, 6)
    months = rng.sample(_MONTHS, k=min(4, len(_MONTHS)))
    pairs = tuple((rng.choice(months), str(rng.randint(1, 31))) for _ in range(n))
    return pairs


def _sample_unique_day(rng: Random) -> Tuple[Any, ...]:
    pairs = _sample_birthdays(rng)
    day = rng.choice([p[1] for p in pairs]) if pairs and rng.random() < 0.7 else str(rng.randint(1, 31))
    return (day, pairs)


def _sample_unique_month(rng: Random) -> Tuple[Any, ...]:
    pairs = _sample_birthdays(rng)
    month = rng.choice([p[0] for p in pairs]) if pairs and rng.random() < 0.7 else rng.choice(_MONTHS)
    return (month, pairs)


def _valid_birthdays(args: Tuple[Any, ...]) -> bool:
    return (
        len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], tuple)
        and all(isinstance(p, tuple) and len(p) == 2 and all(isinstance(x, str) for x in p)
                for p in args[1])
    )


def _sample_remove_extras(rng: Random) -> Tuple[Any, ...]:
    n = rng.randint(0, 8)
    return ([rng.randint(0, 4) for _ in range(n)],)


def _valid_remove_extras(args: Tuple[Any, ...]) -> bool:
    return len(args) == 1 and isinstance(args[0], list) and all(isinstance(v, int) for v in args[0])


def _sample_sort_age(rng: Random) -> Tuple[Any, ...]:
    # the course records are ("M" | "F", age), distinct ages make the order unique
    n = rng.randint(0, 6)
    ages = rng.sample(range(1, 60), k=n)
    return ([(rng.choice(("M", "F")), age) for age in ages],)


def _valid_sort_age(args: Tuple[Any, ...]) -> bool:
    if len(args) != 1 or not isinstance(args[0], list):
        return False
    ages = []
    for pair in args[0]:
        if not (isinstance(pair, tuple) and len(pair) == 2 and pair[0] in ("M", "F")
                and isinstance(pair[1], int) and pair[1] >= 0):
            return False
        ages.append(pair[1])
    return len(ages) == len(set(ages))


def _sample_top_k(rng: Random) -> Tuple[Any, ...]:
    # the course never tests an empty list, contract is n >= 1, 0 <= k <= n
    n = rng.randint(1, 8)
    values = [rng.randint(1, 9) for _ in range(n)]
    return (values, rng.randint(0, n))


def _valid_top_k(args: Tuple[Any, ...]) -> bool:
    return (
        len(args) == 2 and isinstance(args[0], list) and len(args[0]) >= 1
        and all(isinstance(v, int) for v in args[0])
        and isinstance(args[1], int) and 0 <= args[1] <= len(args[0])
    )


@dataclass(frozen=True)
class _Assignment:
    question: int
    entry: str
    parameters: Tuple[str, ...]
    title: str
    concepts: Tuple[str, ...]
    sampler: Callable[[Random], Tuple[Any, ...]]
    validator: Callable[[Tuple[Any, ...]], bool]


_ASSIGNMENTS: Tuple[_Assignment, ...] = (
    _Assignment(1, "search", ("x", "seq"), "Sequential search in a sorted sequence",
                ("iteration", "ordering-relations", "boundary-conditions"), _sample_search, _valid_search),
    _Assignment(2, "unique_day", ("day", "possible_birthdays"), "Unique birthday day",
                ("iteration", "boolean-logic", "sequence-indexing"), _sample_unique_day, _valid_birthdays),
    _Assignment(2, "unique_month", ("month", "possible_birthdays"), "Unique birthday month",
                ("iteration", "boolean-logic", "sequence-indexing"), _sample_unique_month, _valid_birthdays),
    _Assignment(2, "contains_unique_day", ("month", "possible_birthdays"), "Month with a unique day",
                ("iteration", "boolean-logic", "sequence-indexing"), _sample_unique_month, _valid_birthdays),
    _Assignment(3, "remove_extras", ("lst",), "Remove duplicates",
                ("iteration", "array-list", "cost-model"), _sample_remove_extras, _valid_remove_extras),
    _Assignment(4, "sort_age", ("lst",), "Sort by age, descending",
                ("comparison-sorting", "iteration", "sequence-indexing"), _sample_sort_age, _valid_sort_age),
    _Assignment(5, "top_k", ("lst", "k"), "The k largest elements",
                ("iteration", "array-list", "cost-model"), _sample_top_k, _valid_top_k),
)


# loading
@dataclass
class ExternalTask:
    spec: ProblemSpec
    question: int
    submissions: List[Submission] = field(default_factory=list)


def _parse_call(text: str, namespace: Dict[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
    """Split f(a, b) into name and argument values.
    Arguments are literals or names bound by the assignment's global.py (the
    course test harness refers to a shared birthday table that way).
    """
    tree = ast.parse(text.strip(), mode="eval")
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError(f"not a call: {text!r}")
    values = []
    for node in call.args:
        if isinstance(node, ast.Name):
            values.append(namespace[node.id])
        else:
            values.append(ast.literal_eval(node))
    return call.func.id, tuple(values)


def load_refactory(root: Path, *, limit: Optional[int] = None, seed: int = 0) -> List[ExternalTask]:
    """One task per assignment entry point from an unpacked data dir."""
    root = Path(root)
    tasks: List[ExternalTask] = []
    for assignment in _ASSIGNMENTS:
        qdir = root / f"question_{assignment.question}"
        if not qdir.exists():
            continue
        prelude = ""
        namespace: Dict[str, Any] = {}
        global_file = qdir / "code" / "global.py"
        if global_file.exists():
            prelude = global_file.read_text(encoding="utf-8", errors="replace").strip()
            if prelude:
                exec(prelude, namespace)
            prelude = prelude + "\n\n" if prelude else ""
        reference = (qdir / "code" / "reference" / "reference.py").read_text(encoding="utf-8")

        tests: List[TestCase] = []
        for input_path in sorted((qdir / "ans").glob("input_*.txt")):
            index = input_path.stem.split("_")[-1]
            output_path = qdir / "ans" / f"output_{index}.txt"
            if not output_path.exists():
                continue
            name, args = _parse_call(input_path.read_text(encoding="utf-8"), namespace)
            if name != assignment.entry:
                continue
            expected = ast.literal_eval(output_path.read_text(encoding="utf-8").strip())
            if not assignment.validator(args):
                continue
            tests.append(TestCase(args, expected, f"course test {index}"))
        if not tests:
            continue

        spec = ProblemSpec(
            pid=f"refactory_q{assignment.question}_{assignment.entry}",
            title=assignment.title,
            statement=f"Refactory assignment {assignment.question}: {assignment.title}.",
            entry_point=assignment.entry,
            parameters=assignment.parameters,
            concepts=assignment.concepts,
            reference_solution=prelude + reference,
            tests=tuple(tests),
            sampler=assignment.sampler,
            validator=assignment.validator,
            tags=("refactory",),
        )
        task = ExternalTask(spec=spec, question=assignment.question)
        rng = Random(seed)
        for kind, correct in (("correct", True), ("wrong", False)):
            files = sorted((qdir / "code" / kind).glob("*.py"))
            if limit is not None and len(files) > limit:
                files = rng.sample(files, limit)
                files.sort()
            for path in files:
                source = path.read_text(encoding="utf-8", errors="replace")
                task.submissions.append(
                    Submission(
                        sid=f"{path.stem}:{assignment.entry}",
                        problem_id=spec.pid,
                        source=prelude + source,
                        author="refactory",
                        functionally_correct=correct,
                    )
                )
        tasks.append(task)
    return tasks


# measurement
@dataclass
class ExternalRow:
    submission_id: str
    question: int
    entry: str
    course_correct: bool
    parsed: bool
    unsafe: bool
    counterexample: bool
    checks_passed: int
    checks_total: int
    located: bool
    hits: Tuple[Tuple[str, float], ...]
    primary: Optional[str]
    seconds: float

    @property
    def names_correctness_belief(self) -> bool:
        return any(is_correctness_misconception(mid) for mid, _ in self.hits)

    @property
    def names_any_belief(self) -> bool:
        return bool(self.hits)


@dataclass
class _FileOutcome:
    name: str
    course_correct: bool
    usable: bool
    counterexample: bool
    located: bool
    correctness_belief: bool
    any_belief: bool
    seconds: float


@dataclass
class ExternalResult:
    rows: List[ExternalRow] = field(default_factory=list)
    backend: str = "offline"

    @staticmethod
    def _by_file(rows: Sequence[ExternalRow]) -> List["_FileOutcome"]:
        """Collapse per entry rows to one outcome per student file.
        Assignment 2 has three entry points and the course marked a FILE wrong when
        any failed, so combine the same way: detected if any entry gives a
        counterexample, accused if any entry names a correctness belief.
        """
        groups: Dict[str, List[ExternalRow]] = {}
        for row in rows:
            groups.setdefault(row.submission_id.split(":")[0], []).append(row)
        out: List[_FileOutcome] = []
        for name, members in sorted(groups.items()):
            out.append(
                _FileOutcome(
                    name=name,
                    course_correct=members[0].course_correct,
                    usable=all(r.parsed and not r.unsafe for r in members),
                    counterexample=any(r.counterexample for r in members),
                    located=any(r.located for r in members),
                    correctness_belief=any(r.names_correctness_belief for r in members),
                    any_belief=any(r.names_any_belief for r in members),
                    seconds=sum(r.seconds for r in members),
                )
            )
        return out

    def _summary(self, rows: Sequence[ExternalRow]) -> Dict[str, Any]:
        files = self._by_file(rows)
        correct = [f for f in files if f.course_correct and f.usable]
        wrong = [f for f in files if not f.course_correct and f.usable]

        def rate(subset, predicate) -> Optional[float]:
            return round(sum(1 for f in subset if predicate(f)) / len(subset), 4) if subset else None

        wrong_detected = [f for f in wrong if f.counterexample]
        return {
            "n_files": len(files),
            "n_correct": len(correct),
            "n_wrong": len(wrong),
            "unparsed_or_unsafe": sum(1 for f in files if not f.usable),
            # programs the course accepted
            "correct__false_accusation_rate": rate(correct, lambda f: f.correctness_belief),
            "correct__any_finding_rate": rate(correct, lambda f: f.any_belief),
            "correct__cost_or_hygiene_only_rate": rate(
                correct, lambda f: f.any_belief and not f.correctness_belief
            ),
            "correct__counterexample_rate": rate(correct, lambda f: f.counterexample),
            # programs the course rejected
            "wrong__counterexample_rate": rate(wrong, lambda f: f.counterexample),
            "wrong__localised_rate": rate(wrong, lambda f: f.located),
            "wrong__named_correctness_belief_rate": rate(wrong, lambda f: f.correctness_belief),
            "wrong__unexplained_failure_rate": rate(
                wrong, lambda f: f.counterexample and not f.correctness_belief
            ),
            "wrong__detected_then_explained_rate": rate(
                wrong_detected, lambda f: f.correctness_belief
            ),
            "mean_seconds_per_file": round(sum(f.seconds for f in files) / len(files), 4) if files else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        from collections import Counter

        by_question: Dict[str, Any] = {}
        for question in sorted({r.question for r in self.rows}):
            by_question[f"q{question}"] = self._summary([r for r in self.rows if r.question == question])
        primaries_wrong = Counter(r.primary for r in self.rows if not r.course_correct and r.primary)
        primaries_correct = Counter(r.primary for r in self.rows if r.course_correct and r.primary)
        return {
            "backend": self.backend,
            "overall": self._summary(self.rows),
            "by_question": by_question,
            "primary_on_wrong": dict(primaries_wrong.most_common()),
            "primary_on_correct": dict(primaries_correct.most_common()),
            "rows": [
                {
                    "submission": r.submission_id,
                    "question": r.question,
                    "entry": r.entry,
                    "course_correct": r.course_correct,
                    "parsed": r.parsed,
                    "unsafe": r.unsafe,
                    "counterexample": r.counterexample,
                    "checks": [r.checks_passed, r.checks_total],
                    "located": r.located,
                    "hits": [[mid, round(b, 4)] for mid, b in r.hits],
                    "primary": r.primary,
                    "seconds": round(r.seconds, 4),
                }
                for r in self.rows
            ],
        }


class RefactoryStudy:
    """Runs the evaluator over the corpus and tabulates the outcomes above."""

    def __init__(
        self,
        config: Optional[Config] = None,
        orchestrator: Optional[DeductiveOrchestrator] = None,
    ):
        self.config = config or Config()
        self.orchestrator = orchestrator or DeductiveOrchestrator(self.config)

    def run(self, tasks: Sequence[ExternalTask], *, progress: Optional[Callable[[str], None]] = None) -> ExternalResult:
        result = ExternalResult(backend=self.orchestrator.reasoner.name)
        for task in tasks:
            for index, submission in enumerate(task.submissions):
                started = time.perf_counter()
                board = Blackboard(problem=task.spec, submission=submission)
                board.telemetry.backend = self.orchestrator.reasoner.name
                # student programs print a lot, keep that out of the report
                with contextlib.redirect_stdout(io.StringIO()):
                    self.orchestrator.evaluator(board)
                seconds = time.perf_counter() - started
                diagnosis = board.diagnosis
                analysis = board.analysis
                parsed = bool(analysis is not None and analysis.parsed)
                search_source = analysis.search.source if analysis is not None else "none"
                unsafe = search_source in ("compile-error", "missing-entry")
                counterexample = bool(
                    parsed and not unsafe and diagnosis is not None
                    and diagnosis.counterexample is not None
                )
                hits = tuple((h.misconception_id, h.belief) for h in diagnosis.hits) if diagnosis else ()
                result.rows.append(
                    ExternalRow(
                        submission_id=submission.sid,
                        question=task.question,
                        entry=task.spec.entry_point,
                        course_correct=submission.functionally_correct,
                        parsed=parsed,
                        unsafe=unsafe,
                        counterexample=counterexample,
                        checks_passed=diagnosis.checks_passed if diagnosis else 0,
                        checks_total=diagnosis.checks_total if diagnosis else 0,
                        located=bool(diagnosis and diagnosis.divergence is not None),
                        hits=hits,
                        primary=(diagnosis.primary.misconception_id
                                 if diagnosis and diagnosis.primary else None),
                        seconds=seconds,
                    )
                )
                if progress and (index + 1) % 100 == 0:
                    progress(f"{task.spec.pid}: {index + 1}/{len(task.submissions)}")
        return result
