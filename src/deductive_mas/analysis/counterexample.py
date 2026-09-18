"""Counterexample search and shrinking.

A hint is much more convincing when it is concrete ("on [1, 2] looking for 2
your function returns -1"). So before the intervention agent says anything
we look for an input where the student program is wrong and shrink it.

- property based search (QuickCheck style): declared test cases first, then
  random inputs from the problem's generator, the reference is the oracle
- delta debugging (ddmin, but typed / structural): keep proposing smaller
  inputs and keep any that still fails until no single step helps. Result is
  1-minimal.

Every candidate goes through the problem's precondition so we never "find" a
counterexample outside the valid input space.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

from ..domain import Counterexample, ProblemSpec
from .tracer import ExecutionResult, GuardedRunner

Args = Tuple[Any, ...]
Predicate = Callable[[Args], bool]


@dataclass
class SearchOutcome:
    """What the search found and how hard it had to look."""

    counterexample: Optional[Counterexample] = None
    attempts: int = 0
    passed: int = 0
    shrink_steps: int = 0
    source: str = "none"          # "test-case" | "random" | "none"
    tests_failed: Tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return self.counterexample is not None

    @property
    def pass_rate(self) -> float:
        return self.passed / self.attempts if self.attempts else 0.0


# structural shrinking
def shrink_candidates(value: Any) -> Iterator[Any]:
    """Yield simpler versions of value, simplest first.

    For sequences this is the ddmin lattice: drop first half, drop second half,
    then single elements. For numbers it walks towards zero geometrically.
    Trying the big reductions first is what keeps it close to O(log n).
    """
    if isinstance(value, bool):
        if value:
            yield False
        return
    if isinstance(value, int):
        if value == 0:
            return
        yield 0
        if abs(value) > 1:
            yield value // 2
            yield value - 1 if value > 0 else value + 1
        return
    if isinstance(value, float):
        if value == 0.0:
            return
        yield 0.0
        yield value / 2.0
        return
    if isinstance(value, str):
        if not value:
            return
        yield ""
        if len(value) > 1:
            yield value[: len(value) // 2]
            yield value[1:]
            yield value[:-1]
        return
    if isinstance(value, (list, tuple)):
        rebuild = (lambda xs: tuple(xs)) if isinstance(value, tuple) else (lambda xs: list(xs))
        items = list(value)
        n = len(items)
        if n == 0:
            return
        if n > 1:
            half = n // 2
            yield rebuild(items[:half])
            yield rebuild(items[half:])
        for index in range(n):
            yield rebuild(items[:index] + items[index + 1 :])
        for index in range(n):
            for smaller in shrink_candidates(items[index]):
                yield rebuild(items[:index] + [smaller] + items[index + 1 :])
        return
    if isinstance(value, set):
        items = sorted(value, key=repr)
        for index in range(len(items)):
            yield set(items[:index] + items[index + 1 :])
        return
    if isinstance(value, dict):
        keys = sorted(value, key=repr)
        for key in keys:
            reduced = dict(value)
            reduced.pop(key)
            yield reduced
        return


def shrink(
    args: Args,
    fails: Predicate,
    *,
    valid: Optional[Predicate] = None,
    max_steps: int = 400,
) -> Tuple[Args, int]:
    """Greedily shrink args while fails still holds.
    Returns the shrunk args and the number of accepted steps. On exit no single
    reduction of any component still fails (1-minimal).
    """
    current = args
    steps = 0
    budget = max_steps
    improved = True
    while improved and budget > 0:
        improved = False
        for index in range(len(current)):
            for candidate in shrink_candidates(current[index]):
                budget -= 1
                if budget <= 0:
                    break
                trial = current[:index] + (candidate,) + current[index + 1 :]
                if valid is not None and not valid(trial):
                    continue
                if fails(trial):
                    current = trial
                    steps += 1
                    improved = True
                    break
            if improved:
                break
    return current, steps


# search
class CounterexampleSearch:
    """Differential testing of a submission against the reference."""

    def __init__(self, runner: Optional[GuardedRunner] = None):
        self.runner = runner or GuardedRunner()

    def search(
        self,
        problem: ProblemSpec,
        student_source: str,
        *,
        student_entry: Optional[str] = None,
        random_attempts: int = 120,
        rng: Optional[random.Random] = None,
        shrink_budget: int = 400,
    ) -> SearchOutcome:
        rng = rng or random.Random(0xC0FFEE)
        entry = student_entry or problem.entry_point

        try:
            student_ns = self.runner.compile_module(student_source)
        except Exception as exc:
            return SearchOutcome(
                counterexample=Counterexample(
                    args=problem.tests[0].args if problem.tests else (),
                    student_output=None,
                    reference_output=problem.tests[0].expected if problem.tests else None,
                    student_error=f"{type(exc).__name__}: {exc}",
                ),
                attempts=1,
                source="compile-error",
            )
        if not problem.reference_solution.strip():
            return SearchOutcome(source="no-reference")
        reference_ns = self.runner.compile_module(problem.reference_solution)
        student_fn = student_ns.get(entry)
        reference_fn = reference_ns.get(problem.entry_point)
        if not callable(reference_fn):
            return SearchOutcome(source="no-reference")
        if not callable(student_fn):
            return SearchOutcome(
                counterexample=Counterexample(
                    args=(),
                    student_output=None,
                    reference_output=None,
                    student_error=f"entry point {entry!r} is not defined",
                ),
                attempts=1,
                source="missing-entry",
            )

        outcome = SearchOutcome()
        failures: List[str] = []

        def evaluate(args: Args) -> Optional[Counterexample]:
            student = self.runner.run_callable(student_fn, _copy_args(args))
            expected = self.runner.run_callable(reference_fn, _copy_args(args))
            if not expected.ok:
                return None                       # outside the valid input space
            if student.ok and _equivalent(student.value, expected.value):
                return None
            return Counterexample(
                args=args,
                student_output=student.value if student.ok else None,
                reference_output=expected.value,
                student_error=student.error if not student.ok else None,
            )

        def fails(args: Args) -> bool:
            return evaluate(args) is not None

        valid: Optional[Predicate] = problem.validator

        # 1. the declared test cases, cheapest and easiest to read
        for test in problem.tests:
            outcome.attempts += 1
            found = evaluate(test.args)
            if found is None:
                outcome.passed += 1
                continue
            failures.append(test.label or repr(test.args))
            minimal, steps = shrink(test.args, fails, valid=valid, max_steps=shrink_budget)
            final = evaluate(minimal) or found
            outcome.counterexample = Counterexample(
                args=final.args,
                student_output=final.student_output,
                reference_output=final.reference_output,
                student_error=final.student_error,
                shrink_steps=steps,
            )
            outcome.shrink_steps = steps
            outcome.source = "test-case"
            outcome.tests_failed = tuple(failures)
            # keep scoring the remaining tests so the pass rate means something
            for remaining in problem.tests[problem.tests.index(test) + 1 :]:
                outcome.attempts += 1
                if evaluate(remaining.args) is None:
                    outcome.passed += 1
                else:
                    failures.append(remaining.label or repr(remaining.args))
            outcome.tests_failed = tuple(failures)
            return outcome

        # 2. random differential testing
        if problem.sampler is not None:
            for _ in range(random_attempts):
                args = problem.sampler(rng)
                if valid is not None and not valid(args):
                    continue
                outcome.attempts += 1
                found = evaluate(args)
                if found is None:
                    outcome.passed += 1
                    continue
                minimal, steps = shrink(args, fails, valid=valid, max_steps=shrink_budget)
                final = evaluate(minimal) or found
                outcome.counterexample = Counterexample(
                    args=final.args,
                    student_output=final.student_output,
                    reference_output=final.reference_output,
                    student_error=final.student_error,
                    shrink_steps=steps,
                )
                outcome.shrink_steps = steps
                outcome.source = "random"
                outcome.tests_failed = tuple(failures)
                return outcome

        outcome.tests_failed = tuple(failures)
        return outcome


def _copy_args(args: Args) -> List[Any]:
    """Copy so a submission that mutates its input cannot poison the oracle."""
    out: List[Any] = []
    for value in args:
        if isinstance(value, list):
            out.append([_copy_scalar(v) for v in value])
        elif isinstance(value, dict):
            out.append({k: _copy_scalar(v) for k, v in value.items()})
        elif isinstance(value, set):
            out.append(set(value))
        else:
            out.append(value)
    return out


def _copy_scalar(value: Any) -> Any:
    if isinstance(value, list):
        return [_copy_scalar(v) for v in value]
    if isinstance(value, dict):
        return {k: _copy_scalar(v) for k, v in value.items()}
    if isinstance(value, set):
        return set(value)
    return value


def _equivalent(a: Any, b: Any) -> bool:
    """Output comparison that allows list/tuple and int/float mixing."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    return a == b
