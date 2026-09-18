import ast
import random

import pytest

from deductive_mas.analysis.align import (
    classify_divergence,
    locate_divergence,
    needleman_wunsch,
    trajectory,
)
from deductive_mas.analysis.cfg import build_cfg, synchronisation_lines
from deductive_mas.analysis.counterexample import CounterexampleSearch, shrink, shrink_candidates
from deductive_mas.analysis.tracer import GuardedRunner
from deductive_mas.analysis.varmatch import match_variables, series_similarity
from deductive_mas.data.problems import problem
from deductive_mas.domain import DivergenceKind

REFERENCE = (
    "def bs(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] < target:\n"
    "            lo = mid + 1\n"
    "        elif arr[mid] > target:\n"
    "            hi = mid - 1\n"
    "        else:\n"
    "            return mid\n"
    "    return -1\n"
)

RENAMED = (
    "def search(seq, key):\n"
    "    left = 0\n"
    "    right = len(seq) - 1\n"
    "    while left <= right:\n"
    "        centre = (left + right) // 2\n"
    "        if seq[centre] < key:\n"
    "            left = centre + 1\n"
    "        elif seq[centre] > key:\n"
    "            right = centre - 1\n"
    "        else:\n"
    "            return centre\n"
    "    return -1\n"
)

NO_PROGRESS = RENAMED.replace("left = centre + 1", "left = centre")


def _sync(source, entry):
    function = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == entry
    )
    return tuple(sorted(synchronisation_lines(function, build_cfg(function))))


class TestGuardedRunner:
    def test_records_a_value_trace(self):
        result = GuardedRunner().run(REFERENCE, "bs", ([1, 3, 5, 7], 5))
        assert result.ok and result.value == 2
        assert result.events and result.steps > 0

    def test_bounds_a_non_terminating_loop(self):
        source = "def f(n):\n    i = 0\n    while i < n:\n        pass\n    return i\n"
        result = GuardedRunner().run(source, "f", (5,))
        assert result.timed_out and result.error_type == "ExecutionTimeout"

    def test_bounds_a_non_terminating_loop_without_tracing(self):
        runner = GuardedRunner()
        namespace = runner.compile_module("def f(n):\n    i = 0\n    while i < n:\n        pass\n    return i\n")
        result = runner.run_callable(namespace["f"], (5,))
        assert result.timed_out and not result.events

    def test_reports_runaway_recursion(self):
        result = GuardedRunner().run("def f(n):\n    return f(n - 1)\n", "f", (3,))
        assert result.recursion_overflow

    def test_refuses_a_banned_import(self):
        result = GuardedRunner().run("import os\ndef f():\n    return 1\n", "f", ())
        assert not result.ok and "not permitted" in result.error

    def test_refuses_a_banned_builtin(self):
        result = GuardedRunner().run("def f():\n    return eval('1')\n", "f", ())
        assert not result.ok and "not permitted" in result.error

    def test_refuses_dunder_access(self):
        result = GuardedRunner().run("def f():\n    return (1).__class__\n", "f", ())
        assert not result.ok

    def test_allows_a_whitelisted_import(self):
        source = "from collections import deque\ndef f():\n    return len(deque([1, 2]))\n"
        assert GuardedRunner().run(source, "f", ()).value == 2

    def test_reports_a_syntax_error_with_its_line(self):
        result = GuardedRunner().run("def f():\n  return (\n", "f", ())
        assert result.error_type == "SyntaxError"

    def test_reports_a_missing_entry_point(self):
        result = GuardedRunner().run("def other():\n    return 1\n", "f", ())
        assert result.error_type == "MissingEntryPoint"

    def test_locates_the_failing_line(self):
        result = GuardedRunner().run("def f(a):\n    return a[5]\n", "f", ([1, 2],))
        assert result.error_type == "IndexError" and result.error_line == 2

    def test_numeric_series_are_extracted(self):
        result = GuardedRunner().run(REFERENCE, "bs", ([1, 3, 5, 7], 7))
        series = result.variable_series()
        assert {"lo", "hi", "mid"} <= set(series)


class TestVariableMatching:
    def test_matches_by_behaviour_not_by_name(self):
        runner = GuardedRunner()
        args = ([1, 4, 6, 9, 12, 17, 20, 25], 20)
        reference = runner.run(REFERENCE, "bs", args)
        student = runner.run(RENAMED, "search", args)
        mapping = match_variables(
            student, reference, student_params=["seq", "key"], reference_params=["arr", "target"]
        )
        assert mapping.pairs["left"] == "lo"
        assert mapping.pairs["right"] == "hi"
        assert mapping.pairs["centre"] == "mid"
        assert mapping.confidence() > 0.7

    def test_parameters_are_pinned_positionally(self):
        runner = GuardedRunner()
        args = ([1, 2, 3], 2)
        mapping = match_variables(
            runner.run(RENAMED, "search", args),
            runner.run(REFERENCE, "bs", args),
            student_params=["seq", "key"],
            reference_params=["arr", "target"],
        )
        assert "seq" in mapping.forced and mapping.pairs["seq"] == "arr"

    def test_constant_and_varying_series_do_not_match(self):
        assert series_similarity([1.0, 1.0, 1.0], [1.0, 5.0, 9.0]) < 0.5

    def test_identical_series_match_perfectly(self):
        assert series_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


class TestDivergence:
    def _divergence(self, student_source, entry, args):
        runner = GuardedRunner()
        reference = runner.run(REFERENCE, "bs", args)
        student = runner.run(student_source, entry, args)
        mapping = match_variables(
            student,
            reference,
            student_params=["seq", "key"],
            reference_params=["arr", "target"],
        )
        return locate_divergence(
            student,
            reference,
            mapping,
            source_lines=student_source.splitlines(),
            student_sync=_sync(student_source, entry),
            reference_sync=_sync(REFERENCE, "bs"),
        )

    def test_equivalent_code_does_not_diverge(self):
        assert self._divergence(RENAMED, "search", ([1, 4, 6, 9], 6)) is None

    def test_a_missing_increment_is_a_state_mismatch(self):
        divergence = self._divergence(NO_PROGRESS, "search", ([1, 4, 6, 9, 12, 17, 20, 25], 20))
        assert divergence is not None
        assert divergence.kind is DivergenceKind.STATE_MISMATCH
        assert divergence.student_line is not None

    def test_non_termination_is_reported_directly(self):
        divergence = self._divergence(NO_PROGRESS, "search", ([1, 4, 6], 25))
        assert divergence is not None
        assert divergence.kind is DivergenceKind.NON_TERMINATION

    def test_alignment_of_identical_trajectories_is_all_matches(self):
        runner = GuardedRunner()
        args = ([1, 3, 5], 3)
        result = runner.run(REFERENCE, "bs", args)
        path = trajectory(result, ("lo", "hi", "mid"))
        alignment = needleman_wunsch(path, path)
        assert alignment.identity == 1.0 and alignment.cost == 0.0

    def test_alignment_handles_empty_sequences(self):
        assert needleman_wunsch([], []).columns == ()

    def test_classification_falls_back_to_the_output(self):
        runner = GuardedRunner()
        student = runner.run("def f(a):\n    return 1\n", "f", ([1],))
        reference = runner.run("def f(a):\n    return 2\n", "f", ([1],))
        divergence = classify_divergence(student, reference, [], [], ())
        assert divergence is not None and divergence.kind is DivergenceKind.OUTPUT_MISMATCH


class TestCounterexampleSearch:
    def test_finds_and_minimises_a_counterexample(self):
        spec = problem("lower_bound")
        broken = spec.reference_solution.replace("len(a)", "len(a) - 1")
        outcome = CounterexampleSearch().search(spec, broken, rng=random.Random(5))
        assert outcome.found
        assert len(outcome.counterexample.args[0]) <= 2      # shrunk to a tiny witness

    def test_a_correct_solution_yields_none(self):
        spec = problem("lower_bound")
        outcome = CounterexampleSearch().search(spec, spec.reference_solution, rng=random.Random(5))
        assert not outcome.found and outcome.pass_rate == 1.0

    def test_a_compile_error_is_surfaced(self):
        outcome = CounterexampleSearch().search(problem("lower_bound"), "def lower_bound(:\n")
        assert outcome.found and outcome.counterexample.student_error

    def test_shrink_reaches_a_one_minimal_input(self):
        def fails(args):
            return sum(args[0]) >= 3

        minimal, steps = shrink(([5, 5, 5],), fails)
        assert fails(minimal) and steps > 0
        assert sum(minimal[0]) >= 3 and len(minimal[0]) == 1

    def test_shrink_respects_the_precondition(self):
        def fails(args):
            return len(args[0]) >= 1

        def valid(args):
            return list(args[0]) == sorted(args[0])

        minimal, _ = shrink(([1, 2, 3],), fails, valid=valid)
        assert list(minimal[0]) == sorted(minimal[0])

    def test_candidates_shrink_toward_simplicity(self):
        assert 0 in list(shrink_candidates(8))
        assert "" in list(shrink_candidates("ab"))
        assert list(shrink_candidates(0)) == []

    def test_list_candidates_follow_the_ddmin_lattice(self):
        # ddmin drops halves first then single elements, it never goes from two
        # elements to the empty list in one step, that is why it gets to a minimal
        # witness in O(log n)
        candidates = [list(c) for c in shrink_candidates([1, 2])]
        assert [1] in candidates and [2] in candidates
        assert [] in [list(c) for c in shrink_candidates([1])]
