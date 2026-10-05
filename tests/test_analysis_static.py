import ast

import pytest

from deductive_mas.analysis.ast_features import extract
from deductive_mas.analysis.cfg import build_cfg, synchronisation_lines
from deductive_mas.analysis.complexity import Asymptotic, estimate
from deductive_mas.analysis.dataflow import analyse as solve_dataflow
from deductive_mas.analysis.normalize import (
    normalise,
    shape_equivalent,
    structurally_equivalent,
)

BINARY_SEARCH = """
def bs(a, t):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == t:
            return mid
        elif a[mid] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""


def _function(source, name):
    return next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class TestNormalisation:
    def test_renaming_is_invisible(self):
        a = "def f(a, t):\n    lo = 0\n    return lo\n"
        b = "def g(arr, key):\n    left = 0\n    return left\n"
        assert structurally_equivalent(a, b)

    def test_boundary_changes_are_visible(self):
        a = "def f(x, y):\n    return x < y\n"
        b = "def f(x, y):\n    return x <= y\n"
        assert not structurally_equivalent(a, b)
        assert not shape_equivalent(a, b)

    def test_docstrings_are_ignored(self):
        a = 'def f(x):\n    """Doc."""\n    return x\n'
        b = "def f(x):\n    return x\n"
        assert structurally_equivalent(a, b)

    def test_unparsable_source_returns_none(self):
        assert normalise("def f(:\n") is None

    def test_identifier_map_is_recorded(self):
        form = normalise("def f(alpha):\n    beta = alpha\n    return beta\n")
        assert set(form.identifier_map) == {"f", "alpha", "beta"}


class TestControlFlow:
    def test_loop_and_dead_code_are_found(self):
        source = BINARY_SEARCH + "    print('unreachable')\n"
        cfg = build_cfg(_function(source, "bs"))
        assert cfg.back_edges()
        assert cfg.natural_loops()
        assert cfg.unreachable_lines()

    def test_cyclomatic_complexity_grows_with_branching(self):
        simple = build_cfg(_function("def f(x):\n    return x\n", "f"))
        branchy = build_cfg(_function(BINARY_SEARCH, "bs"))
        assert branchy.cyclomatic_complexity > simple.cyclomatic_complexity

    def test_reverse_post_order_starts_at_the_entry(self):
        cfg = build_cfg(_function(BINARY_SEARCH, "bs"))
        assert cfg.rpo()[0] == cfg.entry

    def test_entry_dominates_everything(self):
        cfg = build_cfg(_function(BINARY_SEARCH, "bs"))
        dominators = cfg.dominators()
        assert all(cfg.entry in dom for dom in dominators.values())

    def test_synchronisation_lines_include_the_loop_header(self):
        function = _function(BINARY_SEARCH, "bs")
        lines = synchronisation_lines(function, build_cfg(function))
        assert 4 in lines            # while lo <= hi:

    def test_loop_kind_is_recorded(self):
        cfg = build_cfg(_function("def f(xs):\n    for x in xs:\n        pass\n", "f"))
        kinds = {b.loop_kind for b in cfg.blocks.values() if b.is_loop_header}
        assert kinds == {"for"}


class TestDataflow:
    def test_detects_use_before_definition(self):
        source = "def f(a):\n    return a + z\n"
        function = _function(source, "f")
        facts = solve_dataflow(build_cfg(function), ["a"])
        assert ("z", 2) in facts.uninitialised

    def test_detects_a_dead_store(self):
        source = "def f(a):\n    junk = 42\n    return a\n"
        function = _function(source, "f")
        facts = solve_dataflow(build_cfg(function), ["a"])
        assert any(name == "junk" for name, _, _ in facts.dead_stores)

    def test_flags_a_loop_that_cannot_progress(self):
        source = "def f(n):\n    i = 0\n    while i < n:\n        pass\n    return i\n"
        function = _function(source, "f")
        facts = solve_dataflow(build_cfg(function), ["n"])
        assert facts.non_terminating_loops()

    def test_accepts_a_loop_that_progresses(self):
        source = "def f(n):\n    i = 0\n    while i < n:\n        i += 1\n    return i\n"
        function = _function(source, "f")
        facts = solve_dataflow(build_cfg(function), ["n"])
        assert not facts.non_terminating_loops()

    def test_container_mutation_counts_as_progress(self):
        source = (
            "def f(xs):\n    q = list(xs)\n    out = 0\n"
            "    while q:\n        q.pop()\n        out += 1\n    return out\n"
        )
        function = _function(source, "f")
        facts = solve_dataflow(build_cfg(function), ["xs"])
        assert not facts.non_terminating_loops()

    def test_lambda_parameters_are_not_free_variables(self):
        source = "def f(xs):\n    return sorted(xs, key=lambda p: p[1])\n"
        function = _function(source, "f")
        facts = solve_dataflow(build_cfg(function), ["xs"])
        assert not any(name == "p" for name, _ in facts.uninitialised)

    def test_loop_carried_variables_are_identified(self):
        function = _function(BINARY_SEARCH, "bs")
        facts = solve_dataflow(build_cfg(function), ["a", "t"])
        carried = set().union(*facts.loop_carried.values()) if facts.loop_carried else set()
        assert {"lo", "hi"} <= carried


class TestComplexity:
    @pytest.mark.parametrize(
        "source,entry,expected",
        [
            (BINARY_SEARCH, "bs", "O(log n)"),
            ("def f(a, t):\n    for i in range(len(a)):\n        if a[i] == t: return i\n    return -1\n", "f", "O(n)"),
            ("def f(a):\n    for i in range(len(a)):\n        for j in range(len(a)):\n            pass\n    return a\n", "f", "O(n^2)"),
            ("def fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\n", "fib", "O(2^n)"),
            ("import functools\n@functools.lru_cache(None)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\n", "fib", "O(n)"),
            ("def f(n):\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0: return False\n    return True\n", "f", "O(sqrt(n))"),
            ("def f(a):\n    b = sorted(a)\n    return b\n", "f", "O(n log n)"),
        ],
    )
    def test_known_complexities(self, source, entry, expected):
        assert str(estimate(source, entry).total) == expected

    def test_merge_sort_solves_by_the_master_theorem(self):
        source = (
            "def ms(a):\n"
            "    if len(a) < 2: return a\n"
            "    mid = len(a) // 2\n"
            "    left = ms(a[:mid])\n"
            "    right = ms(a[mid:])\n"
            "    out = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] < right[j]:\n"
            "            out.append(left[i]); i += 1\n"
            "        else:\n"
            "            out.append(right[j]); j += 1\n"
            "    return out + left[i:] + right[j:]\n"
        )
        report = estimate(source, "ms")
        assert str(report.total) == "O(n log n)"
        assert report.recurrence and "T(n) = 2 T(n/2)" in report.recurrence

    def test_graph_traversal_is_amortised(self):
        fast = (
            "from collections import deque\n"
            "def bfs(g, s):\n    q = deque([s]); seen = set()\n"
            "    while q:\n        u = q.popleft()\n"
            "        for v in g[u]:\n"
            "            if v not in seen:\n                seen.add(v); q.append(v)\n"
            "    return seen\n"
        )
        slow = (
            "def bfs(g, s):\n    q = [s]; seen = []\n"
            "    while q:\n        u = q.pop(0)\n"
            "        for v in g[u]:\n"
            "            if v not in seen:\n                seen.append(v); q.append(v)\n"
            "    return seen\n"
        )
        assert str(estimate(fast, "bfs").total) == "O(n)"
        assert estimate(slow, "bfs").total.dominates(estimate(fast, "bfs").total)

    def test_recursion_inside_a_loop_is_exponential(self):
        source = (
            "def f(coins, amount):\n"
            "    if amount == 0: return 0\n"
            "    best = -1\n"
            "    for c in coins:\n"
            "        if c <= amount:\n"
            "            rest = f(coins, amount - c)\n"
            "            if rest >= 0 and (best < 0 or rest + 1 < best): best = rest + 1\n"
            "    return best\n"
        )
        assert estimate(source, "f").total.is_exponential

    def test_asymptotic_ordering(self):
        assert Asymptotic.linear().dominates(Asymptotic.logarithmic())
        assert Asymptotic.linearithmic().dominates(Asymptotic.linear())
        assert Asymptotic.exponential().dominates(Asymptotic.polynomial(9))
        assert not Asymptotic.constant().dominates(Asymptotic.constant())

    def test_syntax_error_is_reported_not_raised(self):
        report = estimate("def f(:\n", "f")
        assert report.notes and "syntax" in report.notes[0]


class TestFeatures:
    def test_container_kinds_are_inferred(self):
        features = extract("def f():\n    s = set()\n    d = {}\n    l = []\n    return s, d, l\n", "f")
        assert features.container_kinds == {"s": "set", "d": "dict", "l": "list"}

    def test_mutable_default_is_detected(self):
        features = extract("def f(x, memo={}):\n    return x\n", "f")
        assert features.mutable_defaults == (("memo", 1),)

    def test_recursion_without_a_base_case(self):
        features = extract("def f(n):\n    return f(n - 1)\n", "f")
        assert features.is_recursive and not features.has_base_case

    def test_two_self_calls_on_one_line_count_twice(self):
        features = extract("def f(n):\n    if n < 2: return n\n    return f(n-1) + f(n-2)\n", "f")
        assert features.self_call_count == 2

    def test_loop_extent_excludes_nodes_without_a_line(self):
        features = extract(
            "def f(xs):\n    y = sorted(xs)\n    for x in y:\n        pass\n    return y\n", "f"
        )
        loop = features.loops[0]
        assert loop.body_lines[0] >= 3          # the sort on line 2 is outside
        assert features.loop_containing(2) is None

    def test_missing_entry_point_falls_back_to_the_first_function(self):
        features = extract("def other(x):\n    return x\n", "expected")
        assert features.function is not None and features.function.name == "other"

    def test_syntax_error_is_recorded(self):
        features = extract("def f(:\n", "f")
        assert not features.parsed and features.parse_error
