"""The dev problem bank: six tasks (searching, graph traversal, DP, two
pointers, greedy). Each has a reference solution (never shown to the
student, used as oracle and as target for the leak gate), boundary tests
plus a random generator with a precondition, and IRT calibration on the
same scale as the ontology.
"""

from random import Random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..domain import ProblemSpec, Submission, TestCase

# 1. Lower bound in a sorted array
_LOWER_BOUND_REF = '''
def lower_bound(a, t):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo
'''.strip()


def _sample_sorted(rng: Random) -> Tuple[Any, ...]:
    n = rng.randint(0, 10)
    values = sorted(rng.randint(-6, 6) for _ in range(n))
    return (values, rng.randint(-7, 7))


def _valid_sorted(args: Tuple[Any, ...]) -> bool:
    return (
        len(args) == 2
        and isinstance(args[0], list)
        and all(isinstance(x, int) for x in args[0])
        and list(args[0]) == sorted(args[0])
        and isinstance(args[1], int)
    )


# 2. Breadth-first shortest hop counts
_BFS_REF = '''
from collections import deque

def hop_counts(adj, source):
    n = len(adj)
    distance = [-1] * n
    if n == 0:
        return distance
    distance[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in adj[node]:
            if distance[neighbour] == -1:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return distance
'''.strip()


def _sample_graph(rng: Random) -> Tuple[Any, ...]:
    n = rng.randint(1, 7)
    adj: List[List[int]] = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < 0.35:
                adj[u].append(v)
    return (adj, rng.randrange(n))


def _valid_graph(args: Tuple[Any, ...]) -> bool:
    if len(args) != 2 or not isinstance(args[0], list) or not isinstance(args[1], int):
        return False
    adj, source = args
    n = len(adj)
    if n == 0 or not 0 <= source < n:
        return False
    return all(
        isinstance(row, list) and all(isinstance(v, int) and 0 <= v < n for v in row)
        for row in adj
    )


# 3. Coin change (minimum number of coins)
_COINS_REF = '''
def min_coins(coins, amount):
    best = [0] + [amount + 1] * amount
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value and best[value - coin] + 1 < best[value]:
                best[value] = best[value - coin] + 1
    return best[amount] if best[amount] <= amount else -1
'''.strip()


def _sample_coins(rng: Random) -> Tuple[Any, ...]:
    coins = sorted({rng.randint(1, 9) for _ in range(rng.randint(1, 4))})
    return (list(coins), rng.randint(0, 30))


def _valid_coins(args: Tuple[Any, ...]) -> bool:
    return (
        len(args) == 2
        and isinstance(args[0], list)
        and len(args[0]) > 0
        and all(isinstance(c, int) and c > 0 for c in args[0])
        and isinstance(args[1], int)
        and 0 <= args[1] <= 60
    )


# 4. Count distinct paths in a grid (recursive / memoised)
_PATHS_REF = '''
def grid_paths(rows, cols):
    if rows <= 0 or cols <= 0:
        return 0
    table = [[1] * cols for _ in range(rows)]
    for r in range(1, rows):
        for c in range(1, cols):
            table[r][c] = table[r - 1][c] + table[r][c - 1]
    return table[rows - 1][cols - 1]
'''.strip()


def _sample_grid(rng: Random) -> Tuple[Any, ...]:
    return (rng.randint(0, 9), rng.randint(0, 9))


def _valid_grid(args: Tuple[Any, ...]) -> bool:
    return (
        len(args) == 2
        and all(isinstance(x, int) and 0 <= x <= 14 for x in args)
    )


# 5. two sum on a sorted array (two pointers)
# phrased as a decision problem on purpose: "return a pair of indices" has
# several correct answers with duplicates and the differential tester would
# flag a correct submission. Unique output is needed for sound diagnosis.
_TWO_SUM_REF = '''
def has_pair(a, target):
    lo, hi = 0, len(a) - 1
    while lo < hi:
        total = a[lo] + a[hi]
        if total == target:
            return True
        if total < target:
            lo += 1
        else:
            hi -= 1
    return False
'''.strip()


# 6. Maximum number of non-overlapping intervals (greedy)
_INTERVALS_REF = '''
def max_intervals(intervals):
    if not intervals:
        return 0
    ordered = sorted(intervals, key=lambda p: p[1])
    count = 0
    last_end = None
    for start, end in ordered:
        if last_end is None or start >= last_end:
            count += 1
            last_end = end
    return count
'''.strip()


def _sample_intervals(rng: Random) -> Tuple[Any, ...]:
    n = rng.randint(0, 6)
    intervals = []
    for _ in range(n):
        start = rng.randint(0, 9)
        intervals.append([start, start + rng.randint(1, 5)])
    return (intervals,)


def _valid_intervals(args: Tuple[Any, ...]) -> bool:
    return (
        len(args) == 1
        and isinstance(args[0], list)
        and all(
            isinstance(p, list) and len(p) == 2
            and isinstance(p[0], int) and isinstance(p[1], int) and p[0] < p[1]
            for p in args[0]
        )
    )


# problem specs
_PROBLEMS: Tuple[ProblemSpec, ...] = (
    ProblemSpec(
        pid="lower_bound",
        title="Lower bound in a sorted array",
        statement=(
            "Given a list `a` sorted in non-decreasing order and a value `t`, return the index of "
            "the first element that is not less than `t`. If every element is smaller than `t`, "
            "return len(a). The solution must run in O(log n)."
        ),
        entry_point="lower_bound",
        parameters=("a", "t"),
        concepts=("binary-search", "half-open-intervals", "boundary-conditions", "lower-upper-bound"),
        reference_solution=_LOWER_BOUND_REF,
        tests=(
            TestCase(([], 3), 0, "empty array"),
            TestCase(([5], 5), 0, "single element, equal"),
            TestCase(([5], 6), 1, "single element, target larger"),
            TestCase(([1, 3, 5, 7], 0), 0, "before the first"),
            TestCase(([1, 3, 5, 7], 8), 4, "past the last"),
            TestCase(([1, 3, 3, 3, 5], 3), 1, "duplicates"),
            TestCase(([1, 3, 5, 7], 6), 3, "between elements"),
        ),
        sampler=_sample_sorted,
        validator=_valid_sorted,
        difficulty=0.0,
        discrimination=1.5,
        tags=("searching", "binary-search"),
    ),
    ProblemSpec(
        pid="hop_counts",
        title="Shortest hop counts from a source",
        statement=(
            "Given a directed graph as an adjacency list `adj` and a `source` vertex, return a "
            "list whose i-th entry is the minimum number of edges from `source` to vertex i, or "
            "-1 if i is unreachable. The solution must run in O(V + E)."
        ),
        entry_point="hop_counts",
        parameters=("adj", "source"),
        concepts=("bfs", "visited-set", "queue-deque", "graph-traversal", "shortest-path-unweighted"),
        reference_solution=_BFS_REF,
        tests=(
            TestCase(([[]], 0), [0], "single vertex"),
            TestCase(([[1], [2], []], 0), [0, 1, 2], "path"),
            TestCase(([[1, 2], [2], []], 0), [0, 1, 1], "diamond shortcut"),
            TestCase(([[1], [0], []], 0), [0, 1, -1], "unreachable vertex"),
            TestCase(([[1], [2], [0]], 1), [2, 0, 1], "cycle"),
        ),
        sampler=_sample_graph,
        validator=_valid_graph,
        difficulty=0.4,
        discrimination=1.4,
        tags=("graphs", "bfs"),
    ),
    ProblemSpec(
        pid="min_coins",
        title="Minimum number of coins",
        statement=(
            "Given coin denominations `coins` (each usable any number of times) and a target "
            "`amount`, return the fewest coins that sum to exactly `amount`, or -1 if it cannot "
            "be done. The solution must run in O(amount * len(coins))."
        ),
        entry_point="min_coins",
        parameters=("coins", "amount"),
        concepts=("tabulation", "state-design", "dp-transition-order", "optimal-substructure"),
        reference_solution=_COINS_REF,
        tests=(
            TestCase(([1, 2, 5], 11), 3, "classic"),
            TestCase(([2], 3), -1, "impossible"),
            TestCase(([1], 0), 0, "zero amount"),
            TestCase(([3, 7], 21), 3, "multiples"),
            TestCase(([5, 6], 11), 2, "greedy trap"),
        ),
        sampler=_sample_coins,
        validator=_valid_coins,
        difficulty=0.9,
        discrimination=1.3,
        tags=("dynamic-programming",),
    ),
    ProblemSpec(
        pid="grid_paths",
        title="Distinct monotone paths in a grid",
        statement=(
            "Count the paths from the top-left to the bottom-right cell of a `rows` x `cols` grid "
            "moving only right or down. Return 0 if either dimension is zero. The solution must "
            "run in time polynomial in rows and cols."
        ),
        entry_point="grid_paths",
        parameters=("rows", "cols"),
        concepts=("overlapping-subproblems", "memoisation", "recursive-decomposition", "base-case"),
        reference_solution=_PATHS_REF,
        tests=(
            TestCase((1, 1), 1, "single cell"),
            TestCase((0, 5), 0, "empty grid"),
            TestCase((2, 2), 2, "two by two"),
            TestCase((3, 3), 6, "three by three"),
            TestCase((4, 5), 35, "rectangle"),
        ),
        sampler=_sample_grid,
        validator=_valid_grid,
        difficulty=0.7,
        discrimination=1.2,
        tags=("dynamic-programming", "recursion"),
    ),
    ProblemSpec(
        pid="two_sum_sorted",
        title="Two-sum on a sorted array",
        statement=(
            "Given a list `a` sorted in non-decreasing order and a `target`, return True if there "
            "are two *distinct* positions i < j with a[i] + a[j] == target, and False otherwise. "
            "The solution must run in O(n) time and O(1) extra space."
        ),
        entry_point="has_pair",
        parameters=("a", "target"),
        concepts=("loop-invariant", "ordering-relations", "boundary-conditions", "cost-model"),
        reference_solution=_TWO_SUM_REF,
        tests=(
            TestCase(([], 4), False, "empty"),
            TestCase(([3], 6), False, "one element cannot pair with itself"),
            TestCase(([1, 2, 3, 4], 5), True, "outer pair"),
            TestCase(([1, 2, 3, 4], 7), True, "inner pair"),
            TestCase(([1, 2, 3, 4], 100), False, "no pair"),
            TestCase(([2, 2], 4), True, "duplicates"),
        ),
        sampler=_sample_sorted,
        validator=_valid_sorted,
        difficulty=0.2,
        discrimination=1.3,
        tags=("arrays", "two-pointers"),
    ),
    ProblemSpec(
        pid="max_intervals",
        title="Maximum non-overlapping intervals",
        statement=(
            "Given a list of `[start, end]` intervals, return the largest number of them that can "
            "be chosen so that no two overlap. Intervals touching at an endpoint do not overlap."
        ),
        entry_point="max_intervals",
        parameters=("intervals",),
        concepts=("greedy-choice-property", "greedy-exchange-argument", "interval-scheduling", "comparison-sorting"),
        reference_solution=_INTERVALS_REF,
        tests=(
            TestCase(([],), 0, "empty"),
            TestCase(([[0, 2]],), 1, "single"),
            TestCase(([[0, 3], [1, 2], [2, 4]],), 2, "shortest-first trap"),
            TestCase(([[0, 1], [1, 2], [2, 3]],), 3, "touching endpoints"),
            TestCase(([[0, 9], [1, 2], [3, 4], [5, 6]],), 3, "one long interval"),
        ),
        sampler=_sample_intervals,
        validator=_valid_intervals,
        difficulty=1.0,
        discrimination=1.2,
        tags=("greedy",),
    ),
)

PROBLEMS: Dict[str, ProblemSpec] = {spec.pid: spec for spec in _PROBLEMS}


def problem(pid: str) -> ProblemSpec:
    """Look up a problem id in the dev bank, then in the held-out bank."""
    if pid in PROBLEMS:
        return PROBLEMS[pid]
    from .heldout import HELDOUT_PROBLEMS

    if pid in HELDOUT_PROBLEMS:
        return HELDOUT_PROBLEMS[pid]
    known = ", ".join(sorted(PROBLEMS) + sorted(HELDOUT_PROBLEMS))
    raise KeyError(f"unknown problem {pid!r}; known problems: {known}")


def all_problems() -> List[ProblemSpec]:
    return list(_PROBLEMS)
