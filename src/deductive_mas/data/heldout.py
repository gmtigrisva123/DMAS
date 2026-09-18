"""Held-out bank: 6 problems, 31 submissions that were not used while writing
the taxonomy or the rules. The protocol matters so it is written here and
not only in a report:

1. the rule set of v1.1.0 was frozen first
2. problems differ from the dev bank in task but use the same curriculum
   (upper bound instead of lower bound, component counting instead of hop
   counting, staircase + 0/1 knapsack instead of coin change + grid paths,
   dedupe + merge instead of two sum + interval scheduling)
3. every flawed submission shows one belief the taxonomy already names and
   its gold label was written down BEFORE running the system on it. Beliefs
   the taxonomy cannot name were left out on purpose, so this measures
   transfer of the detection to new tasks, not taxonomy coverage
4. no rule / threshold / reliability was changed after the first run. Where
   the system misses, the miss is reported.

Labels are our own, not an independent expert annotation. Six correct
submissions are included so the false positive rate is measured on unseen
tasks too.
"""

from random import Random
from typing import Any, Dict, List, Tuple

from ..domain import ProblemSpec, Submission, TestCase

# 1. Upper bound in a sorted array
_UPPER_BOUND_REF = '''
def upper_bound(a, t):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= t:
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


# 2. Connected components of an undirected graph
_COMPONENTS_REF = '''
def count_components(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * n
    count = 0
    for start in range(n):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in adj[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
    return count
'''.strip()


def _sample_components(rng: Random) -> Tuple[Any, ...]:
    n = rng.randint(1, 7)
    edges: List[List[int]] = []
    for _ in range(rng.randint(0, 6)):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            edges.append([u, v])
    return (n, edges)


def _valid_components(args: Tuple[Any, ...]) -> bool:
    if len(args) != 2 or not isinstance(args[0], int) or not isinstance(args[1], list):
        return False
    n, edges = args
    if n < 1:
        return False
    return all(
        isinstance(e, list) and len(e) == 2
        and all(isinstance(x, int) and 0 <= x < n for x in e) and e[0] != e[1]
        for e in edges
    )


# 3. Staircase compositions (ordered ways to climb n stairs)
_CLIMB_REF = '''
def climb_ways(n, steps):
    ways = [0] * (n + 1)
    ways[0] = 1
    for total in range(1, n + 1):
        for step in steps:
            if step <= total:
                ways[total] += ways[total - step]
    return ways[n]
'''.strip()


def _sample_climb(rng: Random) -> Tuple[Any, ...]:
    steps = sorted({rng.randint(1, 4) for _ in range(rng.randint(1, 3))})
    return (rng.randint(0, 12), steps)


def _valid_climb(args: Tuple[Any, ...]) -> bool:
    return (
        len(args) == 2
        and isinstance(args[0], int) and 0 <= args[0] <= 24
        and isinstance(args[1], list) and len(args[1]) > 0
        and all(isinstance(s, int) and s > 0 for s in args[1])
        and len(set(args[1])) == len(args[1])
    )


# 4. De-duplication preserving first occurrences
_DEDUPE_REF = '''
def dedupe(a):
    seen = set()
    out = []
    for x in a:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
'''.strip()


def _sample_dedupe(rng: Random) -> Tuple[Any, ...]:
    n = rng.randint(0, 9)
    # values above the small int cache on purpose, equal values that are
    # different objects is what tells == from is
    pool = [rng.randint(-3, 3) for _ in range(3)] + [rng.randint(300, 305) for _ in range(3)]
    return ([rng.choice(pool) + 0 for _ in range(n)],)


def _valid_dedupe(args: Tuple[Any, ...]) -> bool:
    return len(args) == 1 and isinstance(args[0], list) and all(isinstance(x, int) for x in args[0])


# 5. 0/1 knapsack (maximum value)
_KNAPSACK_REF = '''
def knapsack(weights, values, capacity):
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
    return best[capacity]
'''.strip()


def _sample_knapsack(rng: Random) -> Tuple[Any, ...]:
    n = rng.randint(0, 5)
    weights = [rng.randint(1, 6) for _ in range(n)]
    values = [rng.randint(1, 9) for _ in range(n)]
    return (weights, values, rng.randint(0, 12))


def _valid_knapsack(args: Tuple[Any, ...]) -> bool:
    if len(args) != 3:
        return False
    weights, values, capacity = args
    return (
        isinstance(weights, list) and isinstance(values, list) and len(weights) == len(values)
        and all(isinstance(w, int) and w > 0 for w in weights)
        and all(isinstance(v, int) and v >= 0 for v in values)
        and isinstance(capacity, int) and 0 <= capacity <= 40
    )


# 6. Merging two sorted lists
_MERGE_REF = '''
def merge_sorted(a, b):
    i, j = 0, 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
'''.strip()


def _sample_merge(rng: Random) -> Tuple[Any, ...]:
    a = sorted(rng.randint(-5, 5) for _ in range(rng.randint(0, 6)))
    b = sorted(rng.randint(-5, 5) for _ in range(rng.randint(0, 6)))
    return (a, b)


def _valid_merge(args: Tuple[Any, ...]) -> bool:
    return (
        len(args) == 2
        and all(isinstance(x, list) and list(x) == sorted(x) and all(isinstance(v, int) for v in x)
                for x in args)
    )


_PROBLEMS: Tuple[ProblemSpec, ...] = (
    ProblemSpec(
        pid="upper_bound",
        title="Upper bound in a sorted array",
        statement=(
            "Given a list `a` sorted in non-decreasing order and a value `t`, return the index of "
            "the first element that is strictly greater than `t`. If no element is greater, return "
            "len(a). The solution must run in O(log n)."
        ),
        entry_point="upper_bound",
        parameters=("a", "t"),
        concepts=("binary-search", "half-open-intervals", "boundary-conditions", "lower-upper-bound"),
        reference_solution=_UPPER_BOUND_REF,
        tests=(
            TestCase(([], 3), 0, "empty array"),
            TestCase(([5], 5), 1, "single element, equal"),
            TestCase(([5], 4), 0, "single element, target smaller"),
            TestCase(([1, 3, 3, 3, 5], 3), 4, "duplicates"),
            TestCase(([1, 3, 5, 7], 8), 4, "past the last"),
            TestCase(([1, 3, 5, 7], 0), 0, "before the first"),
            TestCase(([1, 3, 5, 7], 5), 3, "equal to an element"),
        ),
        sampler=_sample_sorted,
        validator=_valid_sorted,
        difficulty=0.1,
        discrimination=1.5,
        tags=("searching", "binary-search"),
    ),
    ProblemSpec(
        pid="count_components",
        title="Connected components",
        statement=(
            "Given `n` vertices numbered 0..n-1 and a list of undirected `edges` (pairs of "
            "vertices), return the number of connected components. The solution must run in "
            "O(n + len(edges))."
        ),
        entry_point="count_components",
        parameters=("n", "edges"),
        concepts=("graph-traversal", "visited-set", "dfs", "graph-representation"),
        reference_solution=_COMPONENTS_REF,
        tests=(
            TestCase((1, []), 1, "single vertex"),
            TestCase((3, []), 3, "no edges"),
            TestCase((3, [[0, 1], [1, 2]]), 1, "path"),
            TestCase((4, [[0, 1], [2, 3]]), 2, "two pairs"),
            TestCase((5, [[0, 1], [1, 2], [2, 0]]), 3, "triangle plus isolated"),
        ),
        sampler=_sample_components,
        validator=_valid_components,
        difficulty=0.4,
        discrimination=1.3,
        tags=("graphs", "dfs"),
    ),
    ProblemSpec(
        pid="climb_ways",
        title="Ways to climb a staircase",
        statement=(
            "Given `n` stairs and a list of distinct allowed step sizes `steps`, return the number "
            "of ordered sequences of steps that climb exactly `n` stairs (there is one way to "
            "climb zero stairs). The solution must run in O(n * len(steps))."
        ),
        entry_point="climb_ways",
        parameters=("n", "steps"),
        concepts=("tabulation", "overlapping-subproblems", "base-case", "recursive-decomposition"),
        reference_solution=_CLIMB_REF,
        tests=(
            TestCase((0, [1, 2]), 1, "zero stairs"),
            TestCase((1, [1, 2]), 1, "one stair"),
            TestCase((4, [1, 2]), 5, "fibonacci"),
            TestCase((3, [2]), 0, "unreachable"),
            TestCase((5, [1, 3]), 4, "two sizes"),
        ),
        sampler=_sample_climb,
        validator=_valid_climb,
        difficulty=0.6,
        discrimination=1.2,
        tags=("dynamic-programming", "recursion"),
    ),
    ProblemSpec(
        pid="dedupe",
        title="Remove duplicates, keeping first occurrences",
        statement=(
            "Given a list `a` of integers, return a new list containing each distinct value once, "
            "in the order of its first occurrence. Do not modify `a`. The solution must run in "
            "O(n) expected time."
        ),
        entry_point="dedupe",
        parameters=("a",),
        concepts=("hash-table", "cost-model", "iteration-mutation-safety", "mutability-aliasing"),
        reference_solution=_DEDUPE_REF,
        tests=(
            TestCase(([],), [], "empty"),
            TestCase(([1, 1, 1],), [1], "all equal"),
            TestCase(([3, 1, 3, 2, 1],), [3, 1, 2], "mixed"),
            TestCase(([300, 300, 301],), [300, 301], "large equal values"),
        ),
        sampler=_sample_dedupe,
        validator=_valid_dedupe,
        difficulty=-0.4,
        discrimination=1.2,
        tags=("hashing", "arrays"),
    ),
    ProblemSpec(
        pid="knapsack",
        title="0/1 knapsack",
        statement=(
            "Given item `weights` and `values` (parallel lists) and a `capacity`, return the largest "
            "total value of a subset of items whose total weight does not exceed `capacity`. Each "
            "item may be used at most once. The solution must run in O(len(weights) * capacity)."
        ),
        entry_point="knapsack",
        parameters=("weights", "values", "capacity"),
        concepts=("knapsack", "state-design", "tabulation", "dp-transition-order"),
        reference_solution=_KNAPSACK_REF,
        tests=(
            TestCase(([], [], 5), 0, "no items"),
            TestCase(([3], [4], 2), 0, "too heavy"),
            TestCase(([1, 3, 4], [15, 20, 30], 4), 35, "classic"),
            TestCase(([2, 2], [3, 3], 4), 6, "both fit"),
            TestCase(([5, 4, 6, 3], [10, 40, 30, 50], 10), 90, "greedy-by-ratio trap"),
        ),
        sampler=_sample_knapsack,
        validator=_valid_knapsack,
        difficulty=1.1,
        discrimination=1.2,
        tags=("dynamic-programming",),
    ),
    ProblemSpec(
        pid="merge_sorted",
        title="Merge two sorted lists",
        statement=(
            "Given two lists `a` and `b`, each sorted in non-decreasing order, return a single "
            "sorted list containing all of their elements. The solution must run in O(len(a) + "
            "len(b))."
        ),
        entry_point="merge_sorted",
        parameters=("a", "b"),
        concepts=("iteration", "loop-invariant", "boundary-conditions", "comparison-sorting"),
        reference_solution=_MERGE_REF,
        tests=(
            TestCase(([], []), [], "both empty"),
            TestCase(([1], []), [1], "one empty"),
            TestCase(([1, 3, 5], [2, 4]), [1, 2, 3, 4, 5], "interleaved"),
            TestCase(([1, 1], [1]), [1, 1, 1], "all equal"),
            TestCase(([5, 6], [1, 2]), [1, 2, 5, 6], "disjoint ranges"),
        ),
        sampler=_sample_merge,
        validator=_valid_merge,
        difficulty=-0.2,
        discrimination=1.3,
        tags=("arrays", "two-pointers"),
    ),
)

HELDOUT_PROBLEMS: Dict[str, ProblemSpec] = {spec.pid: spec for spec in _PROBLEMS}


def _s(sid: str, pid: str, source: str, *gold: str, correct: bool = False) -> Submission:
    return Submission(
        sid=sid,
        problem_id=pid,
        source=source.strip() + "\n",
        gold_misconceptions=gold,
        functionally_correct=correct or not gold,
    )


_SUBMISSIONS: Tuple[Submission, ...] = (
    # upper_bound
    _s(
        "ub_strict_compare", "upper_bound",
        '''
def upper_bound(a, t):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "logic.wrong-comparison-boundary",
    ),
    _s(
        "ub_inclusive_hi", "upper_bound",
        '''
def upper_bound(a, t):
    lo = 0
    hi = len(a) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "bs.interval-convention-mismatch",
    ),
    _s(
        "ub_no_progress", "upper_bound",
        '''
def upper_bound(a, t):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "bs.no-progress-update",
    ),
    _s(
        "ub_linear", "upper_bound",
        '''
def upper_bound(a, t):
    index = 0
    while index < len(a) and a[index] <= t:
        index += 1
    return index
''',
        "search.linear-scan-when-ordered", "cx.asymptotic-gap", correct=True,
    ),
    _s(
        "ub_float_mid", "upper_bound",
        '''
def upper_bound(a, t):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) / 2
        if a[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "py.float-index",
    ),
    _s(
        "ub_correct", "upper_bound",
        '''
def upper_bound(a, t):
    left, right = 0, len(a)
    while left < right:
        centre = left + (right - left) // 2
        if a[centre] > t:
            right = centre
        else:
            left = centre + 1
    return left
''',
    ),

    # count_components
    _s(
        "cc_list_visited", "count_components",
        '''
def count_components(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = []
    count = 0
    for start in range(n):
        if start in visited:
            continue
        count += 1
        stack = [start]
        visited.append(start)
        while stack:
            node = stack.pop()
            for neighbour in adj[node]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    stack.append(neighbour)
    return count
''',
        "graph.list-as-visited", correct=True,
    ),
    _s(
        "cc_edge_scan", "count_components",
        '''
def count_components(n, edges):
    seen = [False] * n
    count = 0
    for start in range(n):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        stack = [start]
        while stack:
            node = stack.pop()
            for u, v in edges:
                if u == node and not seen[v]:
                    seen[v] = True
                    stack.append(v)
                elif v == node and not seen[u]:
                    seen[u] = True
                    stack.append(u)
    return count
''',
        "cx.asymptotic-gap", correct=True,
    ),
    _s(
        "cc_uninitialised", "count_components",
        '''
def count_components(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    count = 0
    for start in range(n):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in adj[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
    return count
''',
        "var.uninitialised",
    ),
    _s(
        "cc_short_table", "count_components",
        '''
def count_components(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * (n - 1)
    count = 0
    for start in range(n):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in adj[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
    return count
''',
        "run.index-out-of-range",
    ),
    _s(
        "cc_correct", "count_components",
        '''
from collections import deque

def count_components(n, edges):
    neighbours = [[] for _ in range(n)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    marked = [False] * n
    components = 0
    for source in range(n):
        if marked[source]:
            continue
        components += 1
        marked[source] = True
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for other in neighbours[current]:
                if not marked[other]:
                    marked[other] = True
                    queue.append(other)
    return components
''',
    ),

    # climb_ways
    _s(
        "cw_naive_recursion", "climb_ways",
        '''
def climb_ways(n, steps):
    if n == 0:
        return 1
    if n < 0:
        return 0
    total = 0
    for step in steps:
        total += climb_ways(n - step, steps)
    return total
''',
        "dp.recomputed-subproblems",
    ),
    _s(
        "cw_no_base_case", "climb_ways",
        '''
def climb_ways(n, steps):
    if n < 0:
        return 0
    total = 0
    for step in steps:
        total += climb_ways(n - step, steps)
    return total
''',
        "rec.missing-base-case",
    ),
    _s(
        "cw_short_table", "climb_ways",
        '''
def climb_ways(n, steps):
    ways = [0] * n
    ways[0] = 1
    for total in range(1, n + 1):
        for step in steps:
            if step <= total:
                ways[total] += ways[total - step]
    return ways[n]
''',
        "run.index-out-of-range",
    ),
    _s(
        "cw_mutable_memo", "climb_ways",
        '''
def climb_ways(n, steps, memo={}):
    if n == 0:
        return 1
    if n < 0:
        return 0
    key = (n, tuple(steps))
    if key in memo:
        return memo[key]
    total = 0
    for step in steps:
        total += climb_ways(n - step, steps, memo)
    memo[key] = total
    return total
''',
        "py.mutable-default-argument", correct=True,
    ),
    _s(
        "cw_loop_order", "climb_ways",
        '''
def climb_ways(n, steps):
    ways = [0] * (n + 1)
    ways[0] = 1
    for step in steps:
        for total in range(step, n + 1):
            ways[total] += ways[total - step]
    return ways[n]
''',
        "dp.transition-order",
    ),
    _s(
        "cw_correct", "climb_ways",
        '''
def climb_ways(n, steps):
    count = [1] + [0] * n
    for height in range(1, n + 1):
        for size in steps:
            if size <= height:
                count[height] += count[height - size]
    return count[n]
''',
    ),

    # dedupe
    _s(
        "dd_membership_scan", "dedupe",
        '''
def dedupe(a):
    result = []
    for x in a:
        if x not in result:
            result.append(x)
    return result
''',
        "cx.membership-in-list", "cx.asymptotic-gap", correct=True,
    ),
    _s(
        "dd_remove_while_iterating", "dedupe",
        '''
def dedupe(a):
    for x in a:
        while a.count(x) > 1:
            a.remove(x)
    return a
''',
        "py.mutate-during-iteration",
    ),
    _s(
        "dd_alias", "dedupe",
        '''
def dedupe(a):
    out = a
    seen = set()
    i = 0
    while i < len(out):
        if out[i] in seen:
            out.pop(i)
        else:
            seen.add(out[i])
            i += 1
    return out
''',
        "py.aliasing-copy", correct=True,
    ),
    _s(
        "dd_identity", "dedupe",
        '''
def dedupe(a):
    out = []
    for x in a:
        found = False
        for y in out:
            if x is y:
                found = True
        if not found:
            out.append(x)
    return out
''',
        "py.identity-vs-equality",
    ),
    _s(
        "dd_correct", "dedupe",
        '''
def dedupe(a):
    already = set()
    kept = []
    for value in a:
        if value in already:
            continue
        already.add(value)
        kept.append(value)
    return kept
''',
    ),

    # knapsack
    _s(
        "ks_ascending_capacity", "knapsack",
        '''
def knapsack(weights, values, capacity):
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(w, capacity + 1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
    return best[capacity]
''',
        "dp.transition-order",
    ),
    _s(
        "ks_greedy_ratio", "knapsack",
        '''
def knapsack(weights, values, capacity):
    order = sorted(range(len(weights)), key=lambda i: values[i] / weights[i], reverse=True)
    total = 0
    room = capacity
    for i in order:
        if weights[i] <= room:
            room -= weights[i]
            total += values[i]
    return total
''',
        "greedy.local-optimum-assumed",
    ),
    _s(
        "ks_exponential", "knapsack",
        '''
def knapsack(weights, values, capacity):
    def best(i, room):
        if i == len(weights):
            return 0
        skip = best(i + 1, room)
        if weights[i] > room:
            return skip
        take = values[i] + best(i + 1, room - weights[i])
        return take if take > skip else skip
    return best(0, capacity)
''',
        "dp.recomputed-subproblems",
    ),
    _s(
        "ks_short_table", "knapsack",
        '''
def knapsack(weights, values, capacity):
    best = [0] * capacity
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
    return best[capacity]
''',
        "run.index-out-of-range",
    ),
    _s(
        "ks_correct", "knapsack",
        '''
def knapsack(weights, values, capacity):
    n = len(weights)
    table = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            table[i][c] = table[i - 1][c]
            if w <= c and table[i - 1][c - w] + v > table[i][c]:
                table[i][c] = table[i - 1][c - w] + v
    return table[n][capacity]
''',
    ),

    # merge_sorted
    _s(
        "mg_dropped_tail", "merge_sorted",
        '''
def merge_sorted(a, b):
    i, j = 0, 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i + 1:])
    out.extend(b[j + 1:])
    return out
''',
        "loop.off-by-one-index",
    ),
    _s(
        "mg_sort_each_step", "merge_sorted",
        '''
def merge_sorted(a, b):
    out = []
    for x in a:
        out.append(x)
        out.sort()
    for x in b:
        out.append(x)
        out.sort()
    return out
''',
        "cx.sort-inside-loop", "cx.asymptotic-gap", correct=True,
    ),
    _s(
        "mg_past_the_end", "merge_sorted",
        '''
def merge_sorted(a, b):
    i, j = 0, 0
    out = []
    while i <= len(a) and j <= len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
''',
        "run.index-out-of-range",
    ),
    _s(
        "mg_correct", "merge_sorted",
        '''
def merge_sorted(a, b):
    merged = []
    p, q = 0, 0
    while p < len(a) or q < len(b):
        if q >= len(b) or (p < len(a) and a[p] <= b[q]):
            merged.append(a[p])
            p += 1
        else:
            merged.append(b[q])
            q += 1
    return merged
''',
    ),
)

HELDOUT_SUBMISSIONS: Dict[str, Submission] = {s.sid: s for s in _SUBMISSIONS}


def heldout_problem(pid: str) -> ProblemSpec:
    try:
        return HELDOUT_PROBLEMS[pid]
    except KeyError as exc:
        known = ", ".join(sorted(HELDOUT_PROBLEMS))
        raise KeyError(f"unknown held-out problem {pid!r}; known: {known}") from exc


def all_heldout_problems() -> List[ProblemSpec]:
    return list(_PROBLEMS)


def all_heldout_submissions() -> List[Submission]:
    return list(_SUBMISSIONS)


def heldout_labelled() -> List[Submission]:
    return [s for s in _SUBMISSIONS if s.gold_misconceptions]
