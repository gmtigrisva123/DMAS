"""The labelled dev submissions: 29 programs written the way students write
them, including 6 correct ones (otherwise the false positive rate cannot be
measured). gold_misconceptions is the label a human tutor would give. A
submission can have more than one label, the benchmark scores it as multi
label with Cohen's kappa like a second annotator.
"""

from typing import Dict, List, Tuple

from ..domain import Submission


def _s(sid: str, pid: str, source: str, *gold: str, correct: bool = False) -> Submission:
    return Submission(
        sid=sid,
        problem_id=pid,
        source=source.strip() + "\n",
        gold_misconceptions=gold,
        functionally_correct=correct or not gold,
    )


_SUBMISSIONS: Tuple[Submission, ...] = (
    # lower_bound
    _s(
        "lb_inclusive_bound", "lower_bound",
        '''
def lower_bound(a, t):
    lo = 0
    hi = len(a) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "bs.interval-convention-mismatch",
    ),
    _s(
        "lb_closed_guard", "lower_bound",
        '''
def lower_bound(a, t):
    lo, hi = 0, len(a)
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "bs.interval-convention-mismatch", "run.index-out-of-range",
    ),
    _s(
        "lb_no_progress", "lower_bound",
        '''
def lower_bound(a, t):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "bs.no-progress-update",
    ),
    _s(
        "lb_linear_scan", "lower_bound",
        '''
def lower_bound(a, t):
    for i in range(len(a)):
        if a[i] >= t:
            return i
    return len(a)
''',
        "search.linear-scan-when-ordered", "cx.asymptotic-gap", correct=True,
    ),
    _s(
        "lb_float_mid", "lower_bound",
        '''
def lower_bound(a, t):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) / 2
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "py.float-index",
    ),
    _s(
        "lb_correct", "lower_bound",
        '''
def lower_bound(a, t):
    left, right = 0, len(a)
    while left < right:
        middle = left + (right - left) // 2
        if a[middle] < t:
            left = middle + 1
        else:
            right = middle
    return left
''',
    ),

    # hop_counts
    _s(
        "hc_mark_on_dequeue", "hop_counts",
        '''
from collections import deque

def hop_counts(adj, source):
    n = len(adj)
    distance = [-1] * n
    seen = set()
    queue = deque([source])
    distance[source] = 0
    while queue:
        node = queue.popleft()
        seen.add(node)
        for neighbour in adj[node]:
            if neighbour not in seen:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return distance
''',
        "graph.visited-on-dequeue",
    ),
    _s(
        "hc_list_queue", "hop_counts",
        '''
def hop_counts(adj, source):
    n = len(adj)
    distance = [-1] * n
    distance[source] = 0
    queue = [source]
    seen = [source]
    while queue:
        node = queue.pop(0)
        for neighbour in adj[node]:
            if neighbour not in seen:
                seen.append(neighbour)
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return distance
''',
        "graph.list-as-queue", "graph.list-as-visited", correct=True,
    ),
    _s(
        "hc_stack_instead_of_queue", "hop_counts",
        '''
def hop_counts(adj, source):
    n = len(adj)
    distance = [-1] * n
    distance[source] = 0
    stack = [source]
    while stack:
        node = stack.pop()
        for neighbour in adj[node]:
            if distance[neighbour] == -1:
                distance[neighbour] = distance[node] + 1
                stack.append(neighbour)
    return distance
''',
        "ds.wrong-container-choice",
    ),
    _s(
        "hc_uninitialised", "hop_counts",
        '''
from collections import deque

def hop_counts(adj, source):
    n = len(adj)
    distance[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in adj[node]:
            if distance[neighbour] == -1:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return distance
''',
        "var.uninitialised",
    ),
    _s(
        "hc_correct", "hop_counts",
        '''
from collections import deque

def hop_counts(adj, source):
    steps = [-1] * len(adj)
    steps[source] = 0
    frontier = deque()
    frontier.append(source)
    while frontier:
        current = frontier.popleft()
        for nxt in adj[current]:
            if steps[nxt] == -1:
                steps[nxt] = steps[current] + 1
                frontier.append(nxt)
    return steps
''',
    ),

    # min_coins
    _s(
        "mc_greedy", "min_coins",
        '''
def min_coins(coins, amount):
    remaining = amount
    used = 0
    for coin in sorted(coins, reverse=True):
        while remaining >= coin:
            remaining -= coin
            used += 1
    if remaining != 0:
        return -1
    return used
''',
        "greedy.local-optimum-assumed",
    ),
    _s(
        "mc_transition_order", "min_coins",
        '''
def min_coins(coins, amount):
    best = [0] + [amount + 1] * amount
    for value in range(1, amount + 1):
        for coin in coins:
            if value + coin <= amount and best[value + coin] + 1 < best[value]:
                best[value] = best[value + coin] + 1
    return best[amount] if best[amount] <= amount else -1
''',
        "dp.transition-order",
    ),
    _s(
        "mc_table_off_by_one", "min_coins",
        '''
def min_coins(coins, amount):
    best = [0] + [amount + 1] * (amount - 1)
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value and best[value - coin] + 1 < best[value]:
                best[value] = best[value - coin] + 1
    return best[amount] if best[amount] <= amount else -1
''',
        "run.index-out-of-range",
    ),
    _s(
        "mc_exponential", "min_coins",
        '''
def min_coins(coins, amount):
    if amount == 0:
        return 0
    best = -1
    for coin in coins:
        if coin <= amount:
            rest = min_coins(coins, amount - coin)
            if rest >= 0 and (best < 0 or rest + 1 < best):
                best = rest + 1
    return best
''',
        "dp.recomputed-subproblems",
    ),
    _s(
        "mc_correct", "min_coins",
        '''
def min_coins(coins, amount):
    infinity = amount + 1
    table = [0] + [infinity] * amount
    for total in range(1, amount + 1):
        for coin in coins:
            if coin <= total:
                candidate = table[total - coin] + 1
                if candidate < table[total]:
                    table[total] = candidate
    return -1 if table[amount] > amount else table[amount]
''',
    ),

    # grid_paths
    _s(
        "gp_no_base_case", "grid_paths",
        '''
def grid_paths(rows, cols):
    return grid_paths(rows - 1, cols) + grid_paths(rows, cols - 1)
''',
        "rec.missing-base-case",
    ),
    _s(
        "gp_naive_recursion", "grid_paths",
        '''
def grid_paths(rows, cols):
    if rows <= 0 or cols <= 0:
        return 0
    if rows == 1 or cols == 1:
        return 1
    return grid_paths(rows - 1, cols) + grid_paths(rows, cols - 1)
''',
        "dp.recomputed-subproblems",
    ),
    _s(
        "gp_mutable_default", "grid_paths",
        '''
def grid_paths(rows, cols, cache={}):
    if rows <= 0 or cols <= 0:
        return 0
    if rows == 1 or cols == 1:
        return 1
    if (rows, cols) in cache:
        return cache[(rows, cols)]
    cache[(rows, cols)] = grid_paths(rows - 1, cols) + grid_paths(rows, cols - 1)
    return cache[(rows, cols)]
''',
        "py.mutable-default-argument", correct=True,
    ),
    _s(
        "gp_off_by_one", "grid_paths",
        '''
def grid_paths(rows, cols):
    if rows <= 0 or cols <= 0:
        return 0
    table = [[1] * cols for _ in range(rows)]
    for r in range(1, rows):
        for c in range(1, cols):
            table[r][c] = table[r - 1][c] + table[r][c - 1]
    return table[rows][cols]
''',
        "run.index-out-of-range",
    ),
    _s(
        "gp_correct", "grid_paths",
        '''
def grid_paths(rows, cols):
    if rows <= 0 or cols <= 0:
        return 0
    ways = [[1] * cols for _ in range(rows)]
    for row in range(1, rows):
        for column in range(1, cols):
            ways[row][column] = ways[row - 1][column] + ways[row][column - 1]
    return ways[rows - 1][cols - 1]
''',
    ),

    # two_sum_sorted
    _s(
        "ts_quadratic", "two_sum_sorted",
        '''
def has_pair(a, target):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] + a[j] == target:
                return True
    return False
''',
        "cx.asymptotic-gap", correct=True,
    ),
    _s(
        "ts_membership_scan", "two_sum_sorted",
        '''
def has_pair(a, target):
    rest = list(a)
    for i in range(len(a)):
        need = target - a[i]
        if need in rest:
            j = rest.index(need)
            if j != i:
                return True
    return False
''',
        "cx.membership-in-list", "cx.asymptotic-gap", correct=True,
    ),
    _s(
        "ts_wrong_boundary", "two_sum_sorted",
        '''
def has_pair(a, target):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        total = a[lo] + a[hi]
        if total == target:
            return True
        if total < target:
            lo += 1
        else:
            hi -= 1
    return False
''',
        "logic.wrong-comparison-boundary",
    ),
    _s(
        "ts_correct", "two_sum_sorted",
        '''
def has_pair(a, target):
    left = 0
    right = len(a) - 1
    while left < right:
        s = a[left] + a[right]
        if s == target:
            return True
        elif s < target:
            left += 1
        else:
            right -= 1
    return False
''',
    ),

    # max_intervals
    _s(
        "mi_sort_by_start", "max_intervals",
        '''
def max_intervals(intervals):
    if not intervals:
        return 0
    ordered = sorted(intervals, key=lambda p: p[0])
    count = 0
    last_end = None
    for start, end in ordered:
        if last_end is None or start >= last_end:
            count += 1
            last_end = end
    return count
''',
        "greedy.local-optimum-assumed",
    ),
    _s(
        "mi_sort_by_length", "max_intervals",
        '''
def max_intervals(intervals):
    if not intervals:
        return 0
    ordered = sorted(intervals, key=lambda p: p[1] - p[0])
    chosen = []
    for start, end in ordered:
        clash = False
        for other_start, other_end in chosen:
            if start < other_end and other_start < end:
                clash = True
        if not clash:
            chosen.append((start, end))
    return len(chosen)
''',
        "greedy.local-optimum-assumed",
    ),
    _s(
        "mi_mutate_while_iterating", "max_intervals",
        '''
def max_intervals(intervals):
    remaining = sorted(intervals, key=lambda p: p[1])
    count = 0
    last_end = None
    for pair in remaining:
        if last_end is None or pair[0] >= last_end:
            count += 1
            last_end = pair[1]
            remaining.remove(pair)
    return count
''',
        "py.mutate-during-iteration",
    ),
    _s(
        "mi_correct", "max_intervals",
        '''
def max_intervals(intervals):
    if not intervals:
        return 0
    by_finish = sorted(intervals, key=lambda pair: pair[1])
    taken = 0
    boundary = None
    for pair in by_finish:
        if boundary is None or pair[0] >= boundary:
            taken += 1
            boundary = pair[1]
    return taken
''',
    ),
)

SUBMISSIONS: Dict[str, Submission] = {s.sid: s for s in _SUBMISSIONS}


def submission(sid: str) -> Submission:
    try:
        return SUBMISSIONS[sid]
    except KeyError as exc:
        known = ", ".join(sorted(SUBMISSIONS))
        raise KeyError(f"unknown submission {sid!r}; known: {known}") from exc


def all_submissions() -> List[Submission]:
    return list(_SUBMISSIONS)


def labelled() -> List[Submission]:
    """Only submissions with at least one gold label."""
    return [s for s in _SUBMISSIONS if s.gold_misconceptions]


def correct_submissions() -> List[Submission]:
    """The correct submissions, for measuring false positives."""
    return [s for s in _SUBMISSIONS if not s.gold_misconceptions]
