"""The grounding corpus: short cards, each tagged with the concepts it grounds
and the kind of claim it makes. Cards are kept terse and checkable (one
fact, one invariant, one cost or one pitfall each) because the grounding
check can only reject an unsupported claim if the corpus is precise.

These are original summaries written for this project, see_also just points
at standard textbooks, it is not a page level citation.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class KnowledgeCard:
    """One citable unit of knowledge."""

    card_id: str
    title: str
    kind: str                      # definition | invariant | cost | pitfall | procedure
    concepts: Tuple[str, ...]
    text: str
    see_also: str = ""

    @property
    def searchable(self) -> str:
        return f"{self.title}. {self.text}"

    def citation(self) -> str:
        return f"[{self.card_id}] {self.title}"


_CARDS: Tuple[KnowledgeCard, ...] = (
    # intervals
    KnowledgeCard(
        "iv-halfopen", "The half-open interval convention", "definition",
        ("half-open-intervals", "sequence-indexing"),
        "Write a range as [lo, hi): lo is included, hi is excluded. Its length is hi - lo, it is "
        "empty exactly when lo == hi, and two adjacent ranges [a, b) and [b, c) tile [a, c) with no "
        "overlap and no gap. Because the empty case is lo == hi rather than lo == hi + 1, loops "
        "written against this convention have the guard `while lo < hi`.",
        "CLRS, Introduction to Algorithms",
    ),
    KnowledgeCard(
        "iv-closed", "The closed interval convention", "definition",
        ("half-open-intervals", "boundary-conditions"),
        "Write a range as [lo, hi] with both ends included. Its length is hi - lo + 1 and it is "
        "empty when lo > hi, so the loop guard is `while lo <= hi`. A closed interval must be "
        "initialised with hi = n - 1, and each side must move strictly past the midpoint "
        "(to mid + 1 or mid - 1) or the interval will not shrink.",
        "CLRS, Introduction to Algorithms",
    ),
    KnowledgeCard(
        "iv-mismatch", "Mixing the two interval conventions", "pitfall",
        ("half-open-intervals", "boundary-conditions", "binary-search"),
        "The two commonest binary-search defects are a guard and an initialiser drawn from "
        "different conventions: `while lo <= hi` with `hi = len(a)` reads one position past the "
        "end, and `while lo < hi` with `hi = len(a) - 1` can never examine the last element. "
        "Choose one convention, write the initialiser, guard and updates from it, and check the "
        "one-element and empty inputs.",
    ),
    KnowledgeCard(
        "iv-empty-boundary", "Boundary inputs are where conventions are tested", "procedure",
        ("boundary-conditions", "loop-invariant"),
        "A convention error is invisible on a typical input and decisive on an extreme one. Test "
        "the empty sequence, the one-element sequence, a target smaller than everything, a target "
        "larger than everything, and a target equal to the first and to the last element.",
    ),

    # loop reasoning
    KnowledgeCard(
        "lp-invariant", "What a loop invariant must satisfy", "definition",
        ("loop-invariant", "iteration"),
        "A loop invariant is a predicate with three properties: initialisation (it holds before "
        "the first iteration), maintenance (if it holds at the start of an iteration it holds at "
        "the start of the next), and termination (when the loop exits, the invariant together "
        "with the negated guard implies the result you want). Proving only the first two proves "
        "nothing about the answer.",
        "CLRS, Introduction to Algorithms",
    ),
    KnowledgeCard(
        "lp-termination", "Termination needs a decreasing measure", "invariant",
        ("loop-termination",),
        "A loop terminates if some non-negative integer quantity strictly decreases on every "
        "iteration. For binary search that measure is hi - lo; for a traversal it is the number "
        "of unvisited vertices. If no guard variable is written in the body, no such measure "
        "exists and the loop can only exit through break or return.",
    ),
    KnowledgeCard(
        "lp-offbyone", "Neighbour access shortens the safe range", "pitfall",
        ("sequence-indexing", "boundary-conditions"),
        "A loop whose body reads a[i + 1] may only run while i + 1 is a valid index, so its bound "
        "is len(a) - 1, not len(a). The general rule: the number of valid windows of width w in a "
        "sequence of length n is n - w + 1.",
    ),
    KnowledgeCard(
        "lp-mutation", "Mutating a container while iterating it", "pitfall",
        ("iteration-mutation-safety", "mutability-aliasing"),
        "A for loop over a list holds a position, not a snapshot. Removing an element shifts every "
        "later element down, so the iterator skips one; appending during iteration can prevent "
        "termination. Iterate over a copy, or build a new container and replace the old one.",
    ),

    # binary search
    KnowledgeCard(
        "bs-core", "Binary search", "definition",
        ("binary-search", "monotone-predicate"),
        "Binary search maintains an interval known to contain the answer and halves it each "
        "iteration by testing the midpoint. It requires only that the predicate being tested is "
        "monotone along the search axis: false ... false true ... true. Sortedness is the special "
        "case where the predicate is 'a[i] >= target'.",
        "CLRS, Introduction to Algorithms; Bentley, Programming Pearls",
    ),
    KnowledgeCard(
        "bs-lowerbound", "Lower bound and upper bound", "definition",
        ("lower-upper-bound", "binary-search"),
        "The lower bound of x is the first index i with a[i] >= x; the upper bound is the first "
        "index with a[i] > x. Both are total functions: they return len(a) when no such index "
        "exists, which is why they are written over the half-open interval [0, len(a)] and why "
        "they never need a separate 'not found' branch.",
    ),
    KnowledgeCard(
        "bs-cost", "Cost of binary search", "cost",
        ("binary-search", "asymptotic-notation"),
        "Each iteration halves the interval, so after k iterations its width is at most n / 2^k "
        "and the loop runs at most ceil(log2(n)) + 1 times: O(log n) comparisons and O(1) extra "
        "space. Linear search over the same data is O(n); the gap is the entire reason sorted "
        "data is worth maintaining.",
    ),
    KnowledgeCard(
        "bs-answer", "Binary search on the answer", "procedure",
        ("binary-search-on-answer", "monotone-predicate"),
        "When a problem asks for the smallest (or largest) value satisfying a feasibility "
        "predicate, and feasibility is monotone in that value, search the value axis instead of "
        "an index axis. The check function replaces the array lookup; everything else about the "
        "interval discipline is unchanged.",
    ),

    # recursion
    KnowledgeCard(
        "rc-basecase", "Base cases", "invariant",
        ("base-case", "recursion-basics"),
        "A recursive definition needs at least one case that returns without recursing, and every "
        "recursive call must move strictly toward it under a well-founded order. Having a base "
        "case is not enough: if the argument can skip past it (decreasing by two from an odd "
        "start, say) the recursion still diverges.",
    ),
    KnowledgeCard(
        "rc-stack", "The call stack bounds recursion depth", "cost",
        ("call-stack", "recursion-basics"),
        "Every pending call holds a frame, so recursion depth d costs O(d) memory and the "
        "interpreter enforces a depth limit. A RecursionError therefore reports a *termination* "
        "defect, not a performance one: depth that grows with n where it should be constant or "
        "logarithmic.",
    ),
    KnowledgeCard(
        "rc-recurrence", "Reading a recurrence off the code", "procedure",
        ("recurrence-relations", "master-theorem"),
        "Count the self-calls (a), see how the argument shrinks (n/b for halving, n-1 for "
        "decrementing) and measure the non-recursive work (f(n)). One call on n-1 gives "
        "T(n) = T(n-1) + f(n) = O(n f(n)); two calls on n-1 give T(n) = 2T(n-1) + f(n), which is "
        "exponential; two calls on n/2 with linear merging give T(n) = 2T(n/2) + O(n) = O(n log n).",
    ),
    KnowledgeCard(
        "rc-master", "The master theorem", "definition",
        ("master-theorem", "divide-and-conquer"),
        "For T(n) = a T(n/b) + f(n) with a >= 1, b > 1, compare f(n) with n^(log_b a). If f is "
        "polynomially smaller, T = Theta(n^(log_b a)). If they match, T = Theta(n^(log_b a) log n). "
        "If f is polynomially larger and satisfies the regularity condition, T = Theta(f(n)).",
        "CLRS, Introduction to Algorithms",
    ),

    # dynamic programming
    KnowledgeCard(
        "dp-two-properties", "The two properties dynamic programming needs", "definition",
        ("optimal-substructure", "overlapping-subproblems"),
        "Dynamic programming applies when a problem has optimal substructure (an optimal solution "
        "contains optimal solutions to subproblems) and overlapping subproblems (the same "
        "subproblem is reached many times). Optimal substructure alone gives divide and conquer; "
        "it is the overlap that makes caching pay.",
        "CLRS, Introduction to Algorithms",
    ),
    KnowledgeCard(
        "dp-memo-cost", "Cost of a memoised recursion", "cost",
        ("memoisation", "overlapping-subproblems"),
        "With a cache, total cost is (number of distinct states) x (work per state), because each "
        "state is computed once and afterwards returned in O(1). Naive Fibonacci makes "
        "Theta(phi^n) calls; memoised, it makes n. The branching factor of the recursion tells "
        "you nothing once the cache is in place.",
    ),
    KnowledgeCard(
        "dp-order", "Transition order in tabulation", "invariant",
        ("dp-transition-order", "tabulation", "dag-properties"),
        "The states of a dynamic program form a directed acyclic graph in which an edge means "
        "'depends on'. Bottom-up filling is valid exactly when the loop order is a topological "
        "order of that DAG: every state must be written before it is read. Reading dp[i][j+1] "
        "while j ascends violates this and consumes an uninitialised value.",
    ),
    KnowledgeCard(
        "dp-state-design", "Choosing the state", "procedure",
        ("state-design", "optimal-substructure"),
        "A state must carry exactly the information the transition needs and nothing more. Too "
        "little and the recurrence is not well defined; too much and the state count explodes. "
        "Write the transition first, then read off which quantities it mentions.",
    ),
    KnowledgeCard(
        "dp-knapsack", "0/1 knapsack", "procedure",
        ("knapsack", "state-design", "tabulation"),
        "State: the best value using the first i items with capacity c. Transition: either skip "
        "item i, or take it and add its value to the best for capacity c - w_i among the first "
        "i-1 items. Iterating capacity downward in a one-dimensional table is what prevents an "
        "item from being taken twice; iterating upward yields the unbounded variant.",
    ),
    KnowledgeCard(
        "dp-lcs", "Longest common subsequence", "procedure",
        ("lcs", "state-design"),
        "State: the LCS length of the first i characters of one string and the first j of the "
        "other. If the characters match the answer is 1 + dp[i-1][j-1]; otherwise it is the "
        "larger of dp[i-1][j] and dp[i][j-1]. The table is (n+1) x (m+1) so that row and column "
        "zero encode the empty-prefix base cases.",
    ),

    # graphs
    KnowledgeCard(
        "gr-representation", "Adjacency lists", "definition",
        ("graph-representation",),
        "An adjacency list stores, for each vertex, the list of its neighbours. Space is "
        "O(V + E), iterating one vertex's neighbours costs its degree, and iterating all of them "
        "over a whole traversal costs O(E) in total, not O(V x E).",
    ),
    KnowledgeCard(
        "gr-bfs", "Breadth-first search", "procedure",
        ("bfs", "graph-traversal", "queue-deque"),
        "Start with the source in a FIFO queue and marked as visited. Repeatedly remove a vertex "
        "and, for each unvisited neighbour, mark it and append it. Marking on *insertion* is what "
        "guarantees each vertex enters the queue exactly once; the traversal is then O(V + E).",
        "CLRS, Introduction to Algorithms",
    ),
    KnowledgeCard(
        "gr-mark-timing", "Mark on insertion, not on removal", "pitfall",
        ("visited-set", "bfs", "graph-traversal"),
        "If a vertex is marked only when it is removed from the worklist, it can be inserted once "
        "per incoming edge before it is ever marked. The queue then holds O(E) entries instead of "
        "O(V), work is repeated, and on dense graphs the traversal degrades badly. The visited "
        "test and the visited mark must happen at the same moment.",
    ),
    KnowledgeCard(
        "gr-bfs-distance", "BFS layers are shortest distances", "invariant",
        ("shortest-path-unweighted", "bfs"),
        "When every edge has the same weight, the order in which BFS first reaches vertices is "
        "non-decreasing in distance from the source, so the distance recorded on first discovery "
        "is optimal and never needs revising. This fails as soon as edge weights differ, which is "
        "why weighted graphs need Dijkstra's priority queue.",
    ),
    KnowledgeCard(
        "gr-dfs", "Depth-first search", "procedure",
        ("dfs", "graph-traversal", "stack"),
        "DFS explores one branch as deeply as possible before backtracking, using an explicit "
        "stack or the call stack. Swapping a queue for a stack in a traversal changes the visit "
        "order from breadth-first to depth-first, and with it every property that depended on "
        "visiting in distance order.",
    ),
    KnowledgeCard(
        "gr-toposort", "Topological sorting", "procedure",
        ("topological-sort", "dag-properties"),
        "A topological order lists the vertices of a DAG so that every edge points forward. Kahn's "
        "algorithm repeatedly removes a vertex of in-degree zero; if fewer than V vertices are "
        "removed, the graph has a cycle. Every bottom-up dynamic program is implicitly a "
        "topological order of its state graph.",
    ),
    KnowledgeCard(
        "gr-dijkstra", "Dijkstra's algorithm", "procedure",
        ("dijkstra", "binary-heap", "greedy-choice-property"),
        "Settle vertices in increasing tentative distance, relaxing outgoing edges as each is "
        "settled. Correctness rests on non-negative weights: with a negative edge, a settled "
        "vertex could still be improved later. With a binary heap the cost is O((V + E) log V).",
    ),

    # data structures
    KnowledgeCard(
        "ds-list-cost", "Cost model of a dynamic array", "cost",
        ("array-list", "cost-model", "amortised-analysis"),
        "Indexing and appending are O(1) (appending amortised, because capacity doubles). "
        "Inserting or deleting at position i is O(n - i), so pop(0) and insert(0, x) are O(n): "
        "they shift every later element. Membership testing with `in` scans, so it is O(n).",
    ),
    KnowledgeCard(
        "ds-set-cost", "Cost model of a hash set", "cost",
        ("hash-table", "cost-model", "visited-set"),
        "Insertion, deletion and membership are O(1) expected for hashable keys, at the price of "
        "losing order and of requiring hashability. Replacing a list by a set for a visited "
        "collection changes a traversal from O(V x E) to O(V + E) without touching its logic.",
    ),
    KnowledgeCard(
        "ds-deque", "Deques", "cost",
        ("queue-deque", "cost-model"),
        "collections.deque supports O(1) append and pop at both ends, which is what a queue needs. "
        "A list used as a queue is correct but quadratic, because every removal from the front "
        "shifts the remaining elements.",
    ),
    KnowledgeCard(
        "ds-heap", "Binary heaps", "cost",
        ("binary-heap", "complete-binary-tree"),
        "A binary heap keeps a complete tree in an array with the parent of index i at (i-1)//2. "
        "Push and pop cost O(log n); reading the minimum costs O(1); building from n items with "
        "heapify costs O(n). It orders only along root-to-leaf paths, so it cannot be iterated in "
        "sorted order without popping.",
    ),
    KnowledgeCard(
        "ds-choice", "Choosing a container", "procedure",
        ("cost-model", "hash-table", "queue-deque"),
        "List the operations the algorithm performs, weight them by how often they run in the "
        "inner loop, and pick the structure that makes the most frequent one cheapest. A "
        "container that is convenient to write is not the same as one that is cheap to run.",
    ),

    # sorting
    KnowledgeCard(
        "so-bound", "The comparison-sorting lower bound", "invariant",
        ("comparison-sorting", "asymptotic-notation"),
        "Any sort that only compares elements needs Omega(n log n) comparisons in the worst case, "
        "because a decision tree with n! leaves has depth at least log2(n!) = Theta(n log n). "
        "Beating the bound requires extra structure, such as a bounded key range.",
        "CLRS, Introduction to Algorithms",
    ),
    KnowledgeCard(
        "so-merge", "Merge sort", "procedure",
        ("merge-sort", "divide-and-conquer"),
        "Split the sequence in half, sort both halves recursively, then merge them in linear time "
        "by repeatedly taking the smaller front element. T(n) = 2T(n/2) + O(n) = O(n log n), and "
        "the merge is stable if ties are broken in favour of the left half.",
    ),
    KnowledgeCard(
        "so-stability", "Stability", "definition",
        ("sorting-stability",),
        "A sort is stable when equal keys keep their original relative order. Stability is what "
        "makes multi-key sorting work by sorting on the least significant key first.",
    ),

    # complexity
    KnowledgeCard(
        "cx-notation", "Asymptotic notation", "definition",
        ("asymptotic-notation",),
        "f(n) = O(g(n)) means f is eventually bounded above by a constant multiple of g. It is an "
        "upper bound, not a description: every algorithm that is O(n) is also O(n^2). Theta is the "
        "tight statement, and it is what 'the complexity is' should normally mean.",
    ),
    KnowledgeCard(
        "cx-composition", "Composing costs", "procedure",
        ("asymptotic-notation", "cost-model"),
        "Sequential blocks take the maximum of their costs; nested loops multiply; a loop "
        "multiplies its trip count by the cost of its body, including the cost of the library "
        "calls in it. Most complexity mistakes come from treating a library call as O(1) without "
        "checking, not from mis-counting loops.",
    ),
    KnowledgeCard(
        "cx-amortised", "Amortised analysis", "definition",
        ("amortised-analysis",),
        "An amortised bound is the average cost per operation over a worst-case sequence, not an "
        "average over random inputs. Dynamic-array append is O(1) amortised because doubling makes "
        "the total cost of n appends O(n), even though one individual append can cost O(n).",
    ),
    KnowledgeCard(
        "cx-hidden", "Where hidden linear costs live", "pitfall",
        ("cost-model", "asymptotic-notation"),
        "Common O(n) operations that look constant: `x in list`, `list.pop(0)`, "
        "`list.insert(0, x)`, slicing, `str` concatenation in a loop, `min`/`max`/`sum` over a "
        "collection, and copying a container. Each of them inside a loop raises the exponent by one.",
    ),

    # greedy
    KnowledgeCard(
        "gd-choice", "The greedy choice property", "definition",
        ("greedy-choice-property", "greedy-exchange-argument"),
        "A greedy algorithm is correct only if some optimal solution begins with its first choice. "
        "The standard proof is an exchange argument: take any optimal solution, swap in the greedy "
        "choice, and show the result is no worse. Without such an argument a greedy rule is a "
        "conjecture, and the way to test it is to hunt for a counterexample.",
        "CLRS, Introduction to Algorithms; Kleinberg & Tardos, Algorithm Design",
    ),
    KnowledgeCard(
        "gd-intervals", "Interval scheduling", "procedure",
        ("interval-scheduling", "greedy-exchange-argument"),
        "To pick the most non-overlapping intervals, repeatedly take the one that finishes "
        "earliest among those still compatible. Sorting by start time or by duration both fail, "
        "and small counterexamples exist for each.",
    ),

    # language
    KnowledgeCard(
        "py-defaults", "Default arguments are evaluated once", "pitfall",
        ("mutability-aliasing", "var-binding"),
        "A default argument expression is evaluated when the function is defined, not when it is "
        "called, so a mutable default is shared by every call and accumulates state across them. "
        "Use None as the default and build the container inside the function.",
    ),
    KnowledgeCard(
        "py-aliasing", "Assignment binds, it does not copy", "pitfall",
        ("mutability-aliasing",),
        "`b = a` makes b another name for the same object. Mutating through either name is visible "
        "through both. Use a slice, `list(a)` or `copy.deepcopy` when independence is required.",
    ),
    KnowledgeCard(
        "py-division", "Integer versus true division", "pitfall",
        ("integer-arithmetic", "sequence-indexing"),
        "`/` always produces a float, which cannot be used as an index; `//` floors toward "
        "negative infinity and stays integral. Midpoints of index ranges must use `//`.",
    ),
    KnowledgeCard(
        "py-identity", "`is` compares identity", "pitfall",
        ("boolean-logic", "var-binding"),
        "`is` asks whether two names refer to the same object; `==` asks whether they have the "
        "same value. Small integers and short strings are often cached, so `is` can appear to work "
        "and then fail on larger values. Use `is` only with None and other singletons.",
    ),
)


def cards() -> List[KnowledgeCard]:
    """The whole corpus."""
    return list(_CARDS)


def cards_for(concept_ids: Tuple[str, ...]) -> List[KnowledgeCard]:
    wanted = set(concept_ids)
    return [card for card in _CARDS if wanted & set(card.concepts)]


CARD_IDS = tuple(card.card_id for card in _CARDS)
