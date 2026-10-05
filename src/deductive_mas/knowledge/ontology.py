"""The concept ontology: 60ish concepts in ten strata, ordered by prerequisite.
Deliberately fine grained where students actually get stuck (interval
conventions, loop invariants, overlapping subproblems vs optimal
substructure) because the diagnosis cannot be finer than the ontology.

Row = (id, name, stratum, difficulty, prerequisites, summary). difficulty is
an IRT b in logits, a median undergrad is around theta = 0.
"""

from typing import List, Tuple

from ..domain import Concept
from .graph import KnowledgeGraph

_ROWS: Tuple[Tuple[str, str, str, float, Tuple[str, ...], str], ...] = (
    # foundations
    ("var-binding", "Variables and assignment", "foundations", -2.6, (),
     "A name is bound to a value; assignment rebinds the name, it does not copy the value."),
    ("boolean-logic", "Boolean logic", "foundations", -2.3, ("var-binding",),
     "Truth tables, short-circuit evaluation, and the negation of compound conditions."),
    ("sequence-indexing", "Zero-based indexing", "foundations", -2.2, ("var-binding",),
     "Valid indices of a length-n sequence are 0..n-1; index n is one past the end."),
    ("mutability-aliasing", "Mutability and aliasing", "foundations", -1.4, ("var-binding",),
     "Two names can refer to one mutable object, so a write through either is visible through both."),
    ("integer-arithmetic", "Integer versus real division", "foundations", -1.9, ("var-binding",),
     "Index arithmetic must stay integral: // floors, / produces a float that cannot index."),
    ("ordering-relations", "Ordering relations", "foundations", -1.7, ("boolean-logic",),
     "Total orders, antisymmetry, and the meaning of a comparator that is consistent."),
    ("proof-by-contradiction", "Proof by contradiction", "foundations", 0.4, ("boolean-logic",),
     "Assume the negation of the claim and derive an impossibility."),

    # control flow
    ("conditionals", "Conditional branching", "control-flow", -2.0, ("boolean-logic",),
     "Mutually exclusive branches, and the difference between chained elif and separate ifs."),
    ("iteration", "Iteration", "control-flow", -1.8, ("conditionals",),
     "A loop repeats a body while its guard holds; each iteration must make progress."),
    ("loop-invariant", "Loop invariants", "control-flow", -0.2, ("iteration",),
     "A predicate true before the loop, preserved by every iteration, and strong enough at exit."),
    ("loop-termination", "Loop termination", "control-flow", -0.6, ("loop-invariant",),
     "A decreasing, well-founded measure guarantees the loop ends."),
    ("half-open-intervals", "Half-open intervals", "control-flow", -0.1, ("sequence-indexing", "loop-invariant"),
     "The convention [lo, hi) makes length hi-lo, concatenation seamless, and empty the base case."),
    ("boundary-conditions", "Boundary conditions", "control-flow", 0.1, ("half-open-intervals", "loop-termination"),
     "The first, last and empty cases are where interval conventions are actually tested."),
    ("iteration-mutation-safety", "Mutation during iteration", "control-flow", -0.4, ("iteration", "mutability-aliasing"),
     "Adding to or removing from a container while iterating it invalidates the traversal."),

    # complexity
    ("asymptotic-notation", "Asymptotic notation", "complexity", -0.9, ("iteration",),
     "O, Omega and Theta describe growth rate as input size increases, ignoring constants."),
    ("cost-model", "Cost model of built-in operations", "complexity", -0.3, ("asymptotic-notation", "array-list"),
     "Indexing is O(1), membership in a list is O(n), membership in a set is O(1) expected."),
    ("amortised-analysis", "Amortised analysis", "complexity", 0.8, ("cost-model",),
     "The average cost per operation over a worst-case sequence, e.g. dynamic array growth."),
    ("recurrence-relations", "Recurrence relations", "complexity", 0.5, ("asymptotic-notation", "recursion-basics"),
     "Expressing a recursive algorithm's cost as T(n) in terms of smaller instances."),
    ("master-theorem", "The master theorem", "complexity", 0.9, ("recurrence-relations",),
     "Solving T(n) = a T(n/b) + f(n) by comparing f(n) with n^(log_b a)."),

    # data structures
    ("array-list", "Dynamic arrays", "data-structures", -1.6, ("sequence-indexing",),
     "Contiguous storage: O(1) random access and append, O(n) insertion or deletion at the front."),
    ("hash-table", "Hash tables", "data-structures", -0.8, ("array-list",),
     "Expected O(1) insertion and membership by hashing keys into buckets."),
    ("linked-structures", "Linked structures", "data-structures", -0.7, ("mutability-aliasing",),
     "Nodes connected by references: O(1) splicing, O(n) access."),
    ("stack", "Stacks", "data-structures", -1.1, ("array-list",),
     "Last in, first out; the discipline behind depth-first exploration and expression parsing."),
    ("queue-deque", "Queues and deques", "data-structures", -0.9, ("linked-structures", "array-list"),
     "First in, first out with O(1) operations at both ends — a list is not a queue."),
    ("tree-structure", "Trees", "data-structures", -0.5, ("linked-structures",),
     "Acyclic connected structures with a root; depth, height and subtree decomposition."),
    ("complete-binary-tree", "Complete binary trees", "data-structures", 0.2, ("tree-structure",),
     "Every level full except possibly the last: the shape that lets an array encode a tree."),
    ("binary-heap", "Binary heaps", "data-structures", 0.6, ("array-list", "complete-binary-tree"),
     "A partially ordered complete tree giving O(log n) insertion and minimum extraction."),
    ("binary-search-tree", "Binary search trees", "data-structures", 0.5, ("tree-structure", "ordering-relations"),
     "The in-order traversal of a BST is sorted; that invariant is what makes lookup logarithmic."),

    # recursion
    ("recursion-basics", "Recursion", "recursion", -0.7, ("conditionals",),
     "A function defined in terms of itself on a strictly smaller instance."),
    ("base-case", "Base cases", "recursion", -0.5, ("recursion-basics",),
     "Every recursive chain must reach a case that returns without recursing."),
    ("call-stack", "The call stack", "recursion", -0.2, ("recursion-basics",),
     "Each call frame holds its own locals; depth is bounded, and unbounded depth overflows."),
    ("recursive-decomposition", "Recursive decomposition", "recursion", 0.1, ("base-case",),
     "Choosing subproblems whose solutions compose into the solution of the whole."),
    ("tail-vs-tree-recursion", "Tail versus tree recursion", "recursion", 0.6, ("recursive-decomposition",),
     "One self-call per branch unrolls to a loop; several multiply into a tree of calls."),
    ("divide-and-conquer", "Divide and conquer", "recursion", 0.7, ("recursive-decomposition", "master-theorem"),
     "Split into independent subproblems of a constant fraction of the size, then combine."),

    # searching
    ("linear-search", "Linear search", "searching", -1.5, ("iteration",),
     "Scan until found; the only option when the data has no exploitable order."),
    ("monotone-predicate", "Monotone predicates", "searching", 0.3, ("ordering-relations", "boolean-logic"),
     "A predicate false ... false true ... true along an axis is what binary search actually needs."),
    ("binary-search", "Binary search", "searching", 0.0, ("half-open-intervals", "ordering-relations", "boundary-conditions"),
     "Halve a search interval while preserving the invariant that the answer stays inside it."),
    ("lower-upper-bound", "Lower and upper bounds", "searching", 0.4, ("binary-search",),
     "The first index not less than (respectively greater than) a key — the useful generalisation."),
    ("binary-search-on-answer", "Binary search on the answer", "searching", 1.1, ("binary-search", "monotone-predicate"),
     "Search the value axis instead of the index axis when feasibility is monotone."),

    # sorting
    ("comparison-sorting", "Comparison sorting", "sorting", -0.6, ("ordering-relations", "iteration"),
     "Any comparison sort needs Omega(n log n) comparisons in the worst case."),
    ("sorting-stability", "Stability", "sorting", 0.3, ("comparison-sorting",),
     "A stable sort preserves the relative order of equal keys, enabling multi-key sorting."),
    ("merge-sort", "Merge sort", "sorting", 0.5, ("divide-and-conquer", "comparison-sorting"),
     "Split in half, sort each half, merge in linear time: T(n) = 2T(n/2) + O(n)."),
    ("quick-sort", "Quicksort", "sorting", 0.7, ("divide-and-conquer", "comparison-sorting"),
     "Partition around a pivot; expected O(n log n), quadratic when the partition is unbalanced."),
    ("counting-sort", "Counting and bucket sort", "sorting", 0.6, ("hash-table", "comparison-sorting"),
     "Beat the comparison bound by exploiting a bounded key range."),

    # graphs
    ("graph-representation", "Graph representation", "graphs", -0.4, ("array-list", "hash-table"),
     "Adjacency lists cost O(V+E) space and make neighbour iteration proportional to degree."),
    ("visited-set", "Visited sets", "graphs", -0.2, ("hash-table",),
     "Marking nodes is what turns an exponential walk into a linear traversal."),
    ("graph-traversal", "Graph traversal", "graphs", 0.2, ("graph-representation", "queue-deque", "stack"),
     "A worklist plus a visited set; the container's discipline decides breadth or depth."),
    ("bfs", "Breadth-first search", "graphs", 0.4, ("graph-traversal", "visited-set"),
     "A FIFO worklist visits vertices in non-decreasing distance from the source."),
    ("dfs", "Depth-first search", "graphs", 0.4, ("graph-traversal", "call-stack", "visited-set"),
     "A LIFO worklist (or the call stack) explores one branch fully before backtracking."),
    ("shortest-path-unweighted", "Unweighted shortest paths", "graphs", 0.6, ("bfs",),
     "BFS layers are exactly the shortest-path distances when every edge costs one."),
    ("dag-properties", "Directed acyclic graphs", "graphs", 0.5, ("graph-representation",),
     "A DAG admits a linear order in which every edge points forward."),
    ("topological-sort", "Topological sorting", "graphs", 0.8, ("dfs", "dag-properties"),
     "Order vertices so dependencies precede dependents; the schedule behind every DP over a DAG."),
    ("union-find", "Disjoint set union", "graphs", 1.0, ("tree-structure", "amortised-analysis"),
     "Near-constant amortised connectivity queries with path compression and union by rank."),
    ("dijkstra", "Dijkstra's algorithm", "graphs", 1.3, ("shortest-path-unweighted", "binary-heap", "greedy-exchange-argument"),
     "Settle vertices in increasing distance; correctness needs non-negative edge weights."),

    # dynamic programming
    ("optimal-substructure", "Optimal substructure", "dynamic-programming", 0.6, ("recursive-decomposition",),
     "An optimal solution is built from optimal solutions of its subproblems."),
    ("overlapping-subproblems", "Overlapping subproblems", "dynamic-programming", 0.7, ("recursive-decomposition", "hash-table"),
     "The same subproblem recurs exponentially often; that redundancy is what caching removes."),
    ("memoisation", "Memoisation", "dynamic-programming", 0.8, ("overlapping-subproblems",),
     "Cache each state's answer on first computation: cost becomes states times work per state."),
    ("state-design", "State design", "dynamic-programming", 1.0, ("optimal-substructure",),
     "Choosing the minimal state that makes the transition well defined and the count polynomial."),
    ("tabulation", "Tabulation", "dynamic-programming", 0.9, ("memoisation", "iteration"),
     "Fill states bottom-up, which requires an order consistent with the dependency DAG."),
    ("dp-transition-order", "Transition order", "dynamic-programming", 1.2, ("tabulation", "dag-properties"),
     "A state may only be read after every state it depends on has been written."),
    ("knapsack", "Knapsack", "dynamic-programming", 1.3, ("state-design", "tabulation"),
     "Capacity as a state dimension; the 0/1 and unbounded variants differ only in loop order."),
    ("lcs", "Longest common subsequence", "dynamic-programming", 1.2, ("state-design", "tabulation"),
     "A two-dimensional table over prefix pairs with a three-way transition."),

    # greedy
    ("greedy-choice-property", "Greedy choice property", "greedy", 0.9, ("optimal-substructure",),
     "A locally optimal choice is safe only if some optimal solution contains it."),
    ("greedy-exchange-argument", "Exchange arguments", "greedy", 1.2, ("greedy-choice-property", "proof-by-contradiction"),
     "Transform any optimal solution into the greedy one without making it worse."),
    ("interval-scheduling", "Interval scheduling", "greedy", 1.0, ("greedy-exchange-argument", "comparison-sorting"),
     "Sorting by earliest finishing time is optimal; sorting by shortest duration is not."),
)


def build_concepts() -> List[Concept]:
    """Turn the rows into Concept objects."""
    return [
        Concept(
            cid=cid,
            name=name,
            stratum=stratum,
            summary=summary,
            prerequisites=prerequisites,
            difficulty=difficulty,
        )
        for cid, name, stratum, difficulty, prerequisites, summary in _ROWS
    ]


_GRAPH: KnowledgeGraph = KnowledgeGraph(build_concepts())


def knowledge_graph() -> KnowledgeGraph:
    """The shared knowledge graph, built once at import."""
    return _GRAPH


CONCEPT_IDS = frozenset(row[0] for row in _ROWS)
