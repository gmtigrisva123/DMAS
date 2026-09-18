"""Dense linear algebra by hand (no numpy). Matrices are List[List[float]],
row major.

- modified_gram_schmidt: stable orthonormalisation
- jacobi_svd: one sided Jacobi SVD (accurate for the small tall matrices
  the range finder produces)
- randomised_svd: Halko-Martinsson-Tropp rank k approximation with power
  iterations, used by LSA
- hungarian: Kuhn-Munkres assignment via shortest augmenting paths with
  potentials, O(n^2 m), used to match variables by behaviour
- dtw_distance: banded dynamic time warping between value traces
"""

import math
import random
from typing import List, Optional, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]

_EPS = 1e-12


# vector helpers
def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in a))


def scale(a: Sequence[float], k: float) -> Vector:
    return [x * k for x in a]


def axpy(k: float, a: Sequence[float], b: Sequence[float]) -> Vector:
    """k * a + b."""
    return [k * x + y for x, y in zip(a, b)]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, 0 when either vector is degenerate."""
    na, nb = norm(a), norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return max(-1.0, min(1.0, dot(a, b) / (na * nb)))


def normalise(a: Sequence[float]) -> Vector:
    n = norm(a)
    return [0.0] * len(a) if n < _EPS else [x / n for x in a]


# matrix helpers
def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(a: Matrix) -> Matrix:
    if not a:
        return []
    return [list(col) for col in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """Triple loop with the inner dimension hoisted (row major friendly)."""
    if not a or not b:
        return []
    inner = len(b)
    cols = len(b[0])
    out = zeros(len(a), cols)
    for i, row in enumerate(a):
        out_row = out[i]
        for k in range(inner):
            aik = row[k]
            if aik == 0.0:
                continue
            brow = b[k]
            for j in range(cols):
                out_row[j] += aik * brow[j]
    return out


def matvec(a: Matrix, x: Sequence[float]) -> Vector:
    return [dot(row, x) for row in a]


def modified_gram_schmidt(a: Matrix) -> Matrix:
    """Orthonormalise the columns of a (same shape back).
    Modified version re-projects against each basis vector in turn, keeps
    orthogonality near machine precision where the classic one falls apart on
    ill conditioned input.
    """
    cols = transpose(a)
    basis: List[Vector] = []
    for col in cols:
        v = list(col)
        for q in basis:
            v = axpy(-dot(q, v), q, v)
        n = norm(v)
        if n < 1e-9:
            v = [0.0] * len(v)
        else:
            v = [x / n for x in v]
        basis.append(v)
    return transpose(basis)


# SVD
def jacobi_svd(a: Matrix, *, max_sweeps: int = 60, tol: float = 1e-11) -> Tuple[Matrix, Vector, Matrix]:
    """One sided Jacobi SVD of a tall matrix a (m x n, m >= n).
    Returns (U, s, V) with a == U @ diag(s) @ V.T, singular values descending.
    Rotates pairs of columns until they are orthogonal, the column norms are
    then the singular values. Slow for big matrices but short and accurate for
    small singular values, which is what the range finder leaves us with.
    """
    if not a:
        return [], [], []
    m, n = len(a), len(a[0])
    if m < n:
        # work on the transpose and swap the factors back
        u, s, v = jacobi_svd(transpose(a), max_sweeps=max_sweeps, tol=tol)
        return v, s, u

    work = [list(row) for row in a]
    v = identity(n)

    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = math.fsum(work[i][p] * work[i][p] for i in range(m))
                beta = math.fsum(work[i][q] * work[i][q] for i in range(m))
                gamma = math.fsum(work[i][p] * work[i][q] for i in range(m))
                if abs(gamma) <= tol * math.sqrt(alpha * beta) or gamma == 0.0:
                    continue
                off += gamma * gamma
                zeta = (beta - alpha) / (2.0 * gamma)
                sign = 1.0 if zeta >= 0.0 else -1.0
                t = sign / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                for i in range(m):
                    xp, xq = work[i][p], work[i][q]
                    work[i][p] = c * xp - s * xq
                    work[i][q] = s * xp + c * xq
                for i in range(n):
                    xp, xq = v[i][p], v[i][q]
                    v[i][p] = c * xp - s * xq
                    v[i][q] = s * xp + c * xq
        if off <= tol:
            break

    sigma = [math.sqrt(math.fsum(work[i][j] * work[i][j] for i in range(m))) for j in range(n)]
    order = sorted(range(n), key=lambda j: -sigma[j])
    s_sorted = [sigma[j] for j in order]
    u = zeros(m, n)
    for rank, j in enumerate(order):
        sj = sigma[j]
        if sj < 1e-12:
            continue
        for i in range(m):
            u[i][rank] = work[i][j] / sj
    v_sorted = [[v[i][j] for j in order] for i in range(n)]
    return u, s_sorted, v_sorted


def randomised_svd(
    a: Matrix,
    k: int,
    *,
    power_iterations: int = 2,
    oversampling: int = 8,
    rng: Optional[random.Random] = None,
) -> Tuple[Matrix, Vector, Matrix]:
    """Rank k truncated SVD via a randomised range finder (Halko, Martinsson &
    Tropp 2011):

    1. Gaussian test matrix Omega, Y = A Omega
    2. q power iterations Y <- A (A^T Y), re-orthonormalised each step
    3. orthonormalise Y -> Q, project B = Q^T A
    4. dense SVD of the small B, lift back through Q

    Within a small factor of the optimal rank k approximation with high
    probability, at a fraction of the cost of a full SVD.
    """
    if not a or k <= 0:
        return [], [], []
    rng = rng or random.Random(0)
    m, n = len(a), len(a[0])
    rank = min(k, m, n)
    sketch_width = min(n, rank + max(0, oversampling))

    omega = [[rng.gauss(0.0, 1.0) for _ in range(sketch_width)] for _ in range(n)]
    y = matmul(a, omega)                         # m x sketch_width
    q = modified_gram_schmidt(y)

    at = transpose(a)
    for _ in range(max(0, power_iterations)):
        z = matmul(at, q)                        # n x sketch_width
        z = modified_gram_schmidt(z)
        y = matmul(a, z)                         # m x sketch_width
        q = modified_gram_schmidt(y)

    b = matmul(transpose(q), a)                  # sketch_width x n

    # decompose the small Gram matrix B B^T (w x w) instead of the tall B^T
    # (n x w). One sided Jacobi on the tall matrix is O(n w^2) per sweep, for a
    # ~580 term vocabulary that is seconds of pure python at every start up,
    # the Gram matrix is O(w^3) per sweep. For symmetric PSD the SVD is the
    # eigendecomposition: singular values = sqrt of eigenvalues, right singular
    # vectors of B = B^T u_i / sigma_i.
    gram = matmul(b, transpose(b))               # w x w, symmetric PSD
    u_small, eigen, _ = jacobi_svd(gram)
    kept = min(rank, len(eigen))
    s: Vector = []
    u_cols: List[Vector] = []
    v_cols: List[Vector] = []
    bt = transpose(b)                            # n x w
    for j in range(kept):
        sigma = math.sqrt(max(0.0, eigen[j]))
        s.append(sigma)
        u_j = [u_small[i][j] for i in range(len(u_small))]
        u_cols.append(u_j)
        if sigma > 1e-12:
            v_cols.append([dot(row, u_j) / sigma for row in bt])
        else:
            v_cols.append([0.0] * len(bt))
    u_full = matmul(q, transpose(u_cols))        # m x kept
    v = transpose(v_cols)                        # n x kept
    return u_full, s, v


# assignment
def hungarian(cost: Matrix) -> List[int]:
    """Min cost assignment (Kuhn-Munkres, shortest augmenting paths).

    cost is n x m with n <= m. Returns a list a of length n, a[i] = column
    matched to row i. Keeps dual potentials u, v with
    cost[i][j] - u[i] - v[j] >= 0 and grows a shortest augmenting path in the
    reduced cost graph. Each of the n phases is O(n*m) so O(n^2 m) total, and
    it is exact (greedy nearest neighbour is not).
    """
    if not cost:
        return []
    n, m = len(cost), len(cost[0])
    if n > m:
        raise ValueError("hungarian requires rows <= columns")

    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = row matched to column j (1-based, 0 = free)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if delta == inf:
                break
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


# sequence comparison
def dtw_distance(a: Sequence[float], b: Sequence[float], *, band: Optional[int] = None) -> float:
    """DTW distance with an optional Sakoe-Chiba band.
    DTW instead of pointwise distance because the student's loop may run more
    or fewer iterations than the reference and still play the same role.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf") if (n or m) else 0.0
    width = band if band is not None else max(n, m)
    width = max(width, abs(n - m))

    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        lo = max(1, i - width)
        hi = min(m, i + width)
        for j in range(lo, hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
        prev[0] = inf
    result = prev[m]
    return result if result != inf else float("inf")


def zscore(values: Sequence[float]) -> List[float]:
    """Standardise a sequence, constant sequences become all zeros."""
    if not values:
        return []
    mean = math.fsum(values) / len(values)
    var = math.fsum((x - mean) ** 2 for x in values) / len(values)
    sd = math.sqrt(var)
    if sd < 1e-9:
        return [0.0] * len(values)
    return [(x - mean) / sd for x in values]
