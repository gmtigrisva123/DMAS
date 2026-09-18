import itertools
import math
import random

import pytest

from deductive_mas.util import linalg as la
from deductive_mas.util.determinism import seeded_rng, stable_hash, stable_unit
from deductive_mas.util.text import (
    code_tokens,
    containment,
    jaccard,
    normalised_levenshtein,
    prose_tokens,
    sentences,
    wrap,
)


class TestText:
    def test_identifiers_are_split_on_case_and_underscore(self):
        assert prose_tokens("lowerBound_idx") == ["lower", "bound", "idx"]

    def test_stemming_is_conservative(self):
        assert prose_tokens("boundaries recursions") == ["boundary", "recursion"]

    def test_spelling_variants_normalise(self):
        assert prose_tokens("memoization") == prose_tokens("memoisation")
        assert prose_tokens("analyze") == prose_tokens("analyse")

    def test_code_tokens_can_erase_identifiers(self):
        tokens = code_tokens("def f(lo, hi):\n    return (lo + hi) // 2\n", normalise_identifiers=True)
        assert "ID" in tokens and "lo" not in tokens
        assert "NUM" in tokens

    def test_code_tokens_survive_a_syntax_error(self):
        assert code_tokens("def f(:\n  x = 1\n")

    def test_levenshtein_bounds(self):
        assert normalised_levenshtein("abc", "abc") == 0.0
        assert normalised_levenshtein("abc", "") == 1.0
        assert 0.0 < normalised_levenshtein("kitten", "sitting") < 1.0

    def test_jaccard_and_containment(self):
        assert jaccard("ab", "ab") == 1.0
        assert jaccard([], []) == 0.0
        assert containment("ab", "abc") == 1.0
        assert containment("abc", "ab") == pytest.approx(2 / 3)

    def test_wrap_never_exceeds_width(self):
        lines = wrap("one two three four five six seven eight", 12, "  ")
        assert all(len(line) <= 12 for line in lines)

    def test_sentences_split_on_terminators(self):
        assert len(sentences("First one. Second one! Third?")) == 3


class TestDeterminism:
    def test_hash_is_stable_across_calls(self):
        assert stable_hash("a", 1) == stable_hash("a", 1)
        assert stable_hash("a", 1) != stable_hash("a", 2)

    def test_unit_interval(self):
        assert all(0.0 <= stable_unit(i) < 1.0 for i in range(200))

    def test_seeded_rng_reproduces(self):
        assert seeded_rng("x").random() == seeded_rng("x").random()


class TestLinearAlgebra:
    def test_jacobi_svd_reconstructs(self):
        rng = random.Random(7)
        a = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(9)]
        u, s, v = la.jacobi_svd(a)
        diag = [[s[i] if i == j else 0.0 for j in range(len(s))] for i in range(len(s))]
        rebuilt = la.matmul(u, la.matmul(diag, la.transpose(v)))
        for i in range(9):
            for j in range(5):
                assert rebuilt[i][j] == pytest.approx(a[i][j], abs=1e-10)

    def test_singular_values_descend(self):
        rng = random.Random(11)
        a = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(6)]
        _, s, _ = la.jacobi_svd(a)
        assert s == sorted(s, reverse=True)

    def test_jacobi_handles_wide_matrices(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        u, s, v = la.jacobi_svd(a)
        assert len(s) == 2 and all(value >= 0 for value in s)

    def test_randomised_svd_approximates_the_leading_spectrum(self):
        rng = random.Random(3)
        a = [[rng.gauss(0, 1) for _ in range(24)] for _ in range(30)]
        _, approx, _ = la.randomised_svd(a, 4, power_iterations=3, rng=random.Random(1))
        _, exact, _ = la.jacobi_svd(a)
        for got, want in zip(approx, exact[:4]):
            assert got == pytest.approx(want, rel=0.05)

    def test_gram_schmidt_orthonormalises(self):
        a = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        q = la.modified_gram_schmidt(a)
        columns = la.transpose(q)
        assert la.dot(columns[0], columns[1]) == pytest.approx(0.0, abs=1e-12)
        assert la.norm(columns[0]) == pytest.approx(1.0)

    @pytest.mark.parametrize("trial", range(40))
    def test_hungarian_matches_brute_force(self, trial):
        rng = random.Random(trial)
        n = rng.randint(1, 4)
        m = rng.randint(n, 5)
        cost = [[float(rng.randint(0, 9)) for _ in range(m)] for _ in range(n)]
        assignment = la.hungarian(cost)
        got = sum(cost[i][assignment[i]] for i in range(n))
        best = min(
            sum(cost[i][p[i]] for i in range(n))
            for p in itertools.permutations(range(m), n)
        )
        assert got == pytest.approx(best)

    def test_hungarian_rejects_more_rows_than_columns(self):
        with pytest.raises(ValueError):
            la.hungarian([[1.0], [2.0], [3.0]])

    def test_dtw_is_zero_for_identical_series(self):
        assert la.dtw_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_dtw_absorbs_a_repeated_sample(self):
        assert la.dtw_distance([1, 2, 3], [1, 1, 2, 3]) == pytest.approx(0.0)

    def test_cosine_is_zero_for_degenerate_vectors(self):
        assert la.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_zscore_of_a_constant_series(self):
        assert la.zscore([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]
