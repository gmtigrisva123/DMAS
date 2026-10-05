"""Stats checked against published table values."""

import math
import random

import pytest

from deductive_mas.stats.agreement import cohen_kappa, krippendorff_alpha, multilabel_report
from deductive_mas.stats.effect import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    effect_with_ci,
    hedges_g,
    interpret_d,
)
from deductive_mas.stats.multiple import benjamini_hochberg, holm_bonferroni
from deductive_mas.stats.power import (
    minimum_detectable_effect,
    plan,
    required_sample_size,
    two_sample_power,
)
from deductive_mas.stats.special import (
    chi2_sf,
    f_sf,
    normal_cdf,
    normal_ppf,
    regularised_incomplete_beta,
    regularised_lower_gamma,
    t_cdf,
    t_ppf,
)
from deductive_mas.stats.tests import (
    chi_square_independence,
    mann_whitney_u,
    mean,
    paired_t_test,
    stdev,
    student_t_test,
    variance,
    welch_t_test,
    wilcoxon_signed_rank,
)


class TestSpecialFunctions:
    @pytest.mark.parametrize(
        "computed,expected",
        [
            (t_cdf(2.0, 10), 0.9633060),
            (t_ppf(0.975, 10), 2.2281389),
            (t_ppf(0.95, 30), 1.6972609),
            (normal_cdf(1.96), 0.9750021),
            (normal_ppf(0.975), 1.9599640),
            (normal_ppf(0.99), 2.3263479),
            (chi2_sf(3.8414588, 1), 0.05),
            (chi2_sf(11.070498, 5), 0.05),
            (f_sf(4.256495, 2, 9), 0.05),
            (regularised_incomplete_beta(2, 3, 0.5), 0.6875),
            (regularised_lower_gamma(3, 2), 0.3233236),
        ],
    )
    def test_matches_published_values(self, computed, expected):
        assert computed == pytest.approx(expected, abs=1e-6)

    def test_t_distribution_is_symmetric(self):
        assert t_cdf(-1.7, 12) == pytest.approx(1.0 - t_cdf(1.7, 12), abs=1e-12)

    def test_t_approaches_the_normal_for_large_df(self):
        assert t_cdf(1.96, 100000) == pytest.approx(normal_cdf(1.96), abs=1e-4)

    def test_ppf_inverts_cdf(self):
        # the normal inverse gets a Halley step against an exact CDF so it is
        # machine precision. The t inverse bisects a continued fraction that is
        # itself only ~1e-8 accurate, so it cannot be held tighter.
        for p in (0.01, 0.25, 0.5, 0.75, 0.99):
            assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-9)
            assert t_cdf(t_ppf(p, 8), 8) == pytest.approx(p, abs=1e-7)

    def test_domain_errors_are_raised(self):
        with pytest.raises(ValueError):
            normal_ppf(0.0)
        with pytest.raises(ValueError):
            t_ppf(1.0, 5)
        with pytest.raises(ValueError):
            t_cdf(1.0, 0)


class TestHypothesisTests:
    A = [7, 8, 9, 6, 7, 8, 9, 10, 8, 7]
    B = [5, 6, 4, 5, 6, 5, 7, 4, 6, 5]

    def test_moments(self):
        assert mean([1, 2, 3]) == 2.0
        assert variance([1, 2, 3]) == pytest.approx(1.0)
        assert stdev([1, 2, 3]) == pytest.approx(1.0)

    def test_a_real_difference_is_detected(self):
        assert welch_t_test(self.A, self.B).significant()
        assert mann_whitney_u(self.A, self.B).significant()

    def test_no_difference_is_not_detected(self):
        rng = random.Random(4)
        x = [rng.gauss(0, 1) for _ in range(200)]
        y = [rng.gauss(0, 1) for _ in range(200)]
        assert not welch_t_test(x, y).significant()

    def test_welch_and_student_agree_on_equal_variances(self):
        welch, student = welch_t_test(self.A, self.B), student_t_test(self.A, self.B)
        assert welch.statistic == pytest.approx(student.statistic, rel=1e-6)

    def test_one_sided_alternatives(self):
        greater = welch_t_test(self.A, self.B, alternative="greater")
        less = welch_t_test(self.A, self.B, alternative="less")
        assert greater.p_value < 0.05 < less.p_value

    def test_unknown_alternative_is_refused(self):
        with pytest.raises(ValueError):
            welch_t_test(self.A, self.B, alternative="sideways")

    def test_tiny_samples_are_refused(self):
        with pytest.raises(ValueError):
            welch_t_test([1.0], [2.0])

    def test_paired_tests_need_equal_lengths(self):
        with pytest.raises(ValueError):
            paired_t_test([1, 2, 3], [1, 2])

    def test_paired_and_signed_rank_agree_in_direction(self):
        assert paired_t_test(self.A, self.B).significant()
        assert wilcoxon_signed_rank(self.A, self.B).significant()

    def test_ties_do_not_break_the_rank_tests(self):
        assert 0.0 <= mann_whitney_u([1, 1, 1], [1, 1, 1]).p_value <= 1.0
        assert wilcoxon_signed_rank([1, 1], [1, 1]).p_value == 1.0

    def test_chi_square_on_a_contingency_table(self):
        result = chi_square_independence([[30, 10], [15, 25]])
        assert result.df == 1 and result.significant()

    def test_chi_square_needs_a_real_table(self):
        with pytest.raises(ValueError):
            chi_square_independence([[1, 2]])

    def test_apa_rendering(self):
        assert "t(" in welch_t_test(self.A, self.B).apa()
        assert "p" in welch_t_test(self.A, self.B).apa()


class TestEffectSizes:
    def test_a_one_sd_shift_gives_d_near_one(self):
        rng = random.Random(9)
        base = [rng.gauss(0, 1) for _ in range(400)]
        shifted = [v + 1.0 for v in base]
        assert cohens_d(shifted, base) == pytest.approx(1.0, abs=0.1)

    def test_hedges_g_corrects_downward(self):
        a, b = [5, 6, 7, 8], [1, 2, 3, 4]
        assert abs(hedges_g(a, b).value) < abs(cohens_d(a, b))

    def test_identical_samples_have_no_effect(self):
        assert cohens_d([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0

    def test_zero_variance_is_handled(self):
        assert cohens_d([2, 2, 2], [2, 2, 2]) == 0.0

    def test_cliffs_delta_is_bounded_and_signed(self):
        assert cliffs_delta([5, 6, 7], [1, 2, 3]).value == pytest.approx(1.0)
        assert cliffs_delta([1, 2, 3], [5, 6, 7]).value == pytest.approx(-1.0)
        assert cliffs_delta([1, 2, 3], [1, 2, 3]).value == pytest.approx(0.0)

    def test_interpretation_bands(self):
        assert interpret_d(0.1) == "negligible"
        assert interpret_d(0.3) == "small"
        assert interpret_d(0.6) == "medium"
        assert interpret_d(1.2) == "large"

    def test_bca_interval_brackets_the_estimate(self):
        rng = random.Random(2)
        a = [rng.gauss(1, 1) for _ in range(60)]
        b = [rng.gauss(0, 1) for _ in range(60)]
        effect = effect_with_ci(a, b, resamples=600, rng=random.Random(1))
        assert effect.ci_low <= effect.value <= effect.ci_high

    def test_bca_excludes_zero_for_a_large_effect(self):
        rng = random.Random(6)
        a = [rng.gauss(2, 1) for _ in range(80)]
        b = [rng.gauss(0, 1) for _ in range(80)]
        effect = effect_with_ci(a, b, resamples=800, rng=random.Random(3))
        assert effect.ci_low > 0.0

    def test_bootstrap_of_a_tiny_sample_degenerates_gracefully(self):
        observed, low, high = bootstrap_ci([1.0], [2.0], lambda x, y: 0.0)
        assert observed == low == high


class TestMultipleComparisons:
    P_VALUES = [("a", 0.001), ("b", 0.013), ("c", 0.04), ("d", 0.31), ("e", 0.60)]

    def test_holm_is_conservative_and_monotone(self):
        results = holm_bonferroni(self.P_VALUES)
        adjusted = [r.adjusted for r in results]
        assert adjusted == sorted(adjusted)
        assert all(r.adjusted >= r.raw for r in results)

    def test_holm_rejects_fewer_than_bh(self):
        holm = sum(1 for r in holm_bonferroni(self.P_VALUES) if r.rejected)
        bh = sum(1 for r in benjamini_hochberg(self.P_VALUES) if r.rejected)
        assert holm <= bh

    def test_bh_adjusted_values_are_monotone(self):
        adjusted = [r.adjusted for r in benjamini_hochberg(self.P_VALUES)]
        assert adjusted == sorted(adjusted)

    def test_empty_input_is_handled(self):
        assert benjamini_hochberg([]) == [] and holm_bonferroni([]) == []


class TestPower:
    def test_matches_the_standard_planning_table(self):
        assert required_sample_size(0.5, 0.80) == 64
        assert required_sample_size(0.8, 0.80) == 26
        assert two_sample_power(0.5, 64) == pytest.approx(0.80, abs=0.01)

    def test_power_increases_with_n_and_effect(self):
        assert two_sample_power(0.5, 100) > two_sample_power(0.5, 30)
        assert two_sample_power(0.9, 40) > two_sample_power(0.3, 40)

    def test_minimum_detectable_effect_is_consistent(self):
        mde = minimum_detectable_effect(60)
        assert two_sample_power(mde, 60) == pytest.approx(0.80, abs=0.02)

    def test_a_zero_effect_is_unreachable(self):
        assert required_sample_size(0.0) >= 100_000

    def test_plan_reports_the_achieved_power(self):
        analysis = plan(0.6)
        assert analysis.power >= 0.80 and analysis.n_per_group > 0


class TestAgreement:
    def test_perfect_and_chance_agreement(self):
        assert cohen_kappa(list("aabbcc"), list("aabbcc")) == 1.0
        assert cohen_kappa(list("aabbcc"), list("bbccaa")) < 0.0

    def test_kappa_of_constant_raters(self):
        assert cohen_kappa(["a"] * 5, ["a"] * 5) == 1.0
        assert cohen_kappa(["a"] * 5, ["b"] * 5) == 0.0

    def test_kappa_requires_aligned_annotations(self):
        with pytest.raises(ValueError):
            cohen_kappa(["a"], ["a", "b"])

    def test_krippendorff_alpha_ranges(self):
        assert krippendorff_alpha([["a", "a"], ["b", "b"], ["c", "c"]]) == pytest.approx(1.0)
        assert krippendorff_alpha([]) == 0.0

    def test_multilabel_report_counts_correctly(self):
        report = multilabel_report(
            [["x"], ["y", "z"], []],
            [["x"], ["y"], ["w"]],
            top_choice=["x", "y", ""],
        )
        assert report.support == 3
        assert report.top1_accuracy == pytest.approx(2 / 3)
        assert 0.0 < report.micro_f1 <= 1.0

    def test_perfect_predictions_score_one(self):
        gold = [["x"], ["y"]]
        report = multilabel_report(gold, gold, top_choice=["x", "y"])
        assert report.micro_f1 == 1.0 and report.exact_match == 1.0 and report.kappa == 1.0

    def test_misaligned_input_is_refused(self):
        with pytest.raises(ValueError):
            multilabel_report([["x"]], [["x"], ["y"]])
