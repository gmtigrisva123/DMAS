import pytest

from deductive_mas.analysis.belief import MassFunction, fuse
from deductive_mas.analysis.leakage import (
    LeakageDetector,
    containment,
    extract_code_spans,
    fingerprint,
    kgram_hashes,
    winnow,
)

REFERENCE = (
    "def lower_bound(a, t):\n"
    "    lo, hi = 0, len(a)\n"
    "    while lo < hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if a[mid] < t:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid\n"
    "    return lo\n"
)
STUDENT = REFERENCE.replace("lo, hi = 0, len(a)", "lo, hi = 0, len(a) - 1")


@pytest.fixture(scope="module")
def gate():
    return LeakageDetector().build(REFERENCE, STUDENT)


class TestWinnowing:
    def test_fingerprint_is_sparse_but_non_empty(self):
        marks = fingerprint(REFERENCE)
        assert 0 < len(marks) < len(kgram_hashes(list("abcdefghijklmnop"), 5)) + 40

    def test_identical_text_has_full_containment(self):
        marks = fingerprint(REFERENCE)
        assert containment(marks, marks) == 1.0

    def test_containment_of_an_empty_query_is_zero(self):
        assert containment(set(), {1, 2}) == 0.0

    def test_winnow_of_nothing_is_nothing(self):
        assert winnow([], 4) == set()

    def test_shared_passages_are_detected(self):
        shared = "def helper(x):\n    return x * 2 + 1\n"
        left = fingerprint("def a():\n    pass\n" + shared)
        right = fingerprint(shared + "def b():\n    pass\n")
        assert left & right


class TestCodeExtraction:
    def test_finds_fenced_blocks(self):
        spans = extract_code_spans("Try this:\n```python\nx = 1\n```\n")
        assert spans and "x = 1" in spans[0]

    def test_finds_inline_spans(self):
        assert "hi = mid" in extract_code_spans("Consider `hi = mid` here.")

    def test_finds_bare_assignment_lines(self):
        assert any("hi = len(a)" in span for span in extract_code_spans("hi = len(a)\n"))

    def test_prose_yields_nothing(self):
        assert extract_code_spans("What does your upper bound mean?") == []


class TestLeakageGate:
    @pytest.mark.parametrize(
        "guidance",
        [
            "On the input [0] with target 1, walk through your loop by hand.",
            "You initialise with `hi = len(a) - 1`. Which position can that never reach?",
            "You wrote `lo, hi = 0, len(a) - 1`. What range of indices can hi take?",
            "How many distinct answers can the function return for an array of length n?",
            "Trace your loop on [5] and write down lo and hi after each iteration.",
        ],
    )
    def test_socratic_guidance_passes(self, gate, guidance):
        assert not gate.score(guidance).rejected(0.18)

    @pytest.mark.parametrize(
        "guidance",
        [
            "Change the initialisation to `hi = len(a)` and it will work.",
            "Use `lo, hi = 0, len(a)` instead.",
            "```python\n" + REFERENCE + "```",
        ],
    )
    def test_answer_disclosure_is_rejected(self, gate, guidance):
        report = gate.score(guidance)
        assert report.rejected(0.18)
        assert report.reasons

    def test_quoting_the_student_is_not_leakage(self, gate):
        report = gate.score("You wrote `hi = len(a) - 1`; what does that assume?")
        assert report.score == 0.0

    def test_structural_reproduction_is_flagged(self, gate):
        assert gate.score("Add `hi = mid` to the else branch.").structural_match is False
        other = LeakageDetector().build(REFERENCE, REFERENCE.replace("hi = mid", "hi = mid - 1"))
        assert other.score("You need `hi = mid` there.").structural_match


class TestBeliefFusion:
    def test_agreeing_sources_reinforce(self):
        a = MassFunction.from_scores({"x": 0.6}, reliability=0.9)
        b = MassFunction.from_scores({"x": 0.7}, reliability=0.9)
        fused = fuse([("a", a), ("b", b)])
        assert fused.mass.belief("x") > max(a.belief("x"), b.belief("x"))

    def test_conflicting_sources_are_flagged(self):
        a = MassFunction.from_scores({"x": 0.95}, reliability=0.95)
        b = MassFunction.from_scores({"y": 0.95}, reliability=0.95)
        fused = fuse([("a", a), ("b", b)])
        assert fused.contested and fused.max_conflict > 0.5

    def test_a_vacuous_source_is_the_identity(self):
        a = MassFunction.from_scores({"x": 0.5}, reliability=0.8)
        with_empty = fuse([("a", a), ("empty", MassFunction.vacuous())])
        alone = fuse([("a", a)])
        assert with_empty.mass.belief("x") == pytest.approx(alone.mass.belief("x"))

    def test_mass_never_reaches_certainty(self):
        mass = MassFunction.from_scores({"x": 1.0}, reliability=1.0)
        assert mass.belief("x") < 1.0 and mass.uncertainty > 0.0

    def test_plausibility_bounds_belief(self):
        mass = MassFunction.from_scores({"x": 0.4, "y": 0.3}, reliability=0.8)
        assert mass.plausibility("x") >= mass.belief("x")

    def test_refutation_lowers_belief(self):
        support = MassFunction.from_scores({"x": 0.7}, reliability=0.86)
        refute = MassFunction.from_scores({"x": 0.0, "absent": 0.6}, reliability=0.92)
        assert fuse([("s", support), ("r", refute)]).mass.belief("x") < support.belief("x")

    def test_pignistic_transform_redistributes_ignorance(self):
        mass = MassFunction.from_scores({"x": 0.3, "y": 0.3}, reliability=0.6)
        pignistic = mass.pignistic()
        assert sum(pignistic.values()) == pytest.approx(1.0)

    def test_scores_of_zero_produce_a_vacuous_function(self):
        assert MassFunction.from_scores({"x": 0.0}).focal == {}
