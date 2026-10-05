import pytest

from deductive_mas.data.problems import problem
from deductive_mas.data.submissions import submission
from deductive_mas.knowledge.ontology import knowledge_graph
from deductive_mas.ui.ansi import Style, display_width, strip, supports_colour
from deductive_mas.ui.render import render_concept, render_tutoring_result
from deductive_mas.ui.widgets import (
    Column,
    banner,
    bar,
    histogram,
    key_values,
    pad,
    panel,
    rule,
    source_listing,
    sparkline,
    table,
    tree,
    truncate_styled,
)


class TestStyle:
    def test_colour_can_be_forced_and_suppressed(self):
        assert Style("always").paint("x", "primary") != "x"
        assert Style("never").paint("x", "primary") == "x"

    def test_no_color_env_is_honoured(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert not supports_colour("auto")

    def test_a_dumb_terminal_gets_no_escapes(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        assert not supports_colour("auto")

    def test_display_width_discounts_escapes(self):
        styled = Style("always").paint("hello", "primary")
        assert display_width(styled) == 5 and len(styled) > 5

    def test_strip_removes_escapes(self):
        assert strip(Style("always").paint("abc", "danger")) == "abc"

    def test_belief_and_severity_render(self, plain_style):
        assert plain_style.belief(0.77) == "0.77"
        assert plain_style.severity("critical") == "CRITICAL"


class TestWidgets:
    def test_pad_measures_printable_width(self):
        styled = Style("always").paint("ab", "primary")
        assert display_width(pad(styled, 6)) == 6

    def test_pad_alignments(self, plain_style):
        assert pad("x", 5) == "x    "
        assert pad("x", 5, "right") == "    x"
        assert pad("x", 5, "center") == "  x  "

    def test_truncate_preserves_width(self):
        assert display_width(truncate_styled("abcdefghij", 5)) <= 5

    def test_rule_fills_the_width(self, plain_style):
        assert len(rule(plain_style, width=40)) == 40

    def test_panel_is_a_closed_box(self, plain_style):
        lines = panel(plain_style, ["body"], title="T", width=40)
        assert lines[0].startswith("╭") and lines[-1].startswith("╰")
        assert all(len(strip(line)) == 40 for line in lines)

    def test_table_never_exceeds_its_width(self, plain_style):
        lines = table(
            plain_style,
            [Column("a"), Column("b"), Column("c")],
            [["a very long value indeed " * 3, "second column value", "third"]],
            width=60,
        )
        assert all(len(strip(line)) <= 60 for line in lines)

    def test_table_of_nothing_says_so(self, plain_style):
        assert "no rows" in table(plain_style, [Column("a")], [])[0]

    def test_bar_is_proportional_and_clamped(self, plain_style):
        assert strip(bar(plain_style, 1.0, width=10)).count("█") == 10
        assert strip(bar(plain_style, 0.0, width=10)).count("█") == 0
        assert strip(bar(plain_style, 5.0, width=10)).count("█") == 10

    def test_sparkline_and_histogram_are_compact(self, plain_style):
        assert len(sparkline(plain_style, [1, 5, 3, 9])) == 4
        assert sparkline(plain_style, []) == ""
        assert histogram(plain_style, [1.0] * 8)

    def test_key_values_aligns_keys(self, plain_style):
        lines = key_values(plain_style, [("a", "1"), ("longer", "2")])
        assert all(line.index("1") == lines[1].index("2") for line in lines[:1])

    def test_tree_draws_connectors(self, plain_style, graph):
        lines = tree(
            plain_style, "binary-search", graph.prerequisites,
            label=lambda cid: graph.name(cid), depth=2,
        )
        assert any("├─" in line or "╰─" in line for line in lines)

    def test_source_listing_marks_the_line(self, plain_style):
        lines = source_listing(
            plain_style, "a = 1\nb = 2\nc = 3\n", highlight=2, note="here", indent=""
        )
        joined = "\n".join(lines)
        assert "▶" in joined and "here" in joined

    def test_source_listing_of_nothing(self, plain_style):
        assert source_listing(plain_style, "") == []


class TestReports:
    def test_the_full_report_fits_its_width(self, orchestrator, plain_style):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        lines = render_tutoring_result(plain_style, result, knowledge_graph())
        assert all(len(strip(line)) <= plain_style.width for line in lines)

    @pytest.mark.parametrize("width", [64, 80, 100, 120])
    def test_the_identifier_is_never_elided(self, orchestrator, width):
        style = Style("never")
        style.width = width
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        rendered = "\n".join(render_tutoring_result(style, result, knowledge_graph()))
        assert "bs.interval-convention-mismatch" in rendered
        assert all(len(strip(line)) <= width for line in rendered.splitlines())

    def test_every_section_is_present(self, orchestrator, plain_style):
        result = orchestrator.tutor(problem("hop_counts"), submission("hc_mark_on_dequeue"))
        rendered = "\n".join(render_tutoring_result(plain_style, result, knowledge_graph()))
        for section in ("Evidence", "Cognitive diagnosis", "Knowledge-graph alignment",
                        "Cognitive intervention", "Provenance"):
            assert section in rendered

    def test_a_clean_submission_reports_no_misconception(self, orchestrator, plain_style):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_correct"))
        rendered = "\n".join(render_tutoring_result(plain_style, result, knowledge_graph()))
        assert "No misconception passed the reporting threshold" in rendered

    def test_concept_pages_render(self, plain_style, graph):
        rendered = "\n".join(render_concept(plain_style, graph, "binary-search"))
        assert "Binary search" in rendered and "prerequisites" in rendered

    def test_a_foundation_concept_says_it_has_none(self, plain_style, graph):
        rendered = "\n".join(render_concept(plain_style, graph, "var-binding"))
        assert "foundation concept" in rendered
