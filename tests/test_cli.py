import json

import pytest

from deductive_mas.cli.main import build_parser, main

OFFLINE = ["--backend", "offline", "--no-cache", "--color", "never"]


def run(capsys, *args):
    code = main(list(args) + OFFLINE)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_json(capsys, *args):
    code, out, err = run(capsys, *args, "--json")
    assert code == 0, err
    return json.loads(out)


class TestParser:
    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_version_exits_cleanly(self):
        with pytest.raises(SystemExit) as info:
            build_parser().parse_args(["--version"])
        assert info.value.code == 0

    def test_every_command_is_registered(self):
        parser = build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        registered = set()
        for action in actions:
            registered |= set(action.choices)
        assert {
            "diagnose", "list", "kg", "retrieve", "bench", "experiment",
            "robustness", "demo", "doctor",
        } <= registered


class TestDoctor:
    def test_reports_every_component(self, capsys):
        code, out, _ = run(capsys, "doctor")
        assert code == 0
        for component in ("knowledge graph", "misconception taxonomy", "grounding corpus",
                          "problem bank", "reasoning backend"):
            assert component in out
        assert "FAIL" not in out


class TestDiagnose:
    def test_renders_a_full_report(self, capsys):
        code, out, _ = run(capsys, "diagnose", "--submission", "lb_inclusive_bound")
        assert code == 0
        for section in ("Cognitive diagnosis", "Knowledge-graph alignment",
                        "Cognitive intervention", "Provenance"):
            assert section in out
        assert "bs.interval-convention-mismatch" in out

    def test_json_output_is_structured(self, capsys):
        payload = run_json(capsys, "diagnose", "--submission", "lb_no_progress")
        assert payload["diagnosis"]["primary"] == "bs.no-progress-update"
        assert payload["intervention"]["hints"]
        assert payload["intervention"]["leakage"] <= 0.18

    def test_diagnoses_a_file(self, capsys, tmp_path):
        path = tmp_path / "attempt.py"
        path.write_text(
            "def lower_bound(a, t):\n"
            "    lo, hi = 0, len(a) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if a[mid] < t:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid\n"
            "    return lo\n",
            encoding="utf-8",
        )
        payload = run_json(capsys, "diagnose", "--file", str(path), "--problem", "lower_bound")
        assert payload["diagnosis"]["primary"]

    def test_a_file_without_a_problem_is_refused(self, capsys, tmp_path):
        path = tmp_path / "x.py"
        path.write_text("def f():\n    pass\n", encoding="utf-8")
        code, _, err = run(capsys, "diagnose", "--file", str(path))
        assert code == 2 and "--problem" in err

    def test_a_missing_file_is_refused(self, capsys):
        code, _, err = run(capsys, "diagnose", "--file", "/nonexistent.py", "--problem", "lower_bound")
        assert code == 2 and "no such file" in err

    def test_no_target_is_refused(self, capsys):
        code, _, err = run(capsys, "diagnose")
        assert code == 2 and "--submission" in err

    def test_an_unknown_submission_is_refused(self, capsys):
        code, _, err = run(capsys, "diagnose", "--submission", "ghost")
        assert code == 2

    def test_brief_mode_omits_the_evidence_trail(self, capsys):
        _, full, _ = run(capsys, "diagnose", "--submission", "lb_inclusive_bound")
        _, brief, _ = run(capsys, "diagnose", "--submission", "lb_inclusive_bound", "--brief")
        assert len(brief) < len(full)

    def test_no_code_omits_the_listing(self, capsys):
        _, out, _ = run(capsys, "diagnose", "--submission", "lb_inclusive_bound", "--no-code")
        assert "Evidence" not in out


class TestKnowledgeGraphCommands:
    def test_show_renders_a_concept(self, capsys):
        code, out, _ = run(capsys, "kg", "show", "binary-search")
        assert code == 0 and "Binary search" in out and "prerequisites" in out

    def test_show_json(self, capsys):
        payload = run_json(capsys, "kg", "show", "binary-search")
        assert payload["id"] == "binary-search" and payload["prerequisites"]

    def test_an_unknown_concept_is_refused(self, capsys):
        code, _, err = run(capsys, "kg", "show", "phlogiston")
        assert code == 2 and "unknown concept" in err

    def test_path_renders_a_chain(self, capsys):
        payload = run_json(capsys, "kg", "path", "binary-search", "sequence-indexing")
        assert payload["path"][0] == "sequence-indexing"
        assert payload["path"][-1] == "binary-search"

    def test_an_impossible_path_is_refused(self, capsys):
        code, _, err = run(capsys, "kg", "path", "var-binding", "dijkstra")
        assert code == 2 and "no prerequisite path" in err

    def test_strata_summarises_the_curriculum(self, capsys):
        payload = run_json(capsys, "kg", "strata")
        assert len(payload) >= 8


class TestRetrieve:
    def test_returns_ranked_cards(self, capsys):
        payload = run_json(capsys, "retrieve", "mixing interval conventions", "--top", "3")
        assert len(payload) == 3
        assert all({"card", "title", "score", "text"} <= set(row) for row in payload)

    def test_concept_hints_are_accepted(self, capsys):
        payload = run_json(
            capsys, "retrieve", "marking vertices", "--concept", "visited-set", "--top", "4"
        )
        assert any("visited-set" in row["concepts"] for row in payload)

    def test_full_prints_card_bodies(self, capsys):
        _, out, _ = run(capsys, "retrieve", "loop invariant", "--top", "2", "--full")
        assert "initialisation" in out or "invariant" in out


class TestListing:
    def test_lists_both_banks(self, capsys):
        code, out, _ = run(capsys, "list")
        assert code == 0 and "Problem bank" in out and "Submission bank" in out

    def test_json_listing(self, capsys):
        payload = run_json(capsys, "list")
        assert payload["problems"] and payload["submissions"]
        assert any(row["gold"] for row in payload["submissions"])


class TestStudyCommands:
    def test_bench_reports_rq1(self, capsys):
        payload = run_json(capsys, "bench")
        assert payload["top1_accuracy"] == 1.0 and payload["false_positive_rate"] == 0.0

    def test_robustness_reports_rq3(self, capsys):
        payload = run_json(capsys, "robustness")
        assert payload["overall"]["top1_stability"] >= 0.85

    def test_experiment_reports_rq2(self, capsys):
        payload = run_json(capsys, "experiment", "--students", "36", "--sessions", "1")
        assert set(payload["arms"]) >= {"control-direct", "treatment-mas"}
        assert payload["comparisons"] and payload["corrections"]

    def test_experiment_can_drop_the_ablation(self, capsys):
        payload = run_json(
            capsys, "experiment", "--students", "24", "--sessions", "1", "--no-ablation"
        )
        assert "ablation-no-kg" not in payload["arms"]


class TestDemo:
    def test_walks_through_a_session(self, capsys):
        code, out, _ = run(capsys, "demo", "--submissions", "lb_inclusive_bound")
        assert code == 0
        assert "Deductive Multi-Agent System" in out
        assert "Cognitive intervention" in out
