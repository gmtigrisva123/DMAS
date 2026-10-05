import io
from pathlib import Path

import pytest

from deductive_mas.cli.main import main
from deductive_mas.cli.session import TutorSession, freeform_problem
from deductive_mas.ui.ansi import Style

BROKEN = (
    "def lower_bound(a, t):\n"
    "    lo = 0\n"
    "    hi = len(a) - 1\n"
    "    while lo < hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if a[mid] < t:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid\n"
    "    return lo\n"
)
FIXED = BROKEN.replace("hi = len(a) - 1", "hi = len(a)")
HALF = FIXED.replace("while lo < hi", "while lo <= hi")      # fixes one thing, breaks another

UNKNOWN_TASK = (
    "def walk(graph, start):\n"
    "    queue = [start]\n"
    "    seen = []\n"
    "    while queue:\n"
    "        node = queue.pop(0)\n"
    "        for nxt in graph[node]:\n"
    "            if nxt not in seen:\n"
    "                seen.append(nxt)\n"
    "                queue.append(nxt)\n"
    "    return seen\n"
)


class Script:
    """A stdin whose reads can have side effects, so file edits land exactly
    between two commands like an editor would.
    """

    def __init__(self, *items):
        self.items = list(items)

    def isatty(self):
        return False

    def readline(self):
        while self.items:
            item = self.items.pop(0)
            if callable(item):
                item()
                continue
            return item + "\n"
        return ""


def drive(config, *items, width=84):
    style = Style("never")
    style.width = width
    out = io.StringIO()
    session = TutorSession(config, style=style, input_stream=Script(*items), output_stream=out)
    code = session.run()
    return code, out.getvalue(), session


@pytest.fixture()
def attempt(tmp_path) -> Path:
    path = tmp_path / "attempt.py"
    path.write_text(BROKEN, encoding="utf-8")
    return path


class TestBasics:
    def test_welcome_and_clean_exit(self, config):
        code, out, _ = drive(config, "quit")
        assert code == 0
        assert "interactive tutor" in out and "session ended" in out

    def test_eof_ends_the_session(self, config):
        code, out, _ = drive(config)
        assert code == 0 and "session ended" in out

    def test_help_lists_every_command(self, config):
        _, out, session = drive(config, "help", "quit")
        for name in ("load", "hint", "retry", "challenge", "why", "mastery", "progress"):
            assert name in out
        assert len({c.name for c in session.commands.values()}) >= 15

    def test_help_on_one_command(self, config):
        _, out, _ = drive(config, "help hint", "quit")
        assert "hint" in out and "rung" in out

    def test_unknown_command_suggests_a_neighbour(self, config):
        _, out, _ = drive(config, "hnit", "quit")
        assert "unknown command" in out and "hint" in out

    def test_blank_lines_and_comments_are_ignored(self, config):
        code, out, _ = drive(config, "", "   ", "# a comment", "quit")
        assert code == 0 and "unknown" not in out

    def test_commands_before_a_load_are_refused_gracefully(self, config):
        _, out, _ = drive(config, "hint", "why", "retry", "challenge", "code", "quit")
        assert out.count("nothing") >= 4 and "Traceback" not in out


class TestLoading:
    def test_load_detects_the_task_and_diagnoses(self, config, attempt):
        _, out, session = drive(config, f"load {attempt}", "quit")
        assert "task  lower_bound" in out or "lower_bound" in out
        assert "DIAGNOSIS" in out and "Mixed interval conventions" in out
        assert session.state.problem.pid == "lower_bound"
        assert len(session.state.attempts) == 1

    def test_a_bare_path_is_treated_as_load(self, config, attempt):
        _, out, session = drive(config, str(attempt), "quit")
        assert len(session.state.attempts) == 1

    def test_load_with_an_explicit_task(self, config, attempt):
        _, _, session = drive(config, f"load {attempt} lower_bound", "quit")
        assert session.state.problem.pid == "lower_bound"

    def test_missing_file_does_not_end_the_session(self, config, tmp_path):
        _, out, _ = drive(config, f"load {tmp_path / 'nope.py'}", "help", "quit")
        assert "no such file" in out and "session ended" in out

    def test_a_directory_is_refused(self, config, tmp_path):
        _, out, _ = drive(config, f"load {tmp_path}", "quit")
        assert "is a directory" in out

    def test_an_empty_file_is_refused(self, config, tmp_path):
        path = tmp_path / "empty.py"
        path.write_text("", encoding="utf-8")
        _, out, _ = drive(config, f"load {path}", "quit")
        assert "empty" in out and "Traceback" not in out

    def test_a_syntax_error_is_survived(self, config, tmp_path):
        path = tmp_path / "bad.py"
        path.write_text("def lower_bound(a, t:\n    return\n", encoding="utf-8")
        _, out, _ = drive(config, f"load {path}", "quit")
        assert "Traceback" not in out and "session ended" in out

    def test_an_unknown_task_falls_back_to_structural_findings(self, config, tmp_path):
        path = tmp_path / "walk.py"
        path.write_text(UNKNOWN_TASK, encoding="utf-8")
        _, out, session = drive(config, f"load {path}", "hint", "quit")
        assert session.state.problem.pid == "freeform"
        assert "structural findings only" in out
        assert "List used as a queue" in out or "graph.list-as-queue" in out
        assert "hint 1/" in out

    def test_freeform_problem_has_no_oracle(self):
        spec = freeform_problem("solve")
        assert spec.reference_solution == "" and spec.tests == ()


class TestSocraticLadder:
    def test_hints_are_revealed_one_rung_at_a_time(self, config, attempt):
        _, out, session = drive(config, f"load {attempt}", "hint", "quit")
        assert "hint 1/3" in out and "hint 2/3" not in out
        assert session.state.revealed == 1

    def test_the_ladder_runs_out(self, config, attempt):
        _, out, _ = drive(config, f"load {attempt}", "hint", "hint", "hint", "hint", "quit")
        assert "hint 3/3" in out and "last rung" in out

    def test_the_diagnosis_never_prints_the_ladder_unasked(self, config, attempt):
        _, out, _ = drive(config, f"load {attempt}", "quit")
        assert "hint 1/" not in out and "Take the input" not in out

    def test_the_challenge_asks_for_a_prediction(self, config, attempt):
        _, out, session = drive(config, f"load {attempt}", "challenge", "quit")
        assert "Counter-factual challenge" in out and "predict" in out.lower()
        assert session.state.challenge_shown

    def test_no_rung_ever_discloses_the_fix(self, config, attempt):
        _, out, _ = drive(config, f"load {attempt}", "hint", "hint", "hint", "challenge", "quit")
        assert "hi = len(a)\n" not in out and "```" not in out

    def test_why_shows_the_evidence_trail(self, config, attempt):
        _, out, _ = drive(config, f"load {attempt}", "why", "quit")
        assert "Evidence" in out and "static" in out and "root cause" in out

    def test_code_marks_the_divergence(self, config, attempt):
        _, out, _ = drive(config, f"load {attempt}", "code", "quit")
        assert "▶" in out and "smallest failing input" in out

    def test_report_prints_everything_and_syncs_the_ladder(self, config, attempt):
        _, out, session = drive(config, f"load {attempt}", "report", "hint", "quit")
        assert "Provenance" in out
        assert "last rung" in out            # report revealed all rungs


class TestRetry:
    def test_unchanged_file_is_reported(self, config, attempt):
        _, out, session = drive(config, f"load {attempt}", "retry", "quit")
        assert "has not changed" in out and len(session.state.attempts) == 1

    def test_a_fix_is_recognised_and_credited(self, config, attempt):
        _, out, session = drive(
            config,
            f"load {attempt}",
            "hint",
            lambda: attempt.write_text(FIXED, encoding="utf-8"),
            "retry",
            "quit",
        )
        assert "NO FINDING" in out
        assert "resolved  Mixed interval conventions" in out
        assert "→ 87/87" in out or "87/87 inputs pass" in out
        first, second = session.state.attempts
        target = first.result.intervention.target_concept
        assert second.result.mastery_posterior[target] > first.result.mastery_posterior[target]
        assert "↑" in out

    def test_a_partial_fix_is_diagnosed_honestly(self, config, attempt):
        _, out, session = drive(
            config,
            f"load {attempt}",
            lambda: attempt.write_text(HALF, encoding="utf-8"),
            "retry",
            "quit",
        )
        assert "persists" in out or "new" in out
        assert len(session.state.attempts) == 2
        assert session.state.revealed == 0          # new attempt resets the ladder

    def test_progress_lists_every_attempt(self, config, attempt):
        _, out, _ = drive(
            config,
            f"load {attempt}",
            lambda: attempt.write_text(FIXED, encoding="utf-8"),
            "retry",
            "progress",
            "quit",
        )
        assert "bs.interval-convention-mismatch" in out and "clean" in out

    def test_mastery_is_carried_across_attempts(self, config, attempt):
        _, out, session = drive(
            config,
            f"load {attempt}",
            lambda: attempt.write_text(FIXED, encoding="utf-8"),
            "retry",
            "mastery",
            "quit",
        )
        assert session.state.mastery is not None
        assert "evidence about" in out and "Loop invariants" in out

    def test_mastery_before_any_diagnosis(self, config):
        _, out, _ = drive(config, "mastery", "quit")
        assert "nothing tracked" in out

    def test_reset_clears_the_history(self, config, attempt):
        _, out, session = drive(config, f"load {attempt}", "reset", "progress", "quit")
        assert "cleared" in out and session.state.attempts == []


class TestExploration:
    def test_concept_defaults_to_the_target(self, config, attempt):
        _, out, session = drive(config, f"load {attempt}", "concept", "quit")
        target = session.state.latest.result.intervention.target_concept
        assert target in out and "prerequisites" in out

    def test_concept_by_id_and_unknown_id(self, config):
        _, out, _ = drive(config, "concept binary-search", "concept phlogiston", "quit")
        assert "Binary search" in out and "unknown concept" in out

    def test_problems_and_problem_pinning(self, config, attempt):
        _, out, session = drive(config, "problems", "problem hop_counts", "problem", "quit")
        assert "lower_bound" in out and "hop_counts" in out
        assert session.state.problem.pid == "hop_counts"
        assert "Shortest hop counts" in out

    def test_pinning_a_task_re_diagnoses_a_loaded_file(self, config, attempt):
        _, out, session = drive(config, f"load {attempt}", "problem lower_bound", "quit")
        assert len(session.state.attempts) == 2

    def test_unknown_task_is_refused(self, config):
        _, out, _ = drive(config, "problem ghost", "quit")
        assert "unknown problem" in out

    def test_edit_without_an_editor_is_explained(self, config, attempt, monkeypatch):
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)
        _, out, _ = drive(config, f"load {attempt}", "edit", "quit")
        assert "$EDITOR" in out


class TestEntryPoint:
    def test_session_subcommand_accepts_piped_input(self, attempt, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("hint\nquit\n"))
        code = main([
            "session", "--file", str(attempt), "--backend", "offline",
            "--no-cache", "--color", "never", "--width", "84",
        ])
        out = capsys.readouterr().out
        assert code == 0
        assert "DIAGNOSIS" in out and "hint 1/3" in out and "session ended" in out

    def test_session_subcommand_can_pin_the_task(self, attempt, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("quit\n"))
        code = main([
            "session", "--file", str(attempt), "--problem", "lower_bound",
            "--backend", "offline", "--no-cache", "--color", "never",
        ])
        assert code == 0 and "lower_bound" in capsys.readouterr().out
