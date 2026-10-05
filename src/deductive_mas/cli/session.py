"""Interactive tutoring session.

A batch report is the wrong shape for Socratic teaching, printing all three
rungs at once is a slow worked answer. So the session shows one rung per
request. It also has continuity: mastery carries across attempts, so after
the learner edits the file and runs retry we can answer the question that
matters, was the belief repaired or just the symptom?

    dmas session                    start empty, then `load path/to/attempt.py`
    dmas session --file attempt.py  load and diagnose right away
"""

import ast
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..agents.orchestrator import DeductiveOrchestrator
from ..config import Config
from ..data.problems import all_problems, problem as get_problem
from ..domain import ProblemSpec, Submission, TutoringResult
from ..errors import DMASError
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.mastery import MasteryState
from ..knowledge.ontology import knowledge_graph
from ..ui.ansi import Style
from ..ui.render import (
    render_alignment,
    render_attempt_diff,
    render_concept,
    render_diagnosis,
    render_evidence_scene,
    render_hint,
    render_intervention,
    render_mastery,
    render_progress,
    render_verdict,
)
from ..ui.widgets import (
    Column,
    banner,
    bullet,
    inline,
    key_values,
    panel,
    paragraph,
    rule,
    table,
)
from ..version import CODENAME, __version__

PROMPT = "dmas ▸ "
MAX_SOURCE_BYTES = 256 * 1024


def _short_path(path: Path) -> str:
    """Path the way a person would write it: relative, or under ~."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        pass
    try:
        return "~/" + str(resolved.relative_to(Path.home()))
    except ValueError:
        return str(resolved)


def freeform_problem(entry_point: str) -> ProblemSpec:
    """A task with no oracle, for a file the tutor has never seen.
    Everything comparative is off and the pipeline says so. The structural
    rules, dataflow facts and cost analysis still work, which is most of it.
    """
    return ProblemSpec(
        pid="freeform",
        title="Uploaded file",
        statement="An uploaded file with no reference solution.",
        entry_point=entry_point,
        parameters=(),
        concepts=(),
        reference_solution="",
        tests=(),
    )


@dataclass
class Attempt:
    """One diagnosed version of the learner's file."""

    index: int
    label: str
    source: str
    result: TutoringResult
    at: float = field(default_factory=time.time)


@dataclass
class SessionState:
    path: Optional[Path] = None
    problem: Optional[ProblemSpec] = None
    source: str = ""
    attempts: List[Attempt] = field(default_factory=list)
    mastery: Optional[MasteryState] = None
    revealed: int = 0
    challenge_shown: bool = False

    @property
    def latest(self) -> Optional[Attempt]:
        return self.attempts[-1] if self.attempts else None

    @property
    def previous(self) -> Optional[Attempt]:
        return self.attempts[-2] if len(self.attempts) > 1 else None

    @property
    def label(self) -> str:
        return self.path.name if self.path else "(no file)"


@dataclass
class Command:
    name: str
    aliases: Tuple[str, ...]
    summary: str
    usage: str
    handler: Callable[["TutorSession", List[str]], bool]


class TutorSession:
    """The read-eval-print loop."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        style: Optional[Style] = None,
        orchestrator: Optional[DeductiveOrchestrator] = None,
        graph: Optional[KnowledgeGraph] = None,
        input_stream=None,
        output_stream=None,
    ):
        self.config = config or Config()
        self.style = style or Style(self.config.color)
        self.orchestrator = orchestrator or DeductiveOrchestrator(self.config)
        self.graph = graph or knowledge_graph()
        self.input = input_stream or sys.stdin
        self.output = output_stream or sys.stdout
        self.state = SessionState()
        self.commands: Dict[str, Command] = {}
        self._register()

    # i/o
    def emit(self, lines):
        if isinstance(lines, str):
            lines = [lines]
        self.output.write("\n".join(lines) + "\n")
        self.output.flush()

    def blank(self):
        self.output.write("\n")

    def say(self, message: str, colour: Optional[str] = None):
        self.emit(paragraph(self.style, message, indent="  ", colour=colour))

    def problem_error(self, message: str):
        self.emit("  " + self.style.danger("!  ") + message)

    # commands
    def _register(self):
        definitions = [
            Command("load", ("open", "l"), "read a code file and diagnose it",
                    "load <path>", TutorSession.cmd_load),
            Command("retry", ("again", "r"), "re-read the file and show what changed",
                    "retry", TutorSession.cmd_retry),
            Command("hint", ("h",), "reveal the next rung of the Socratic ladder",
                    "hint", TutorSession.cmd_hint),
            Command("challenge", ("c",), "show the counter-factual challenge",
                    "challenge", TutorSession.cmd_challenge),
            Command("why", ("evidence",), "show the evidence behind the diagnosis",
                    "why", TutorSession.cmd_why),
            Command("report", ("full",), "print the complete report for this attempt",
                    "report", TutorSession.cmd_report),
            Command("code", ("show", "src"), "show the loaded source, marked at the divergence",
                    "code", TutorSession.cmd_code),
            Command("problem", ("task",), "show or set the task this file answers",
                    "problem [<id>|auto]", TutorSession.cmd_problem),
            Command("problems", ("tasks",), "list the built-in tasks",
                    "problems", TutorSession.cmd_problems),
            Command("concept", ("kg",), "explore a concept (defaults to the target)",
                    "concept [<id>]", TutorSession.cmd_concept),
            Command("mastery", ("skills",), "what the session believes you have mastered",
                    "mastery", TutorSession.cmd_mastery),
            Command("progress", ("history",), "attempts so far in this session",
                    "progress", TutorSession.cmd_progress),
            Command("edit", (), "open the loaded file in $EDITOR",
                    "edit", TutorSession.cmd_edit),
            Command("reset", (), "forget the mastery estimate and history",
                    "reset", TutorSession.cmd_reset),
            Command("help", ("?",), "list the commands",
                    "help [<command>]", TutorSession.cmd_help),
            Command("quit", ("exit", "q"), "leave the session",
                    "quit", TutorSession.cmd_quit),
        ]
        for command in definitions:
            self.commands[command.name] = command
            for alias in command.aliases:
                self.commands[alias] = command

    # run
    def run(self, *, path: Optional[str] = None, problem_id: Optional[str] = None) -> int:
        self.emit(self.welcome())
        if problem_id:
            self.cmd_problem(["problem", problem_id])
        if path:
            self.cmd_load(["load", path])

        self._setup_readline()
        while True:
            try:
                line = self._read()
            except (EOFError, KeyboardInterrupt):
                self.blank()
                self.emit("  " + self.style.muted("session ended"))
                return 0
            if line is None:
                self.blank()
                self.emit("  " + self.style.muted("session ended"))
                return 0
            if not self.dispatch(line):
                return 0

    def _read(self) -> Optional[str]:
        interactive = getattr(self.input, "isatty", lambda: False)()
        if interactive:
            return input(self.style.paint(PROMPT, "primary", bold=True))
        line = self.input.readline()
        if not line:
            return None
        text = line.rstrip("\n")
        if text.strip():
            # echo piped commands so a transcript reads like a conversation
            self.emit(self.style.muted(PROMPT) + text)
        return text

    def dispatch(self, line: str) -> bool:
        """Run one command line, returns False when the session should end."""
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        name = parts[0].lower()

        command = self.commands.get(name)
        if command is None:
            # a bare path is the most common first thing to type, treat it as load
            if Path(parts[0]).expanduser().exists():
                return self.cmd_load(["load"] + parts)
            near = self._suggest(name)
            self.problem_error(
                f"unknown command {name!r}" + (f" — did you mean `{near}`?" if near else "")
            )
            self.emit("  " + self.style.muted("`help` lists everything"))
            return True
        try:
            return command.handler(self, parts)
        except DMASError as exc:
            self.problem_error(str(exc))
            return True
        except KeyError as exc:
            self.problem_error(str(exc).strip("'"))
            return True
        except Exception as exc:  # a session must survive a bad command
            self.problem_error(f"{type(exc).__name__}: {exc}")
            return True

    def _suggest(self, name: str) -> Optional[str]:
        from ..util.text import normalised_levenshtein

        ranked = sorted(
            (normalised_levenshtein(name, candidate), candidate)
            for candidate in self.commands
        )
        return ranked[0][1] if ranked and ranked[0][0] <= 0.5 else None

    # screens
    def welcome(self) -> List[str]:
        style = self.style
        out = banner(
            style,
            f"{CODENAME} · interactive tutor",
            f"v{__version__}  ·  reasoning backend: {self.orchestrator.reasoner.name}",
        )
        out.append("")
        out += paragraph(
            style,
            "Load a Python file and I will find where your reasoning first left the correct "
            "path, name the belief behind it, and ask the question that exposes it. I will "
            "not show you the answer.",
            indent="  ",
        )
        out.append("")
        out += key_values(
            style,
            [
                ("load <file>", style.muted("read a file and diagnose it")),
                ("hint", style.muted("one rung at a time — think between them")),
                ("retry", style.muted("after you edit, see whether the belief actually changed")),
                ("help", style.muted("everything else")),
            ],
            indent="  ",
        )
        return out

    # handlers
    def cmd_load(self, parts: List[str]) -> bool:
        if len(parts) < 2:
            self.problem_error("usage: load <path>")
            return True
        path = Path(parts[1]).expanduser()
        if not path.exists():
            self.problem_error(f"no such file: {path}")
            return True
        if path.is_dir():
            self.problem_error(f"{path} is a directory")
            return True
        if path.stat().st_size > MAX_SOURCE_BYTES:
            self.problem_error(f"{path} is larger than {MAX_SOURCE_BYTES // 1024} KB")
            return True
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self.problem_error(f"cannot read {path}: {exc}")
            return True

        self.state.path = path
        self.state.source = source
        if len(parts) > 2:
            self.state.problem = get_problem(parts[2])
        elif self.state.problem is None or self.state.problem.pid == "freeform":
            self.state.problem = self._detect_problem(source)
        self.blank()
        shown = _short_path(path)
        lines = len(source.splitlines())
        self.emit(
            inline(
                self.style,
                [
                    (f"loaded  {shown}", self.style.muted("loaded  ") + self.style.paint(shown, "text")),
                    (f"{lines} lines", self.style.muted(f"{lines} lines")),
                    (
                        f"task  {self.state.problem.pid}",
                        self.style.muted("task  ") + self.style.accent(self.state.problem.pid),
                    ),
                ],
            )
        )
        return self._diagnose(reason="load")

    def cmd_retry(self, parts: List[str]) -> bool:
        if self.state.path is None:
            self.problem_error("nothing loaded yet — `load <path>` first")
            return True
        try:
            source = self.state.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self.problem_error(f"cannot re-read {self.state.path}: {exc}")
            return True
        if source == self.state.source and self.state.attempts:
            self.blank()
            self.say(
                "the file has not changed since the last attempt — edit it and run `retry` "
                "again, or ask for another `hint`",
                colour="muted",
            )
            return True
        self.state.source = source
        return self._diagnose(reason="retry")

    def cmd_hint(self, parts: List[str]) -> bool:
        attempt = self._require_attempt()
        if attempt is None:
            return True
        hints = attempt.result.intervention.hints
        if not hints:
            self.blank()
            self.say("there is no guidance for this attempt — nothing was diagnosed.", "muted")
            return True
        if self.state.revealed >= len(hints):
            self.blank()
            self.say(
                "that was the last rung. Try `challenge`, or edit the file and run `retry`.",
                colour="muted",
            )
            return True
        hint = hints[self.state.revealed]
        self.state.revealed += 1
        self.blank()
        self.emit(
            render_hint(
                self.style, hint,
                index=self.state.revealed, total=len(hints), graph=self.graph,
            )
        )
        return True

    def cmd_challenge(self, parts: List[str]) -> bool:
        attempt = self._require_attempt()
        if attempt is None:
            return True
        challenge = attempt.result.intervention.challenge
        if challenge is None:
            self.blank()
            self.say("no counter-factual challenge was produced for this attempt.", "muted")
            return True
        inner = self.style.width - 4
        body = list(paragraph(self.style, challenge.prompt, indent="", width=inner))
        if challenge.trace_question:
            body.append("")
            body += paragraph(
                self.style, challenge.trace_question, indent="", width=inner, colour="muted"
            )
        self.blank()
        self.emit(panel(self.style, body, title="Counter-factual challenge", colour="accent"))
        self.state.challenge_shown = True
        return True

    def cmd_why(self, parts: List[str]) -> bool:
        attempt = self._require_attempt()
        if attempt is None:
            return True
        self.blank()
        self.emit(rule(self.style, "Evidence"))
        self.blank()
        self.emit(render_diagnosis(self.style, attempt.result.diagnosis, show_evidence=True))
        self.blank()
        self.emit(rule(self.style, "Where it comes from"))
        self.blank()
        self.emit(render_alignment(self.style, attempt.result.alignment, self.graph))
        return True

    def cmd_report(self, parts: List[str]) -> bool:
        attempt = self._require_attempt()
        if attempt is None:
            return True
        from ..ui.render import render_tutoring_result

        self.blank()
        self.emit(render_tutoring_result(self.style, attempt.result, self.graph))
        # a full report shows every rung, so the ladder counter has to agree
        self.state.revealed = len(attempt.result.intervention.hints)
        return True

    def cmd_code(self, parts: List[str]) -> bool:
        attempt = self._require_attempt()
        if attempt is None:
            return True
        self.blank()
        self.emit(render_evidence_scene(self.style, attempt.result))
        return True

    def cmd_problem(self, parts: List[str]) -> bool:
        if len(parts) < 2:
            if self.state.problem is None:
                self.say("no task selected — `problems` lists them, or just `load` a file.", "muted")
                return True
            spec = self.state.problem
            self.blank()
            self.emit(banner(self.style, spec.title, f"{spec.pid}  ·  entry point {spec.entry_point}()"))
            self.blank()
            self.emit(paragraph(self.style, spec.statement or "(no statement)", indent="  "))
            if spec.concepts:
                self.blank()
                self.emit(
                    "  " + self.style.muted("concepts  ")
                    + self.style.muted(", ").join(
                        self.style.accent(self.graph.name(c)) if c in self.graph else c
                        for c in spec.concepts
                    )
                )
            return True
        choice = parts[1]
        if choice == "auto":
            self.state.problem = self._detect_problem(self.state.source)
        else:
            self.state.problem = get_problem(choice)
        self.emit(
            "  " + self.style.muted("task set to ") + self.style.accent(self.state.problem.pid)
        )
        if self.state.source:
            return self._diagnose(reason="problem")
        return True

    def cmd_problems(self, parts: List[str]) -> bool:
        rows = [
            [spec.pid, spec.title, f"{spec.difficulty:+.1f}", spec.entry_point + "()"]
            for spec in all_problems()
        ]
        self.blank()
        self.emit(
            table(
                self.style,
                [Column("id", min_width=14, priority=100), Column("title", min_width=18),
                 Column("difficulty", "right", 10, priority=30), Column("entry point", priority=60)],
                rows,
                indent="  ",
            )
        )
        self.blank()
        self.emit("  " + self.style.muted("`problem <id>` to pin one, or `load` a file and I will guess"))
        return True

    def cmd_concept(self, parts: List[str]) -> bool:
        target = parts[1] if len(parts) > 1 else None
        if target is None:
            attempt = self.state.latest
            if attempt is not None:
                target = (
                    attempt.result.intervention.target_concept
                    or attempt.result.alignment.root_cause
                )
        if target is None:
            self.problem_error("no concept in focus — `concept <id>`, or diagnose something first")
            return True
        if target not in self.graph:
            self.problem_error(f"unknown concept: {target}")
            return True
        self.blank()
        self.emit(render_concept(self.style, self.graph, target))
        return True

    def cmd_mastery(self, parts: List[str]) -> bool:
        state = self.state.mastery
        if state is None:
            self.blank()
            self.say("nothing tracked yet — diagnose a submission first", "muted")
            return True
        # only show what this session actually observed. The cold start prior
        # covers all 65 concepts and listing Dijkstra as "not yet" for someone who
        # only tried binary search is noise.
        observed = {cid: n for cid, n in state.observations.items() if n > 0}
        posterior = {cid: state.posterior[cid] for cid in observed if cid in state.posterior}
        self.blank()
        self.emit(rule(self.style, "Concepts this session has evidence about"))
        self.blank()
        self.emit(
            render_mastery(
                self.style, posterior or dict(state.posterior), self.graph,
                threshold=self.config.knowledge.mastery_threshold,
            )
        )
        return True

    def cmd_progress(self, parts: List[str]) -> bool:
        self.blank()
        self.emit(rule(self.style, "This session"))
        self.blank()
        self.emit(render_progress(self.style, self.state.attempts, self.graph))
        return True

    def cmd_edit(self, parts: List[str]) -> bool:
        if self.state.path is None:
            self.problem_error("nothing loaded yet — `load <path>` first")
            return True
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            self.problem_error("set $EDITOR (or $VISUAL) to edit from here")
            return True
        try:
            subprocess.call(shlex.split(editor) + [str(self.state.path)])
        except OSError as exc:
            self.problem_error(f"could not launch {editor!r}: {exc}")
            return True
        self.say("run `retry` when you are ready to see what changed.", "muted")
        return True

    def cmd_reset(self, parts: List[str]) -> bool:
        self.state.mastery = None
        self.state.attempts = []
        self.state.revealed = 0
        self.state.challenge_shown = False
        self.emit("  " + self.style.muted("mastery estimate and history cleared"))
        return True

    def cmd_help(self, parts: List[str]) -> bool:
        if len(parts) > 1:
            command = self.commands.get(parts[1].lower())
            if command is None:
                self.problem_error(f"unknown command: {parts[1]}")
                return True
            self.blank()
            self.emit("  " + self.style.heading(command.usage))
            self.emit("  " + self.style.muted(command.summary))
            if command.aliases:
                self.emit("  " + self.style.muted("aliases: " + ", ".join(command.aliases)))
            return True

        seen = []
        rows = []
        for command in self.commands.values():
            if command.name in seen:
                continue
            seen.append(command.name)
            rows.append([
                command.usage,
                command.summary,
                ", ".join(command.aliases) or "—",
            ])
        self.blank()
        self.emit(
            table(
                self.style,
                [Column("command", min_width=20, priority=100), Column("what it does", min_width=24),
                 Column("aliases", priority=20)],
                rows,
                indent="  ",
            )
        )
        return True

    def cmd_quit(self, parts: List[str]) -> bool:
        self.blank()
        if self.state.attempts:
            self.emit(render_progress(self.style, self.state.attempts, self.graph))
            self.blank()
        self.emit("  " + self.style.muted("session ended"))
        return False

    # engine
    def _require_attempt(self) -> Optional[Attempt]:
        attempt = self.state.latest
        if attempt is None:
            self.problem_error("nothing diagnosed yet — `load <path>` first")
        return attempt

    def _detect_problem(self, source: str) -> ProblemSpec:
        """Match a file to a task by the function it defines."""
        names: List[str] = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            names = []
        else:
            names = [
                node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
        matches = [spec for spec in all_problems() if spec.entry_point in names]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            self.say(
                "several tasks match this file: "
                + ", ".join(spec.pid for spec in matches)
                + ". Using the first; `problem <id>` to choose.",
                colour="muted",
            )
            return matches[0]
        entry = names[0] if names else "solve"
        self.say(
            "no built-in task matches this file, so I will report structural findings only — "
            "no counterexample search and no complexity comparison. `problems` lists the tasks "
            "I have references for.",
            colour="muted",
        )
        return freeform_problem(entry)

    def _diagnose(self, *, reason: str) -> bool:
        if not self.state.source.strip():
            self.problem_error("the file is empty")
            return True
        spec = self.state.problem or freeform_problem("solve")
        index = len(self.state.attempts) + 1
        submission = Submission(
            sid=f"{self.state.label}#{index}",
            problem_id=spec.pid,
            source=self.state.source,
            author="session",
        )

        started = time.perf_counter()
        result = self.orchestrator.tutor(spec, submission, mastery=self.state.mastery)
        elapsed = time.perf_counter() - started

        if result.mastery_posterior:
            self.state.mastery = MasteryState(
                posterior=dict(result.mastery_posterior),
                observations=dict(result.mastery_observations),
            )
        previous = self.state.latest
        if (
            previous is not None
            and not result.diagnosis.hits
            and previous.result.diagnosis.hits
            and self.state.mastery is not None
        ):
            # the learner was pointed at a concept, edited, and the finding is gone.
            # That is evidence about that concept (the strongest kind a tutor gets) and
            # only the session, which remembers the previous turn, can record it.
            repaired = [
                cid for cid in (
                    previous.result.intervention.target_concept,
                    previous.result.alignment.root_cause,
                )
                if cid and cid in self.graph
            ]
            if repaired:
                self.state.mastery = self.orchestrator.tracker.apply_success(
                    self.state.mastery, dict.fromkeys(repaired), weight=0.8
                )
                result.mastery_posterior = dict(self.state.mastery.posterior)
                result.mastery_observations = dict(self.state.mastery.observations)
        self.state.attempts.append(
            Attempt(index=index, label=self.state.label, source=self.state.source, result=result)
        )
        self.state.revealed = 0
        self.state.challenge_shown = False

        self.blank()
        self.emit(render_verdict(self.style, result))
        for warning in result.telemetry.warnings:
            self.emit("  " + self.style.warning("!  ") + self.style.muted(warning))

        if previous is not None and reason in ("retry", "problem"):
            self.blank()
            self.emit(rule(self.style, f"Attempt {index} vs {previous.index}"))
            self.blank()
            self.emit(render_attempt_diff(self.style, previous.result, result, self.graph))

        self.blank()
        self.emit(self._next_steps(result, elapsed))
        return True

    def _next_steps(self, result: TutoringResult, elapsed: float) -> List[str]:
        style = self.style
        if not result.diagnosis.hits:
            return bullet(
                style,
                "Nothing to diagnose. Try a harder task with `problems`, or load another file.",
                indent="  ",
            )
        commands = ("hint", "why", "code", "challenge")
        parts = [("next", style.muted("next "))] + [
            (name, style.paint(name, "primary", bold=True)) for name in commands
        ]
        out = inline(style, parts, separator="  ·  ")
        out.append("  " + style.muted(f"diagnosed in {elapsed * 1000:.0f} ms  ·  `help` explains each"))
        return out

    # readline
    def _setup_readline(self):
        if not getattr(self.input, "isatty", lambda: False)():
            return
        try:
            import readline
        except ImportError:
            return

        names = sorted({command.name for command in self.commands.values()})

        def complete(text: str, index: int) -> Optional[str]:
            buffer = readline.get_line_buffer().lstrip()
            if " " in buffer:
                options = [
                    str(entry)
                    for entry in Path(".").glob(text + "*")
                ] + [spec.pid for spec in all_problems() if spec.pid.startswith(text)]
            else:
                options = [name for name in names if name.startswith(text)]
            return options[index] if index < len(options) else None

        readline.set_completer(complete)
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")


def start(
    config: Optional[Config] = None,
    *,
    style: Optional[Style] = None,
    path: Optional[str] = None,
    problem_id: Optional[str] = None,
    input_stream=None,
    output_stream=None,
) -> int:
    session = TutorSession(
        config, style=style, input_stream=input_stream, output_stream=output_stream
    )
    return session.run(path=path, problem_id=problem_id)
