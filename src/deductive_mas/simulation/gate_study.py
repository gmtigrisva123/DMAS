"""Measuring the gate, two ways.

Adversarial benchmark: candidate hints are manufactured from every (problem,
submission) pair so the right verdict is known by construction (a reference
statement the student never wrote must be rejected, the student's own line
quoted back must be accepted, ...). Eight categories cover the ways a tutor
can leak (verbatim, renamed, prose repair, whole solution) and the ways a
legit hint looks (quote, question, fallback rung). One category is outside
what the gate is designed for (a repair in plain words with no code span),
included so the limitation is a number and not just a sentence.

Live audit: run the full pipeline with a reasoning model and tabulate what
the model proposed vs what the student finally saw: how often a rung went
through as generated, how often it was replaced and on which channel. The
leak scores of the replaced rungs say what the student would have seen if
the gate was only advisory.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..agents.intervention import _FALLBACK_RUNGS
from ..agents.orchestrator import DeductiveOrchestrator
from ..agents.verifier import ComplianceVerifier
from ..config import Config
from ..data.heldout import all_heldout_submissions
from ..data.problems import problem as get_problem
from ..data.submissions import all_submissions
from ..domain import ProblemSpec, Submission
from ..retrieval.ma_rag import GroundedContext
from ..util.text import code_tokens

# a repair for every problem in prose with no code span. These are the
# leaks the gate is NOT built to see, the benchmark reports how many it
# catches anyway.
PROSE_REPAIRS: Dict[str, str] = {
    "lower_bound": (
        "Initialise the upper bound to the length of the list rather than to the last index, "
        "and keep the loop running while the lower bound is strictly less than it."
    ),
    "hop_counts": (
        "Mark each vertex as visited at the moment you append it to the queue, not when you "
        "remove it, and use a deque so that removing from the front is constant time."
    ),
    "min_coins": (
        "Fill the table from smaller amounts to larger ones and, for each amount, take one plus "
        "the best answer for the amount minus each coin, keeping the minimum."
    ),
    "grid_paths": (
        "Return zero when either dimension is zero, fill a table where every cell is the sum of "
        "the cell above and the cell to its left, and answer from the bottom-right cell."
    ),
    "two_sum_sorted": (
        "Keep the loop running only while the left index is strictly less than the right one, "
        "moving the left pointer up when the sum is too small and the right one down otherwise."
    ),
    "max_intervals": (
        "Sort the intervals by their finishing time, then take each interval whose start is not "
        "before the finish of the last one you took."
    ),
    "upper_bound": (
        "Move the lower bound past the midpoint whenever the middle element is less than or "
        "equal to the target, and initialise the upper bound to the length of the list."
    ),
    "count_components": (
        "Allocate the visited array with one entry per vertex, and count a component each time "
        "the outer loop reaches a vertex that has not been visited."
    ),
    "climb_ways": (
        "Return one when no stairs remain, and fill the table from one stair upward so that each "
        "entry sums the entries reachable by one allowed step."
    ),
    "dedupe": (
        "Keep a set of the values already seen and build a new list, appending a value only when "
        "it is not yet in the set; never remove from the list you are iterating over."
    ),
    "knapsack": (
        "Iterate the capacity from the largest value downward so each item is used at most once, "
        "and size the table with one more entry than the capacity."
    ),
    "merge_sorted": (
        "After the main loop, append the remaining elements from the current index onward, not "
        "from the index after it, and stop the loop before an index reaches the length."
    ),
}


@dataclass(frozen=True)
class Candidate:
    """One manufactured hint with its known verdict."""

    category: str
    text: str
    should_reject: bool
    submission_id: str


@dataclass
class GateBenchmarkResult:
    verdicts: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for category in sorted({v["category"] for v in self.verdicts}):
            rows = [v for v in self.verdicts if v["category"] == category]
            rejected = sum(1 for v in rows if v["rejected"])
            by_leakage = sum(1 for v in rows if v["rejected_by_leakage"])
            by_grounding = sum(1 for v in rows if v["rejected_by_grounding_only"])
            out[category] = {
                "n": len(rows),
                "should_reject": rows[0]["should_reject"],
                "rejection_rate": round(rejected / len(rows), 4),
                "rejected_by_leakage_rate": round(by_leakage / len(rows), 4),
                "rejected_by_grounding_only_rate": round(by_grounding / len(rows), 4),
                "channels": _count(v["channel"] for v in rows if v["rejected_by_leakage"]),
            }
        leaks = [v for v in self.verdicts if v["should_reject"] and v["category"] != "prose-repair"]
        legit = [v for v in self.verdicts if not v["should_reject"]]
        out["_overall"] = {
            "leak_recall_excluding_prose": round(
                sum(1 for v in leaks if v["rejected"]) / len(leaks), 4
            ) if leaks else None,
            "legitimate_false_rejection_rate": round(
                sum(1 for v in legit if v["rejected"]) / len(legit), 4
            ) if legit else None,
            "legitimate_false_rejection_by_leakage_rate": round(
                sum(1 for v in legit if v["rejected_by_leakage"]) / len(legit), 4
            ) if legit else None,
        }
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "verdicts": self.verdicts}


def _count(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))


# manufacturing candidates
def _statements(source: str) -> List[str]:
    """Simple statements of a program as the author wrote them.
    Source text is quoted, not ast.unparse, so a "quote" of the student's line
    is really the line they wrote.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Return, ast.If, ast.While)):
            line = lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else ""
            if line:
                out.append(line)
    return list(dict.fromkeys(out))


def _normalised(statement: str) -> str:
    return " ".join(code_tokens(statement, normalise_identifiers=True))


def _rename(statement: str, mapping: Dict[str, str]) -> str:
    def swap(match: "re.Match[str]") -> str:
        return mapping.get(match.group(0), match.group(0))

    return re.sub(r"[A-Za-z_][A-Za-z0-9_]*", swap, statement)


def _names(source: str) -> List[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    return list(dict.fromkeys(names))


_KEYWORDS = frozenset(
    """if while for in not and or return def else elif True False None
    len range set list dict tuple sorted min max abs sum enumerate zip deque print
    append pop popleft add remove extend insert count index sort reverse keys values items""".split()
)


def manufacture(problem: ProblemSpec, submission: Submission) -> List[Candidate]:
    """Every candidate category for one (problem, submission) pair."""
    reference_statements = _statements(problem.reference_solution)
    student_statements = _statements(submission.source)
    student_norm = {_normalised(s) for s in student_statements}
    novel = [s for s in reference_statements if _normalised(s) not in student_norm]
    out: List[Candidate] = []
    sid = submission.sid
    # a correct submission is a solution, quoting the reference to its author
    # leaks nothing, so leak categories only apply to flawed submissions
    flawed = bool(submission.gold_misconceptions)

    # 1. a reference statement the student did not write, verbatim
    for statement in novel[:3] if flawed else []:
        out.append(Candidate("reference-line", f"Consider the line `{statement}`.", True, sid))
    # 2. same, all identifiers renamed
    for statement in novel[:3] if flawed else []:
        names = [n for n in _names(statement) if n not in _KEYWORDS and n not in ("len", "range")]
        mapping = {name: f"v{i}" for i, name in enumerate(names)}
        out.append(Candidate("reference-line-renamed", f"Consider the line `{_rename(statement, mapping)}`.", True, sid))
    # 3. a repair as prose around a code span: "set X to Y"
    for statement in novel if flawed else []:
        if "=" in statement and not statement.startswith(("if", "while", "return")) and "==" not in statement:
            target, _, value = statement.partition("=")
            target, value = target.strip(), value.strip()
            if target and value and not target.startswith(("+", "-")):
                out.append(Candidate("prose-assignment", f"Set `{target}` to `{value}` and run it again.", True, sid))
                break
    # 4. the whole reference solution in a fence
    if flawed:
        out.append(Candidate("full-solution", "Here is a correct version:\n```python\n" + problem.reference_solution + "\n```", True, sid))
    # 5. a repair in plain words, no code span
    if flawed and problem.pid in PROSE_REPAIRS:
        out.append(Candidate("prose-repair", PROSE_REPAIRS[problem.pid], True, sid))
    # 6. the student's own statement quoted back
    for statement in student_statements[:3]:
        out.append(Candidate("student-line", f"Look again at `{statement}`. What did you intend it to do?", False, sid))
    # 7. a pure question naming a student variable
    for name in [n for n in _names(submission.source) if n not in _KEYWORDS][:2]:
        out.append(Candidate("question", f"What value does `{name}` hold the last time the loop guard is checked?", False, sid))
    # 8. the fixed fallback rungs
    for rung in _FALLBACK_RUNGS[:2]:
        out.append(Candidate("fallback-rung", rung, False, sid))
    return out


class GateBenchmark:
    """Scores the gate on manufactured candidates with known verdicts."""

    def __init__(self, config: Optional[Config] = None, verifier: Optional[ComplianceVerifier] = None):
        self.config = config or Config()
        self.verifier = verifier or ComplianceVerifier(self.config)

    def run(self, submissions: Optional[Sequence[Submission]] = None) -> GateBenchmarkResult:
        bank = list(submissions if submissions is not None else all_submissions() + all_heldout_submissions())
        result = GateBenchmarkResult()
        rag = self.verifier.rag
        for submission in bank:
            problem = get_problem(submission.problem_id)
            leakage_context = self.verifier.context_for(problem.reference_solution, submission.source)
            grounding = rag.ground(
                problem_title=problem.title,
                misconception_ids=list(submission.gold_misconceptions),
                misconception_texts=[],
                concepts=tuple(problem.concepts),
                frontier=(),
                divergence_kind=None,
                complexity_gap=None,
            )
            for candidate in manufacture(problem, submission):
                verdict = self.verifier.review(candidate.text, leakage_context, grounding)
                leak_rejected = verdict.leakage.rejected(self.config.intervention.max_leakage)
                result.verdicts.append(
                    {
                        "submission": submission.sid,
                        "category": candidate.category,
                        "should_reject": candidate.should_reject,
                        "text": candidate.text,
                        "rejected": not verdict.accepted,
                        "rejected_by_leakage": leak_rejected,
                        "rejected_by_grounding_only": (not verdict.accepted) and not leak_rejected,
                        "leakage": round(verdict.leakage.score, 4),
                        "channel": verdict.leakage.channel,
                    }
                )
        return result


# the live audit
@dataclass
class GateAuditResult:
    backend: str = "offline"
    records: List[Dict[str, Any]] = field(default_factory=list)
    degraded_rows: int = 0

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for item in sorted({r["item"] for r in self.records}):
            rows = [r for r in self.records if r["item"] == item and r["candidate"]]
            if not rows:
                continue
            leak = [r for r in rows if r["leakage"] is not None and (
                r["leakage"] > 0.18 or r["structural_match"])]
            out[item] = {
                "n": len(rows),
                "reached_student_as_generated": round(
                    sum(1 for r in rows if r["final_source"] == "generated") / len(rows), 4),
                "replaced": _count(r["final_source"] for r in rows if r["final_source"] != "generated"),
                "rejected_for_leakage_rate": round(len(leak) / len(rows), 4),
                "rejected_for_grounding_only_rate": round(
                    sum(1 for r in rows if not r["accepted"] and r not in leak) / len(rows), 4),
                "leakage_channels": _count(r["leakage_channel"] for r in leak),
                "max_raw_leakage": max((r["leakage"] or 0.0) for r in rows),
            }
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "degraded_rows": self.degraded_rows,
            "summary": self.summary(),
            "records": self.records,
        }


class GateAudit:
    """Runs the full pipeline and collects the gate audit records."""

    def __init__(self, config: Optional[Config] = None, orchestrator: Optional[DeductiveOrchestrator] = None):
        self.config = config or Config()
        self.orchestrator = orchestrator or DeductiveOrchestrator(self.config)

    def run(self, submissions: Optional[Sequence[Submission]] = None) -> GateAuditResult:
        bank = list(submissions if submissions is not None else all_submissions() + all_heldout_submissions())
        result = GateAuditResult(backend=self.orchestrator.reasoner.name)
        for submission in bank:
            problem = get_problem(submission.problem_id)
            tutoring = self.orchestrator.tutor(problem, submission)
            degraded = any("degraded" in w for w in tutoring.telemetry.warnings)
            if degraded:
                result.degraded_rows += 1
            for record in tutoring.gate_log:
                row = dict(record)
                row["submission"] = submission.sid
                row["degraded"] = degraded
                row["final_leakage"] = round(tutoring.intervention.leakage, 4)
                result.records.append(row)
        return result
