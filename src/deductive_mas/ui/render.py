"""Turn result objects into report lines. Pure functions (lines in, lines out)
so the same report works on a terminal, in a file or in a test. Order
follows the reasoning: the code and its failing input, the diagnosis and
evidence, the prerequisite chain, the hint ladder.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from ..domain import (
    AlignmentReport,
    Diagnosis,
    Intervention,
    Severity,
    TutoringResult,
)
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.misconceptions import MISCONCEPTIONS
from .ansi import Style
from .widgets import (
    Column,
    badge,
    inline,
    banner,
    bar,
    bullet,
    histogram,
    key_values,
    panel,
    paragraph,
    rule,
    source_listing,
    table,
    tree,
)


# tutoring session
def render_tutoring_result(
    style: Style,
    result: TutoringResult,
    graph: KnowledgeGraph,
    *,
    show_code: bool = True,
    show_evidence: bool = True,
) -> List[str]:
    """The main report for one submission."""
    out: List[str] = []
    out += banner(
        style,
        f"Diagnosis  {result.submission.sid}",
        f"{result.problem.title}  ·  problem {result.problem.pid}",
    )
    out.append("")

    diagnosis = result.diagnosis
    if show_code:
        out += render_evidence_scene(style, result)
        out.append("")

    out.append(rule(style, "Cognitive diagnosis"))
    out.append("")
    out += render_diagnosis(style, diagnosis, show_evidence=show_evidence)
    out.append("")

    out.append(rule(style, "Knowledge-graph alignment"))
    out.append("")
    out += render_alignment(style, result.alignment, graph)
    out.append("")

    out.append(rule(style, "Cognitive intervention"))
    out.append("")
    out += render_intervention(
        style, result.intervention, graph, root_cause=result.alignment.root_cause
    )
    out.append("")

    out.append(rule(style, "Provenance"))
    out.append("")
    out += render_telemetry(style, result)
    return out


def render_evidence_scene(style: Style, result: TutoringResult) -> List[str]:
    """The code with the divergence marked, plus the smallest failing input."""
    diagnosis = result.diagnosis
    divergence = diagnosis.divergence
    note = ""
    highlight = None
    if divergence is not None and divergence.student_line:
        highlight = divergence.student_line
        note = f"{divergence.kind} — first checkpoint where the executions differ"

    body = source_listing(
        style, result.submission.source, highlight=highlight, context=3, note=note, indent=""
    )
    counterexample = diagnosis.counterexample
    if counterexample is not None:
        body.append("")
        if counterexample.crashed:
            body.append(
                style.muted("smallest failing input  ")
                + style.code(repr(counterexample.args))
                + style.muted("  raises  ")
                + style.danger(str(counterexample.student_error))
            )
        else:
            body.append(
                style.muted("smallest failing input  ")
                + style.code(repr(counterexample.args))
                + style.muted("  yields  ")
                + style.danger(repr(counterexample.student_output))
                + style.muted("  where  ")
                + style.success(repr(counterexample.reference_output))
                + style.muted("  is required")
            )
        if counterexample.shrink_steps:
            body.append(
                style.muted(
                    f"minimised by delta debugging in {counterexample.shrink_steps} reduction step(s)"
                )
            )
    return panel(style, body, title="Evidence", colour="muted")


def render_diagnosis(style: Style, diagnosis: Diagnosis, *, show_evidence: bool = True) -> List[str]:
    out: List[str] = []
    if not diagnosis.hits:
        out += paragraph(
            style,
            "No misconception passed the reporting threshold. The submission shows no "
            "structural warning sign and matched the reference on every input examined.",
            colour="muted",
        )
        return out

    # the id is the one string the reader must be able to copy and look up, so
    # it is never the column that gets cut. On a narrow terminal the readable
    # name is dropped instead (it is repeated in the prose below).
    widest = max(len(hit.misconception_id) for hit in diagnosis.hits)
    show_name = style.width >= widest + 62

    rows = []
    for hit in diagnosis.hits:
        entry = MISCONCEPTIONS.get(hit.misconception_id)
        row = [style.paint(hit.misconception_id, "text")]
        if show_name:
            row.append(entry.name if entry else "")
        row += [
            bar(style, hit.belief, width=12) + " " + style.belief(hit.belief),
            style.severity(str(hit.severity)),
            style.muted(str(hit.span) if hit.span else "—"),
            style.muted("+".join(str(k) for k in hit.kinds)),
        ]
        rows.append(row)

    columns = [Column("misconception", min_width=widest, priority=100)]
    if show_name:
        columns.append(Column("name", min_width=12, priority=20))
    columns += [
        Column("belief", min_width=18, priority=90),
        Column("severity", priority=60),
        Column("at", priority=40),
        Column("evidence", priority=30),
    ]
    out += table(style, columns, rows, indent="  ")
    out.append("")

    primary = diagnosis.primary
    entry = MISCONCEPTIONS.get(primary.misconception_id) if primary else None
    if entry:
        out += paragraph(style, style.paint("The flawed belief: ", "muted") + style.paint(
            f"“{entry.student_voice}”", "accent", italic=True))
        out.append("")
        out += paragraph(style, entry.description)
        out.append("")
    if diagnosis.narrative:
        out += paragraph(style, diagnosis.narrative, colour="muted")
        out.append("")

    if show_evidence and primary is not None:
        out.append("  " + style.muted("why:", bold=True))
        for support in sorted(primary.supports, key=lambda e: -e.belief)[:4]:
            out += bullet(
                style,
                f"{style.muted('[' + str(support.kind) + ']')} {support.rationale}",
                indent="    ",
            )
        out.append("")

    parts = [
        (
            f"complexity  {diagnosis.student_complexity}",
            style.muted("complexity  ") + style.code(diagnosis.student_complexity),
        ),
        (
            f"target  {diagnosis.reference_complexity}",
            style.muted("target  ") + style.code(diagnosis.reference_complexity),
        ),
    ]
    if diagnosis.complexity_gap:
        parts.append(("GAP", badge(style, "GAP", "warning")))
    out += inline(style, parts)
    if diagnosis.inconclusive and diagnosis.hits:
        out += paragraph(
            style,
            "PROVISIONAL — the evidence supports this ranking but does not settle it; "
            "treat it as a hypothesis to test with the student, not a verdict",
            indent="  ",
            colour="warning",
        )
    return out


def render_alignment(style: Style, alignment: AlignmentReport, graph: KnowledgeGraph) -> List[str]:
    out: List[str] = []
    if not alignment.blame:
        return paragraph(style, "No concept could be implicated.", colour="muted")

    rows = []
    for cid, mass in alignment.top_blame(6):
        before = alignment.mastery_before.get(cid)
        after = alignment.mastery_after.get(cid)
        shift = ""
        if before is not None and after is not None:
            arrow = style.danger("↓") if after < before - 1e-6 else style.muted("·")
            shift = f"{style.muted(f'{before:.2f}')} {arrow} {style.paint(f'{after:.2f}', 'text')}"
        rows.append([
            style.accent(cid) if cid == alignment.root_cause else cid,
            style.muted(graph.name(cid)) if cid in graph else "",
            bar(style, min(1.0, mass * 3.0), width=10, colour="accent") + f" {mass:.3f}",
            shift,
        ])
    out += table(
        style,
        [Column("concept", min_width=19, priority=100),
         Column("name", min_width=12, priority=20),
         Column("blame", min_width=16, priority=80),
         Column("mastery", min_width=12, priority=50)],
        rows,
        indent="  ",
    )
    out.append("")

    if alignment.root_cause:
        name = graph.name(alignment.root_cause)
        depth = graph.depth(alignment.root_cause)
        out += inline(
            style,
            [
                (f"root cause  {name}", style.muted("root cause  ") + style.accent(name, bold=True)),
                (
                    f"{alignment.root_cause}, depth {depth}",
                    style.muted(f"{alignment.root_cause}, depth {depth}"),
                ),
            ],
        )
        out += paragraph(
            style,
            "the deepest implicated concept whose own prerequisites are already secure",
            indent="  ",
            colour="muted",
        )
        out.append("")

    for path in alignment.prerequisite_paths[:2]:
        names = [graph.name(cid) for cid in path]
        plain = "learning path  " + " → ".join(names)
        if len(plain) + 2 <= style.width:
            chain = style.muted(" → ").join(
                style.accent(name) if index == 0 else style.paint(name, "text")
                for index, name in enumerate(names)
            )
            out.append("  " + style.muted("learning path  ") + chain)
        else:
            # wrap instead of overflow, an uncontrolled wrap loses the indentation
            out += paragraph(style, plain, indent="  ", colour="muted")
    return out


def render_intervention(
    style: Style,
    intervention: Intervention,
    graph: KnowledgeGraph,
    *,
    root_cause: Optional[str] = None,
) -> List[str]:
    out: List[str] = []
    if not intervention.hints:
        return paragraph(style, "No guidance was produced.", colour="muted")

    target = intervention.target_concept
    parts = []
    if target:
        name = graph.name(target) if target in graph else target
        parts.append((f"target  {name}", style.muted("target  ") + style.accent(name)))
    probability = f"{intervention.zpd_probability:.0%}"
    parts.append(
        (
            f"success probability  {probability}",
            style.muted("success probability  ") + style.paint(probability, "text"),
        )
    )
    in_band = 0.45 <= intervention.zpd_probability <= 0.85
    parts.append(
        ("IN ZPD", badge(style, "IN ZPD", "success")) if in_band
        else ("OUTSIDE ZPD", badge(style, "OUTSIDE ZPD", "warning"))
    )
    out += inline(style, parts)
    if root_cause and target and root_cause != target:
        # the frontier says where the failure comes from, the ZPD says where the
        # learner can be reached today. When they differ say so.
        out += paragraph(
            style,
            "the located root cause is {}; this target was chosen instead because it is "
            "closer to the learner's current reach".format(
                graph.name(root_cause) if root_cause in graph else root_cause
            ),
            indent="  ",
            colour="muted",
        )
    out.append("")

    for hint in intervention.hints:
        label = style.paint(f"{hint.level}", "primary", bold=True) + style.muted(
            f" · {hint.kind}"
        )
        out.append("  " + label)
        out += paragraph(style, hint.text, indent="     ")
        out.append("")

    if intervention.challenge is not None:
        inner = style.width - 4
        body = list(paragraph(style, intervention.challenge.prompt, indent="", width=inner))
        if intervention.challenge.trace_question:
            body.append("")
            body += paragraph(
                style, intervention.challenge.trace_question, indent="", width=inner, colour="muted"
            )
        out += panel(style, body, title="Counter-factual challenge", colour="accent")
        out.append("")

    leakage = f"{intervention.leakage:.2f}"
    grounding = f"{intervention.grounding:.2f}"
    gate = [
        (
            f"answer leakage  {leakage}",
            style.muted("answer leakage  ")
            + (style.success(leakage) if intervention.leakage < 0.18 else style.danger(leakage)),
        ),
        (
            f"grounding  {grounding}",
            style.muted("grounding  ")
            + (style.success(grounding) if intervention.grounding >= 0.40
               else style.warning(grounding)),
        ),
    ]
    if intervention.regenerations:
        replaced = f"{intervention.regenerations} rung(s) replaced by the gate"
        gate.append((replaced, style.warning(replaced)))
    out += inline(style, gate)
    if intervention.citations:
        out += paragraph(
            style, "grounded in  " + ", ".join(intervention.citations),
            indent="  ", colour="muted",
        )
    return out


def render_telemetry(style: Style, result: TutoringResult) -> List[str]:
    telemetry = result.telemetry
    pairs = [
        ("backend", style.paint(telemetry.backend, "text")),
        ("model calls", str(telemetry.model_calls)),
        ("wall clock", f"{telemetry.total_seconds * 1000:.0f} ms"),
    ]
    if telemetry.stages:
        slowest = max(telemetry.stages, key=lambda kv: kv[1])
        pairs.append(("slowest stage", f"{slowest[0]} ({slowest[1] * 1000:.0f} ms)"))
    out = key_values(style, pairs, indent="  ")
    for warning in telemetry.warnings:
        out += bullet(style, style.warning(warning), indent="  ")
    return out


# studies
def render_benchmark(style: Style, payload: Dict[str, object]) -> List[str]:
    out = banner(style, "RQ1 · Diagnostic accuracy", "Socratic Evaluator against expert labels")
    out.append("")
    metrics = [
        ("bank · mode", f"{payload.get('bank', 'dev')} · {payload.get('mode', 'full')}"),
        ("backend", str(payload.get("backend", "offline"))
         + (style.danger(f"  ({payload['degraded_rows']} rows degraded)")
            if payload.get("degraded_rows") else "")),
        ("labelled submissions", str(payload["n_labelled"])),
        ("clean submissions", str(payload["n_clean"])),
        ("top-1 accuracy", _metric(style, float(payload["top1_accuracy"]))),
        ("recall (any gold label)", _metric(style, float(payload["recall_any"]))),
        ("micro F1", _metric(style, float(payload["micro_f1"]))),
        ("macro F1", _metric(style, float(payload["macro_f1"]))),
        ("Cohen's kappa", _metric(style, float(payload["cohen_kappa"]))),
        ("false-positive rate", _metric(style, 1.0 - float(payload["false_positive_rate"]),
                                        shown=float(payload["false_positive_rate"]))),
        ("divergence localised", _metric(style, float(payload["divergence_localisation_rate"]))),
        ("test-runner baseline", _metric(style, float(payload["baseline_top1_accuracy"]))),
        ("mean latency", f"{float(payload['mean_seconds_per_submission']) * 1000:.0f} ms"),
    ]
    out += key_values(style, metrics, indent="  ")
    out.append("")
    per_label = payload.get("per_label") or {}
    if per_label:
        rows = [
            [label, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", str(v["support"])]
            for label, v in sorted(per_label.items(), key=lambda kv: (-kv[1]["support"], kv[0]))
        ]
        out += table(
            style,
            [Column("misconception"), Column("P", "right", 5), Column("R", "right", 5),
             Column("F1", "right", 5), Column("n", "right", 3)],
            rows,
            indent="  ",
        )
        out.append("")
    out += paragraph(style, str(payload["caveat"]), indent="  ", colour="warning")
    return out


def render_experiment(style: Style, payload: Dict[str, object]) -> List[str]:
    out = banner(style, "RQ2 · Intervention effectiveness", "Randomised comparison of tutoring arms")
    out.append("")
    out += paragraph(style, str(payload["disclaimer"]), indent="  ", colour="warning")
    out.append("")

    arms = payload["arms"]
    rows = []
    for name, summary in arms.items():
        rows.append([
            style.paint(name, "text", bold=name.startswith("treatment")),
            str(int(summary["n"])),
            f"{summary['retention_mean']:.3f} ± {summary['retention_sd']:.3f}",
            f"{summary['repeat_error_mean']:.3f}",
            bar(style, summary["on_target_rate"], width=10) + f" {summary['on_target_rate']:.0%}",
            f"{summary['in_zpd_rate']:.0%}",
            (style.success("0.00") if summary["max_leakage"] < 0.18
             else style.danger(f"{summary['max_leakage']:.2f}")),
        ])
    out += table(
        style,
        [Column("arm"), Column("n", "right", 4), Column("retention"), Column("repeat err", "right"),
         Column("on target"), Column("in ZPD", "right"), Column("max leak", "right")],
        rows,
        indent="  ",
    )
    out.append("")

    comparisons = payload["comparisons"]
    if comparisons:
        rows = [
            [c["contrast"], c["measure"], c["welch"], c["hedges_g"], c["cliffs_delta"]]
            for c in comparisons
        ]
        out += table(
            style,
            [Column("contrast"), Column("measure"), Column("Welch t"), Column("Hedges g"),
             Column("Cliff's delta")],
            rows,
            indent="  ",
        )
        out.append("")

    corrections = payload["corrections"]
    if corrections:
        out.append("  " + style.muted("Holm-Bonferroni family-wise correction", bold=True))
        rows = [
            [c["hypothesis"], f"{c['p']:.4f}", f"{c['p_adjusted']:.4f}",
             style.success("reject H0") if c["rejected"] else style.muted("retain H0")]
            for c in corrections
        ]
        out += table(
            style,
            [Column("hypothesis"), Column("p", "right", 8), Column("p adj", "right", 8), Column("decision")],
            rows,
            indent="  ",
        )
        out.append("")

    power = payload.get("power") or {}
    if power:
        out += key_values(
            style,
            [
                ("n per arm", str(int(power.get("n_per_arm", 0)))),
                ("observed effect", f"{power.get('observed_effect', 0.0):+.3f}"),
                ("achieved power", _metric(style, float(power.get("achieved_power", 0.0)))),
                ("minimum detectable effect", f"{power.get('minimum_detectable_effect', 0.0):.3f}"),
            ],
            indent="  ",
        )
    return out


def render_robustness(style: Style, payload: Dict[str, object]) -> List[str]:
    out = banner(style, "RQ3 · Robustness", "Diagnosis under semantics-preserving mutation")
    out.append("")
    overall = payload["overall"]
    out += key_values(
        style,
        [
            ("verified mutants", str(int(overall.get("n", 0)))),
            ("top-1 stability", _metric(style, float(overall.get("top1_stability", 0.0)))),
            ("mean finding overlap", _metric(style, float(overall.get("mean_overlap", 0.0)))),
        ],
        indent="  ",
    )
    out.append("")
    rows = [
        [name, str(int(stats["n"])),
         bar(style, stats["top1_stability"], width=12) + f" {stats['top1_stability']:.0%}",
         f"{stats['mean_overlap']:.2f}"]
        for name, stats in (payload["by_transform"] or {}).items()
    ]
    out += table(
        style,
        [Column("transformation"), Column("n", "right", 4), Column("top-1 stability"),
         Column("overlap", "right", 8)],
        rows,
        indent="  ",
    )
    regressions = payload.get("regressions") or []
    if regressions:
        out.append("")
        out.append("  " + style.muted("diagnoses that shifted", bold=True))
        for row in regressions[:8]:
            out += bullet(
                style,
                f"{row['submission']} under {row['transform']}: "
                f"{row['original']} → {row['mutant']}",
                indent="  ",
            )
    return out


def render_concept(style: Style, graph: KnowledgeGraph, cid: str, *, depth: int = 3) -> List[str]:
    concept = graph.concept(cid)
    out = banner(style, concept.name, f"{cid}  ·  {concept.stratum}  ·  depth {graph.depth(cid)}")
    out.append("")
    out += paragraph(style, concept.summary)
    out.append("")
    out.append("  " + style.muted("prerequisites", bold=True))
    if graph.prerequisites(cid):
        out += tree(
            style,
            cid,
            graph.prerequisites,
            label=lambda node: style.paint(graph.name(node), "accent" if node == cid else "text")
            + style.muted(f"  ({node})"),
            depth=depth,
            indent="  ",
        )
    else:
        out.append("    " + style.muted("none — this is a foundation concept"))
    dependents = graph.dependents(cid)
    if dependents:
        out.append("")
        out.append("  " + style.muted("unlocks", bold=True))
        out.append("    " + style.muted(", ").join(style.paint(graph.name(d), "text") for d in dependents))
    return out


def _metric(style: Style, value: float, *, shown: Optional[float] = None) -> str:
    displayed = value if shown is None else shown
    return bar(style, value, width=14) + "  " + style.paint(f"{displayed:.3f}", "text")


# interactive session
def render_hint(style: Style, hint, *, index: int, total: int, graph: KnowledgeGraph) -> List[str]:
    """One rung of the ladder on its own.
    Rungs are shown one at a time on purpose, printing the whole ladder is
    just a slow worked answer.
    """
    out = [
        "  "
        + style.paint(f"hint {index}/{total}", "primary", bold=True)
        + style.muted(f"  ·  {hint.kind}")
        + (
            style.muted(f"  ·  {graph.name(hint.concept_id)}")
            if hint.concept_id and hint.concept_id in graph
            else ""
        )
    ]
    out.append("")
    out += paragraph(style, hint.text, indent="  ")
    out.append("")
    if index < total:
        out.append("  " + style.muted("sit with that one first — `hint` again when you are stuck"))
    else:
        out.append("  " + style.muted("that is the last rung — try `challenge`, then `retry` when you have edited"))
    return out


def render_verdict(style: Style, result: TutoringResult) -> List[str]:
    """One line headline printed after every diagnosis in a session."""
    diagnosis = result.diagnosis
    checks = diagnosis.check_summary
    if not diagnosis.hits:
        return [
            "  "
            + badge(style, "NO FINDING", "success")
            + "  "
            + style.muted(
                f"{checks}; no structural warning sign either"
                if diagnosis.checks_total
                else "no structural warning sign"
            )
        ]
    primary = diagnosis.primary
    entry = MISCONCEPTIONS.get(primary.misconception_id)
    label = "PROVISIONAL" if diagnosis.inconclusive else "DIAGNOSIS"
    colour = "warning" if diagnosis.inconclusive else "primary"
    out = [
        "  "
        + badge(style, label, colour)
        + "  "
        + style.paint(entry.name if entry else primary.misconception_id, "text", bold=True)
        + style.muted(f"  ·  {primary.belief:.2f}")
        + (style.muted(f"  ·  {primary.span}") if primary.span else "")
    ]
    if diagnosis.checks_total:
        out.append("  " + style.muted(checks))
    return out


def render_attempt_diff(
    style: Style,
    previous: TutoringResult,
    current: TutoringResult,
    graph: KnowledgeGraph,
) -> List[str]:
    """What changed between two attempts at the same task.
    A tutor that cannot tell whether the belief was repaired (vs the symptom
    patched) is back to marking answers.
    """
    before = {hit.misconception_id for hit in previous.diagnosis.hits}
    after = {hit.misconception_id for hit in current.diagnosis.hits}

    out: List[str] = []
    for mid in sorted(before - after):
        entry = MISCONCEPTIONS.get(mid)
        out.append(
            "  " + style.success("resolved  ") + style.paint(entry.name if entry else mid, "text")
        )
    for mid in sorted(before & after):
        entry = MISCONCEPTIONS.get(mid)
        out.append(
            "  " + style.warning("persists  ") + style.paint(entry.name if entry else mid, "text")
        )
    for mid in sorted(after - before):
        entry = MISCONCEPTIONS.get(mid)
        out.append(
            "  " + style.danger("new       ") + style.paint(entry.name if entry else mid, "text")
        )
    if not out:
        out.append("  " + style.muted("no change in the findings"))

    if previous.diagnosis.checks_total and current.diagnosis.checks_total:
        was = previous.diagnosis.checks_passed / previous.diagnosis.checks_total
        now = current.diagnosis.checks_passed / current.diagnosis.checks_total
        arrow = style.success("→") if now > was else style.muted("→")
        out.append(
            "  "
            + style.muted("checks    ")
            + style.muted(previous.diagnosis.check_summary)
            + f" {arrow} "
            + style.paint(current.diagnosis.check_summary, "text")
        )

    # report on the concept the learner was working on (previous attempt's
    # target), not whatever the new diagnosis points at
    focus = (
        previous.intervention.target_concept
        or previous.alignment.root_cause
        or current.intervention.target_concept
    )
    if focus and focus in previous.mastery_posterior and focus in current.mastery_posterior:
        was = previous.mastery_posterior[focus]
        now = current.mastery_posterior[focus]
        arrow = style.success("↑") if now > was + 1e-6 else (
            style.danger("↓") if now < was - 1e-6 else style.muted("·")
        )
        out.append(
            "  "
            + style.muted("mastery   ")
            + style.accent(graph.name(focus) if focus in graph else focus)
            + style.muted(f"  {was:.2f} ")
            + arrow
            + style.paint(f" {now:.2f}", "text")
        )
    return out


def render_mastery(
    style: Style,
    posterior: Dict[str, float],
    graph: KnowledgeGraph,
    *,
    threshold: float = 0.62,
    limit: int = 8,
) -> List[str]:
    """The learner's weakest concepts as the session currently sees them."""
    if not posterior:
        return [
            "  " + style.muted("nothing tracked yet — diagnose a submission first")
        ]
    ranked = sorted(posterior.items(), key=lambda kv: (kv[1], kv[0]))[:limit]
    rows = [
        [
            graph.name(cid) if cid in graph else cid,
            bar(style, value, width=14) + f" {value:.2f}",
            style.danger("not yet") if value < threshold else style.success("secure"),
        ]
        for cid, value in ranked
    ]
    out = table(
        style,
        [Column("concept", min_width=20, priority=100), Column("mastery", min_width=20),
         Column("status", priority=40)],
        rows,
        indent="  ",
    )
    unmastered = sum(1 for value in posterior.values() if value < threshold)
    out.append("")
    out.append(
        "  " + style.muted(f"{unmastered} of {len(posterior)} concepts below the mastery threshold")
    )
    return out


def render_progress(style: Style, attempts: Sequence, graph: KnowledgeGraph) -> List[str]:
    """Short history of the session's attempts."""
    if not attempts:
        return ["  " + style.muted("no attempts yet")]
    rows = []
    for attempt in attempts:
        diagnosis = attempt.result.diagnosis
        primary = diagnosis.primary
        rows.append([
            str(attempt.index),
            attempt.label,
            (style.success("clean") if not primary
             else style.paint(primary.misconception_id, "text")),
            f"{primary.belief:.2f}" if primary else "—",
            diagnosis.check_summary if diagnosis.checks_total else "—",
        ])
    return table(
        style,
        [Column("#", "right", 3), Column("source", min_width=14),
         Column("leading finding", min_width=24, priority=100),
         Column("belief", "right", 7), Column("checks", priority=40)],
        rows,
        indent="  ",
    )
