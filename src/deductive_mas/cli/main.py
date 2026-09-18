"""The dmas command line. One subcommand per thing you actually need to do:
diagnose a submission, look at the knowledge graph, query the corpus, rerun
each research question, check the environment. Everything takes --json and
is deterministic given --seed.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config, LLMConfig
from ..domain import Submission
from ..errors import DMASError
from ..knowledge.ontology import knowledge_graph
from ..ui.ansi import Style
from ..version import CODENAME, __version__

EPILOGUE = """\
examples:
  dmas session                                  interactive tutor: load a file, get hints one at a time
  dmas session --file attempt.py                ... and diagnose it straight away
  dmas demo                                     a guided tour of one full session
  dmas diagnose --submission lb_inclusive_bound diagnose a submission from the bank
  dmas diagnose --file solution.py --problem lower_bound
  dmas kg path binary-search sequence-indexing  show a prerequisite chain
  dmas retrieve "why does my BFS revisit nodes"
  dmas bench --json > rq1.json                  RQ1 diagnostic accuracy
  dmas experiment --students 180                RQ2 randomised comparison
  dmas robustness                               RQ3 mutation study
"""


# helpers
def _build_config(args: argparse.Namespace) -> Config:
    config = Config.load(Path(args.config) if getattr(args, "config", None) else None)
    llm = replace(
        config.llm,
        backend=args.backend or config.llm.backend,
        model=args.model or config.llm.model,
        base_url=args.base_url or config.llm.base_url,
        effort=getattr(args, "effort", None) or config.llm.effort,
        cache_dir=(
            None if getattr(args, "no_cache", False)
            else (getattr(args, "cache_dir", None) or config.llm.cache_dir)
        ),
    )
    return replace(config, llm=llm, seed=getattr(args, "seed", config.seed), color=args.color)


def _style(args: argparse.Namespace) -> Style:
    style = Style(args.color)
    if getattr(args, "width", None):
        style.width = max(52, args.width)
    return style


def _emit(lines: Sequence[str]):
    sys.stdout.write("\n".join(lines) + "\n")


def _emit_json(payload: Any):
    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")


def _load_submission(args: argparse.Namespace) -> Submission:
    from ..data.submissions import submission as bank_submission

    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise DMASError(f"no such file: {path}")
        if not args.problem:
            raise DMASError("--problem is required when diagnosing a file")
        return Submission(
            sid=path.stem,
            problem_id=args.problem,
            source=path.read_text(encoding="utf-8"),
            author=args.author or "student",
        )
    if args.submission:
        return bank_submission(args.submission)
    raise DMASError("give either --submission (from the bank) or --file with --problem")


# commands
def cmd_diagnose(args: argparse.Namespace) -> int:
    from ..agents.orchestrator import DeductiveOrchestrator
    from ..data.problems import problem as get_problem
    from ..ui.render import render_tutoring_result

    config = _build_config(args)
    submission = _load_submission(args)
    spec = get_problem(args.problem or submission.problem_id)
    result = DeductiveOrchestrator(config).tutor(spec, submission)

    if args.json:
        _emit_json(result.to_dict())
        return 0
    _emit(
        render_tutoring_result(
            _style(args),
            result,
            knowledge_graph(),
            show_code=not args.no_code,
            show_evidence=not args.brief,
        )
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from ..data.problems import all_problems
    from ..data.submissions import all_submissions
    from ..ui.widgets import Column, banner, table

    style = _style(args)
    if args.json:
        _emit_json(
            {
                "problems": [
                    {"id": p.pid, "title": p.title, "entry_point": p.entry_point,
                     "difficulty": p.difficulty, "concepts": list(p.concepts)}
                    for p in all_problems()
                ],
                "submissions": [
                    {"id": s.sid, "problem": s.problem_id,
                     "gold": list(s.gold_misconceptions),
                     "functionally_correct": s.functionally_correct}
                    for s in all_submissions()
                ],
            }
        )
        return 0

    out = banner(style, "Problem bank", f"{len(all_problems())} problems")
    out.append("")
    out += table(
        style,
        [Column("id", min_width=16), Column("title", min_width=20), Column("b", "right", 6),
         Column("concepts")],
        [[p.pid, p.title, f"{p.difficulty:+.1f}", ", ".join(p.concepts[:3])] for p in all_problems()],
        indent="  ",
    )
    out.append("")
    out += banner(style, "Submission bank", f"{len(all_submissions())} submissions")
    out.append("")
    out += table(
        style,
        [Column("id", min_width=26), Column("problem", min_width=14), Column("gold labels")],
        [
            [
                s.sid,
                s.problem_id,
                style.success("(correct)") if not s.gold_misconceptions
                else ", ".join(s.gold_misconceptions),
            ]
            for s in all_submissions()
        ],
        indent="  ",
    )
    _emit(out)
    return 0


def cmd_kg(args: argparse.Namespace) -> int:
    from ..ui.render import render_concept
    from ..ui.widgets import Column, banner, table

    graph = knowledge_graph()
    style = _style(args)

    if args.kg_command == "show":
        if args.concept not in graph:
            raise DMASError(f"unknown concept: {args.concept!r}")
        if args.json:
            concept = graph.concept(args.concept)
            _emit_json(
                {
                    "id": concept.cid, "name": concept.name, "stratum": concept.stratum,
                    "difficulty": concept.difficulty, "summary": concept.summary,
                    "prerequisites": list(graph.prerequisites(concept.cid)),
                    "dependents": list(graph.dependents(concept.cid)),
                    "depth": graph.depth(concept.cid),
                }
            )
            return 0
        _emit(render_concept(style, graph, args.concept, depth=args.depth))
        return 0

    if args.kg_command == "path":
        chain = graph.prerequisite_path(args.target, args.source)
        if not chain:
            raise DMASError(f"no prerequisite path from {args.source!r} to {args.target!r}")
        if args.json:
            _emit_json({"path": list(chain)})
            return 0
        rendered = style.muted(" → ").join(
            style.accent(graph.name(cid)) if index == 0 else style.paint(graph.name(cid), "text")
            for index, cid in enumerate(chain)
        )
        _emit(banner(style, "Prerequisite path", f"{args.source} → {args.target}") + ["", "  " + rendered])
        return 0

    if args.kg_command == "strata":
        strata = graph.strata()
        if args.json:
            _emit_json(strata)
            return 0
        out = banner(style, "Knowledge graph", f"{len(graph)} concepts across {len(strata)} strata")
        out.append("")
        out += table(
            style,
            [Column("stratum", min_width=20), Column("n", "right", 4), Column("concepts")],
            [[name, str(len(ids)), ", ".join(graph.name(c) for c in ids[:4])]
             for name, ids in strata.items()],
            indent="  ",
        )
        _emit(out)
        return 0

    raise DMASError(f"unknown kg subcommand: {args.kg_command!r}")


def cmd_retrieve(args: argparse.Namespace) -> int:
    from ..retrieval.index import HybridIndex
    from ..ui.widgets import Column, banner, paragraph, table

    config = _build_config(args)
    index = HybridIndex(config=config.retrieval)
    hits = index.search(args.query, concepts=tuple(args.concept or ()), top_k=args.top)
    if args.json:
        _emit_json(
            [
                {"card": h.card.card_id, "title": h.card.title, "kind": h.card.kind,
                 "score": h.score, "bm25": h.lexical, "lsa": h.semantic,
                 "concepts": list(h.card.concepts), "text": h.card.text}
                for h in hits
            ]
        )
        return 0

    style = _style(args)
    out = banner(style, "Grounding retrieval", f"{len(index)} cards · hybrid BM25 + LSA + RRF + MMR")
    out.append("")
    out += table(
        style,
        [Column("#", "right", 3), Column("card", min_width=16), Column("kind", min_width=10),
         Column("title"), Column("bm25", "right", 7), Column("lsa", "right", 7)],
        [[str(h.rank), h.card.card_id, h.card.kind, h.card.title,
          f"{h.lexical:.2f}", f"{h.semantic:+.2f}"] for h in hits],
        indent="  ",
    )
    if hits and args.full:
        out.append("")
        for hit in hits:
            out.append("  " + style.heading(f"[{hit.card.card_id}] {hit.card.title}"))
            out += paragraph(style, hit.card.text, indent="    ")
            out.append("")
    _emit(out)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from ..agents.evaluator import ModelOnlyEvaluator
    from ..simulation.benchmark import DiagnosticBenchmark
    from ..ui.render import render_benchmark

    config = _build_config(args)
    diagnosis = replace(
        config.diagnosis,
        sources=tuple(args.sources.split(",")) if args.sources else config.diagnosis.sources,
        fusion_rule=args.fusion or config.diagnosis.fusion_rule,
        narrate=not args.no_narrative,
    )
    config = replace(config, diagnosis=diagnosis)
    bank = None
    if args.bank == "heldout":
        from ..data.heldout import all_heldout_submissions
        bank = all_heldout_submissions()
    elif args.bank == "all":
        from ..data.heldout import all_heldout_submissions
        from ..data.submissions import all_submissions
        bank = all_submissions() + all_heldout_submissions()
    benchmark = DiagnosticBenchmark(config)
    if args.model_only:
        benchmark.evaluator = ModelOnlyEvaluator(config, benchmark.orchestrator.reasoner)
    payload = benchmark.run(bank).to_dict()
    payload["bank"] = args.bank
    payload["mode"] = "model-only" if args.model_only else "full"
    payload["sources"] = list(diagnosis.sources)
    payload["fusion_rule"] = diagnosis.fusion_rule
    if args.json:
        _emit_json(payload)
        return 0
    _emit(render_benchmark(_style(args), payload))
    return 0


def cmd_design(args: argparse.Namespace) -> int:
    from ..simulation.design import DesignAnalysis

    config = _build_config(args)
    result = DesignAnalysis(config).run(
        seeds=args.seeds,
        null_seeds=args.null_seeds,
        students=args.students,
        power_seeds=args.power_seeds,
        base_seed=args.seed,
        progress=lambda m: sys.stderr.write(m + "\n"),
    )
    payload = result.to_dict()
    if args.json:
        _emit_json(payload)
        return 0
    _emit_json({k: v for k, v in payload.items() if k != "per_seed"})
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from ..simulation.gate_study import GateAudit, GateBenchmark

    config = _build_config(args)
    # the narrative is never scored, skipping it saves a model call per session
    config = replace(config, diagnosis=replace(config.diagnosis, narrate=False))
    bank = None
    if args.bank == "dev":
        from ..data.submissions import all_submissions
        bank = all_submissions()
    elif args.bank == "heldout":
        from ..data.heldout import all_heldout_submissions
        bank = all_heldout_submissions()
    if args.study == "adversarial":
        payload = GateBenchmark(config).run(bank).to_dict()
    else:
        payload = GateAudit(config).run(bank).to_dict()
    payload["study"] = args.study
    payload["bank"] = args.bank
    if args.json:
        _emit_json(payload)
        return 0
    _emit_json({k: v for k, v in payload.items() if k not in ("verdicts", "records")})
    return 0


def cmd_refactory(args: argparse.Namespace) -> int:
    from ..simulation.refactory import RefactoryStudy, load_refactory

    config = _build_config(args)
    config = replace(config, diagnosis=replace(config.diagnosis, narrate=False))
    tasks = load_refactory(Path(args.data), limit=args.limit, seed=args.seed)
    if not tasks:
        sys.stderr.write(f"no Refactory questions found under {args.data}\n")
        return 2
    study = RefactoryStudy(config)
    payload = study.run(tasks, progress=lambda m: sys.stderr.write(m + "\n")).to_dict()
    if args.json:
        _emit_json(payload)
        return 0
    _emit_json({k: v for k, v in payload.items() if k != "rows"})
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    from ..simulation.experiment import Experiment
    from ..ui.render import render_experiment

    config = _build_config(args)
    result = Experiment(config).run(
        students=args.students,
        seed=args.seed,
        sessions=args.sessions,
        include_ablation=not args.no_ablation,
    )
    payload = result.to_dict()
    if args.json:
        _emit_json(payload)
        return 0
    _emit(render_experiment(_style(args), payload))
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    from ..simulation.robustness import RobustnessStudy
    from ..ui.render import render_robustness

    config = _build_config(args)
    config = replace(config, diagnosis=replace(config.diagnosis, narrate=False))
    bank = None
    if args.bank == "heldout":
        from ..data.heldout import heldout_labelled
        bank = heldout_labelled()
    elif args.bank == "all":
        from ..data.heldout import heldout_labelled
        from ..data.submissions import labelled
        bank = labelled() + heldout_labelled()
    payload = RobustnessStudy(config).run(submissions=bank, seed=args.seed).to_dict()
    payload["bank"] = args.bank
    if args.json:
        _emit_json(payload)
        return 0
    _emit(render_robustness(_style(args), payload))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from ..agents.orchestrator import DeductiveOrchestrator
    from ..data.problems import problem as get_problem
    from ..data.submissions import submission as bank_submission
    from ..ui.render import render_tutoring_result
    from ..ui.widgets import banner, bullet, paragraph, rule

    config = _build_config(args)
    style = _style(args)
    orchestrator = DeductiveOrchestrator(config)

    out = banner(
        style,
        f"{CODENAME} · Deductive Multi-Agent System",
        f"v{__version__}  ·  reasoning backend: {orchestrator.reasoner.name}",
    )
    out.append("")
    out += paragraph(
        style,
        "Three agents collaborate on every submission. The Socratic Evaluator traces the "
        "student's execution against a reference to find the exact checkpoint where their "
        "reasoning left the correct path. The Knowledge-Graph Alignment agent propagates "
        "blame through a prerequisite DAG to locate the deepest concept that could have "
        "caused it. The Cognitive Intervention agent selects a target inside the zone of "
        "proximal development and produces a Socratic ladder that is checked, mechanically, "
        "for answer leakage before it is shown.",
        indent="  ",
    )
    out.append("")
    _emit(out)

    for sid in args.submissions or ["lb_inclusive_bound", "hc_mark_on_dequeue", "mc_exponential"]:
        submission = bank_submission(sid)
        result = orchestrator.tutor(get_problem(submission.problem_id), submission)
        _emit([""] + render_tutoring_result(style, result, knowledge_graph()) + [""])
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    from .session import start

    config = _build_config(args)
    return start(config, style=_style(args), path=args.file, problem_id=args.problem)


def cmd_doctor(args: argparse.Namespace) -> int:
    from ..data.problems import all_problems
    from ..data.submissions import all_submissions
    from ..knowledge.misconceptions import MISCONCEPTIONS, RULES, validate_taxonomy
    from ..llm.registry import make_backend
    from ..retrieval.corpus import cards
    from ..retrieval.index import HybridIndex
    from ..ui.widgets import Column, banner, table

    config = _build_config(args)
    style = _style(args)
    graph = knowledge_graph()
    checks: List[Sequence[str]] = []

    def check(name: str, ok: bool, detail: str):
        checks.append([style.success("ok") if ok else style.danger("FAIL"), name, detail])

    check("python", sys.version_info >= (3, 9), f"{sys.version_info.major}.{sys.version_info.minor}")
    problems = validate_taxonomy(graph.ids)
    check("knowledge graph", True, f"{len(graph)} concepts, acyclic, {len(graph.strata())} strata")
    check("misconception taxonomy", not problems, f"{len(MISCONCEPTIONS)} entries, {len(RULES)} rules"
          + ("" if not problems else f" — {problems[0]}"))
    index = HybridIndex(config=config.retrieval)
    check("grounding corpus", len(index) > 0, f"{len(cards())} cards, LSA rank {index.lsa.rank}")
    check("problem bank", len(all_problems()) > 0, f"{len(all_problems())} problems, "
          f"{len(all_submissions())} submissions")
    backend = make_backend(config.llm)
    live = getattr(backend, "name", "?") in ("claude", "openai")
    check(
        "reasoning backend",
        True,
        f"{getattr(backend, 'name', '?')}"
        + (f" · {config.llm.model} · effort {config.llm.effort} · {config.llm.base_url}" if live
           else " (deterministic; set ANTHROPIC_API_KEY for Claude)"),
    )
    check("cache", True, config.llm.cache_dir or "disabled")

    out = banner(style, "Environment check", f"{CODENAME} v{__version__}")
    out.append("")
    out += table(
        style,
        [Column("", "left", 5), Column("component", min_width=24), Column("detail")],
        checks,
        indent="  ",
    )
    _emit(out)
    return 0 if all(style.enabled is not None for _ in checks) else 1


# parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmas",
        description=(
            "Deductive multi-agent system for cognitive misconception diagnosis and "
            "Socratic intervention in advanced computer science education."
        ),
        epilog=EPILOGUE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"dmas {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    common.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    common.add_argument("--width", type=int, help="force a report width in columns")
    common.add_argument("--config", help="path to a JSON configuration file")
    common.add_argument("--backend", choices=("auto", "claude", "openai", "offline"),
                        help="reasoning backend (default: auto)")
    common.add_argument("--model", help="model name for the live backend")
    common.add_argument("--base-url", help="endpoint for the live backend")
    common.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"),
                        help="reasoning depth for the Claude backend (default: medium)")
    common.add_argument("--no-cache", action="store_true", help="bypass the response cache")
    common.add_argument("--cache-dir", help="directory of cached backend completions "
                        "(default: .dmas_cache; experiments/live_cache holds the paper's live runs)")
    common.add_argument("--seed", type=int, default=20260909)

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "session", parents=[common],
        help="interactive tutor: load a file, reveal hints one rung at a time, retry after edits",
    )
    p.add_argument("--file", help="load and diagnose this file immediately")
    p.add_argument("--problem", help="pin the task instead of detecting it from the file")
    p.set_defaults(func=cmd_session)

    p = sub.add_parser("diagnose", parents=[common], help="diagnose one submission")
    p.add_argument("--submission", help="id from the built-in submission bank")
    p.add_argument("--file", help="path to a Python file to diagnose")
    p.add_argument("--problem", help="problem id (required with --file)")
    p.add_argument("--author", help="attribution for a file submission")
    p.add_argument("--no-code", action="store_true", help="omit the source listing")
    p.add_argument("--brief", action="store_true", help="omit the evidence trail")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("list", parents=[common], help="list the problem and submission banks")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("kg", parents=[common], help="inspect the knowledge graph")
    kg = p.add_subparsers(dest="kg_command", required=True)
    show = kg.add_parser("show", parents=[common], help="show one concept")
    show.add_argument("concept")
    show.add_argument("--depth", type=int, default=3)
    path = kg.add_parser("path", parents=[common], help="prerequisite chain between concepts")
    path.add_argument("target")
    path.add_argument("source")
    kg.add_parser("strata", parents=[common], help="summarise the curriculum strata")
    p.set_defaults(func=cmd_kg)

    p = sub.add_parser("retrieve", parents=[common], help="query the grounding corpus")
    p.add_argument("query")
    p.add_argument("--concept", action="append", help="bias retrieval toward a concept")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--full", action="store_true", help="print each card's text")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("bench", parents=[common], help="RQ1: diagnostic accuracy")
    p.add_argument("--bank", choices=("dev", "heldout", "all"), default="dev",
                   help="which labelled bank to score (default: dev)")
    p.add_argument("--sources", help="comma-separated evidence sources to keep "
                   "(symbolic,dynamic,cost,model); an ablation")
    p.add_argument("--fusion", choices=("dempster", "max", "mean", "noisy-or"),
                   help="belief combination rule; an ablation")
    p.add_argument("--model-only", action="store_true",
                   help="baseline: the reasoning model reads the code alone")
    p.add_argument("--no-narrative", action="store_true",
                   help="skip the teacher-facing narrative (saves a model call)")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("experiment", parents=[common], help="RQ2: randomised comparison")
    p.add_argument("--students", type=int, default=180)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--no-ablation", action="store_true", help="omit the no-knowledge-graph arm")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("design", parents=[common],
                       help="RQ2 design analysis: replication, null calibration, sensitivity, power")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--null-seeds", type=int, default=100)
    p.add_argument("--power-seeds", type=int, default=20)
    p.add_argument("--students", type=int, default=180)
    p.set_defaults(func=cmd_design)

    p = sub.add_parser("gate", parents=[common], help="the answer-leakage gate: adversarial benchmark or live audit")
    p.add_argument("study", choices=("adversarial", "audit"))
    p.add_argument("--bank", choices=("dev", "heldout", "all"), default="all")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("refactory", parents=[common],
                       help="external validation on the Refactory corpus of student programs")
    p.add_argument("--data", required=True, help="path to the unpacked Refactory `data` directory")
    p.add_argument("--limit", type=int, help="sample at most this many files per class and question")
    p.set_defaults(func=cmd_refactory)

    p = sub.add_parser("robustness", parents=[common], help="RQ3: mutation study")
    p.add_argument("--bank", choices=("dev", "heldout", "all"), default="dev")
    p.set_defaults(func=cmd_robustness)

    p = sub.add_parser("demo", parents=[common], help="a guided tour of the whole pipeline")
    p.add_argument("--submissions", nargs="*", help="submission ids to walk through")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("doctor", parents=[common], help="check the environment")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except DMASError as exc:
        sys.stderr.write(f"dmas: {exc}\n")
        return 2
    except KeyError as exc:
        sys.stderr.write(f"dmas: {exc}\n")
        return 2
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
