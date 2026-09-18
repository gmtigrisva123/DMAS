"""Write numbers.tex: every number quoted in the paper as a LaTeX macro.

    python3 experiments/make_numbers.py

The paper never types a result by hand, it says \heldTopOne and this fills
it from results/. Missing input -> '--' so the paper still compiles while a
study is running.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT.parent / "deductive_multi_agent_system_(MAS)_paper" / "numbers.tex"

macros: dict = {}


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def pct(x, digits=0):
    return "--" if x is None else f"{100 * x:.{digits}f}"


def num(x, digits=2):
    return "--" if x is None else f"{x:.{digits}f}"


def define(name, value):
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(name)
    macros[name] = str(value)


# RQ1
def bench(name, prefix):
    d = load(name)
    ok = d is not None and d.get("degraded_rows", 0) == 0
    define(prefix + "TopOne", num(d["top1_accuracy"]) if ok else "--")
    define(prefix + "Recall", num(d["recall_any"]) if ok else "--")
    define(prefix + "MicroF", num(d["micro_f1"]) if ok else "--")
    define(prefix + "MacroF", num(d["macro_f1"]) if ok else "--")
    define(prefix + "MicroP", num(d["micro_precision"]) if ok else "--")
    define(prefix + "Kappa", num(d["cohen_kappa"]) if ok else "--")
    define(prefix + "FPR", num(d["false_positive_rate"]) if ok else "--")
    define(prefix + "Loc", num(d["divergence_localisation_rate"]) if ok else "--")
    define(prefix + "Ms", f"{1000 * d['mean_seconds_per_submission']:.0f}" if ok else "--")
    define(prefix + "Hits", str(sum(1 for r in d["rows"] if r["correct"])) if ok else "--")
    define(prefix + "N", str(d["n_labelled"]) if ok else "--")
    define(prefix + "Clean", str(d["n_clean"]) if ok else "--")
    return d if ok else None


bench("rq1_dev_full_offline.json", "dev")
bench("rq1_heldout_full_offline.json", "held")
bench("rq1_dev_full_gpt-oss-120b.json", "devModel")
bench("rq1_heldout_full_gpt-oss-120b.json", "heldModel")
bench("rq1_dev_modelonly_gpt-oss-120b.json", "devOnly")
bench("rq1_heldout_modelonly_gpt-oss-120b.json", "heldOnly")
bench("rq1_dev_full_gpt-oss-20b.json", "devSmall")
bench("rq1_heldout_full_gpt-oss-20b.json", "heldSmall")
bench("rq1_dev_modelonly_gpt-oss-20b.json", "devOnlySmall")
bench("rq1_heldout_modelonly_gpt-oss-20b.json", "heldOnlySmall")

for bank, prefix in (("dev", "dev"), ("heldout", "held")):
    for src, tag in (("symbolic", "Sym"), ("symbolic+cost", "SymCost"), ("dynamic", "Dyn"),
                     ("symbolic+dynamic", "SymDyn"), ("symbolic+dynamic+cost", "NoModel")):
        d = load(f"rq1_{bank}_sources_{src}_offline.json")
        define(prefix + "Abl" + tag, num(d["top1_accuracy"]) if d else "--")
    for rule, tag in (("max", "Max"), ("mean", "Mean"), ("noisy-or", "NoisyOr")):
        d = load(f"rq1_{bank}_fusion_{rule}_offline.json")
        define(prefix + "Fus" + tag, num(d["top1_accuracy"]) if d else "--")

# external
d = load("refactory_full_offline.json")
if d:
    ov = d["overall"]
    define("refFiles", f"{ov['n_files']:,}")
    define("refCorrect", f"{ov['n_correct']:,}")
    define("refWrong", f"{ov['n_wrong']:,}")
    define("refFalseAcc", pct(ov["correct__false_accusation_rate"], 1))
    define("refAnyFinding", pct(ov["correct__any_finding_rate"], 1))
    define("refCostOnly", pct(ov["correct__cost_or_hygiene_only_rate"], 1))
    define("refCEonAccepted", pct(ov["correct__counterexample_rate"], 1))
    define("refWrongCE", pct(ov["wrong__counterexample_rate"], 1))
    define("refWrongLoc", pct(ov["wrong__localised_rate"], 1))
    define("refWrongNamed", pct(ov["wrong__named_correctness_belief_rate"], 1))
    define("refWrongUnexplained", pct(ov["wrong__unexplained_failure_rate"], 1))
    define("refMs", f"{1000 * ov['mean_seconds_per_file']:.0f}")
    accused = round(ov["correct__false_accusation_rate"] * ov["n_correct"])
    define("refAccusedFiles", str(accused))
    for q in range(1, 6):
        v = d["by_question"][f"q{q}"]
        define(f"refQ{'ABCDE'[q-1]}FalseAcc", pct(v["correct__false_accusation_rate"], 1))
        define(f"refQ{'ABCDE'[q-1]}CE", pct(v["wrong__counterexample_rate"], 1))
        define(f"refQ{'ABCDE'[q-1]}Named", pct(v["wrong__named_correctness_belief_rate"], 1))

# gate
d = load("gate_adversarial_offline.json")
if d:
    s = d["summary"]
    for cat, tag in (("full-solution", "Full"), ("reference-line", "RefLine"), ("reference-line-renamed", "RefRenamed"),
                     ("prose-assignment", "ProseAssign"), ("prose-repair", "ProseRepair"), ("student-line", "StudentLine"),
                     ("question", "Question"), ("fallback-rung", "Fallback")):
        define("gate" + tag, pct(s[cat]["rejection_rate"], 1))
        define("gate" + tag + "N", str(s[cat]["n"]))
    define("gateLeakRecall", pct(s["_overall"]["leak_recall_excluding_prose"], 1))
    define("gateFalseReject", pct(s["_overall"]["legitimate_false_rejection_rate"], 1))
    leaks = [v for v in d["verdicts"] if v["should_reject"] and v["category"] != "prose-repair"]
    legit = [v for v in d["verdicts"] if not v["should_reject"]]
    define("gateLeakN", str(len(leaks)))
    define("gateLegitN", str(len(legit)))
    define("gateLeakMissed", str(sum(1 for v in leaks if not v["rejected"])))

for name, tag in (("gate_audit_offline.json", "Audit"), ("gate_audit_gpt-oss-120b.json", "AuditModel"),
                  ("gate_audit_gpt-oss-20b.json", "AuditSmall")):
    d = load(name)
    ok = d is not None and d.get("degraded_rows", 0) == 0
    hints = [r for r in d["records"] if r["item"].startswith("hint") and r["candidate"]] if ok else []
    chall = [r for r in d["records"] if r["item"] == "challenge" and r["candidate"]] if ok else []
    def share(rows, source):
        return pct(sum(1 for r in rows if r["final_source"] == source) / len(rows), 1) if rows else "--"
    define(tag + "HintN", str(len(hints)) if ok else "--")
    define(tag + "HintGenerated", share(hints, "generated"))
    define(tag + "HintTemplate", share(hints, "template"))
    define(tag + "HintSanitised", share(hints, "sanitised"))
    define(tag + "HintFallback", share(hints, "fallback"))
    define(tag + "HintLeak", pct(sum(1 for r in hints if (r["leakage"] or 0) > 0.18 or r["structural_match"]) / len(hints), 1) if hints else "--")
    define(tag + "HintGroundingOnly", pct(sum(1 for r in hints if not r["accepted"] and not ((r["leakage"] or 0) > 0.18 or r["structural_match"])) / len(hints), 1) if hints else "--")
    define(tag + "ChallengeGenerated", share(chall, "generated"))
    define(tag + "ChallengeLeak", pct(sum(1 for r in chall if (r["leakage"] or 0) > 0.18 or r["structural_match"]) / len(chall), 1) if chall else "--")
    define(tag + "MaxFinalLeak", num(max((r["final_leakage"] for r in d["records"]), default=0.0)) if ok else "--")

# design
d = load("rq2_design_offline.json")
if d and d.get("replication"):
    rep, null = d["replication"], d["null_calibration"]
    define("desSeeds", str(rep["seeds"]))
    define("desStudents", str(rep["students"]))
    define("desRetG", num(rep["retention_g"]["mean"]))
    define("desRetGlo", num(rep["retention_g"]["q05"]))
    define("desRetGhi", num(rep["retention_g"]["q95"]))
    define("desErrG", num(rep["repeat_error_g"]["mean"]))
    define("desErrGlo", num(rep["repeat_error_g"]["q05"]))
    define("desErrGhi", num(rep["repeat_error_g"]["q95"]))
    define("desFWER", pct(rep["family_wise_rejection_rate"], 0))
    define("desNullSeeds", str(null["seeds"]))
    define("desNullFWER", pct(null["family_wise_rejection_rate"], 1))
    define("desNullRetG", num(null["retention_g"]["mean"]))
    define("desOnTargetFull", pct(rep["on_target"]["treatment-mas"]["mean"], 0))
    define("desOnTargetAbl", pct(rep["on_target"]["ablation-no-kg"]["mean"], 0))
    define("desInZpdFull", pct(rep["in_zpd"]["treatment-mas"]["mean"], 0))
    define("desInZpdAbl", pct(rep["in_zpd"]["ablation-no-kg"]["mean"], 0))
    define("desRetControl", num(rep["retention_means"]["control-direct"]["mean"], 3))
    define("desRetFull", num(rep["retention_means"]["treatment-mas"]["mean"], 3))
    define("desRetAbl", num(rep["retention_means"]["ablation-no-kg"]["mean"], 3))
    define("desErrControl", num(rep["repeat_error_means"]["control-direct"]["mean"], 3))
    define("desErrFull", num(rep["repeat_error_means"]["treatment-mas"]["mean"], 3))
    define("desErrAbl", num(rep["repeat_error_means"]["ablation-no-kg"]["mean"], 3))
    for p in d["power"]:
        define("desPowerRet" + {60: "Twenty", 120: "Forty", 180: "Sixty", 240: "Eighty"}[p["students"]], pct(p["empirical_power_retention"], 0))
        define("desPowerErr" + {60: "Twenty", 120: "Forty", 180: "Sixty", 240: "Eighty"}[p["students"]], pct(p["empirical_power_repeat_error"], 0))
    sens0 = d["sensitivity"][0]
    define("desSensNullRetG", num(sens0["retention_g"]["mean"]))
    define("desSensNullFWER", pct(sens0["family_wise_rejection_rate"], 0))

# robustness
for bank, tag in (("dev", "rob"), ("heldout", "robHeld")):
    d = load(f"rq3_robustness_{bank}_offline.json")
    if d:
        define(tag + "Mutants", str(int(d["overall"]["n"])))
        define(tag + "Stability", num(d["overall"]["top1_stability"], 3))
        define(tag + "Overlap", num(d["overall"]["mean_overlap"], 3))
        define(tag + "Shifted", str(len(d["regressions"])))
        for t, k in (("rename-identifiers", "Rename"), ("all-combined", "All"), ("invert-conditionals", "Invert")):
            define(tag + k, num(d["by_transform"][t]["top1_stability"], 3))

# codebase
try:
    tests = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=str(ROOT),
                           env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True).stdout
    count = 0
    for line in tests.splitlines():
        m = re.match(r"^tests/\S+: (\d+)$", line)
        if m:
            count += int(m.group(1))
    define("nTests", str(count) if count else "--")
except Exception:
    define("nTests", "--")
loc = sum(len(p.read_text().splitlines()) for p in (ROOT / "src").rglob("*.py"))
define("nLoc", f"{loc:,}")

lines = ["% Generated by experiments/make_numbers.py — do not edit by hand."]
for name, value in sorted(macros.items()):
    lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(macros)} macros)")
