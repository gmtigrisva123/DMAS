"""Draw every figure in the paper from the json in results/.

    /usr/local/bin/python3 experiments/make_figures.py

Only needs matplotlib (the package itself has no deps, this script is the
one place a third party lib shows up). Figures with missing inputs are
skipped with a note so it can be rerun as results come in.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parent / "src"))

# palette (fixed categorical order, one sequential hue)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
GREY, INK, MUTED, GRID = "#8a8983", "#0b0b0b", "#52514e", "#dddcd7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

TEXT_WIDTH = 5.5


def load(name: str):
    path = RESULTS / name
    if not path.exists():
        print(f"  (missing {name})")
        return None
    with path.open() as handle:
        return json.load(handle)


PAPER_FIGURES = ROOT.parent.parent / "deductive_multi_agent_system_(MAS)_paper" / "figures"


def save(fig, name: str):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=220)
    if PAPER_FIGURES.parent.exists():
        PAPER_FIGURES.mkdir(exist_ok=True)
        fig.savefig(PAPER_FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def light_grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


ARCH_PANELS = [
    ("Socratic Evaluator", "#f4f6fa", BLUE, BLUE, [
        "differential testing against the reference; ddmin to a 1-minimal failing input",
        "variables matched by behaviour (DTW + Hungarian), not by name",
        "earliest divergent checkpoint",
        "static rules · asymptotic cost",
        "taxonomy-bound reasoning model",
        "Dempster–Shafer fusion",
    ]),
    ("Knowledge-Graph Alignment", "#f1faf6", AQUA, "#0f7a55", [
        "65-concept prerequisite DAG",
        "blame: personalised PageRank over reversed prerequisites",
        "Bayesian knowledge tracing with soft, graph-coupled evidence",
        "deficiency frontier = unmastered concepts whose prerequisites are secure (the ZPD)",
    ]),
    ("Cognitive Intervention", "#fdf4ef", ORANGE, "#b34a1e", [
        "target chosen in the ZPD by item response theory",
        "evidence cards: BM25 + LSA, rank fusion, MMR",
        "three-rung Socratic ladder and a predict-then-run challenge",
        "every sentence grounded",
        "answer-leakage gate, enforced by replacement rather than warning",
    ]),
]

ARCH_NOTE = ("Never shown to the student: the reference solution (an oracle for differential "
             "testing and the target of the leakage gate) and the reasoning model's chain of "
             "thought, which is captured for audit only.")
ARCH_LABELS = {"in_above": "student", "in_below": "program", "belief": "belief",
               "frontier": "frontier", "out_above": "hint +", "out_below": "challenge"}


# 1. architecture
def fig_architecture():
    """Pipeline overview. Boxes are sized from measured text extents (data units
    are inches, axes fill the figure) so nothing overflows.
    """
    W = TEXT_WIDTH
    fig = plt.figure(figsize=(W, 2.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    renderer = fig.canvas.get_renderer()

    def width_of(text, size, weight="normal", style="normal"):
        probe = ax.text(0, 0, text, fontsize=size, fontweight=weight, style=style)
        extent = probe.get_window_extent(renderer=renderer)
        probe.remove()
        return extent.width / fig.dpi * 1.04      # 4% slack for pdf font metrics

    def wrap(text, size, max_width):
        lines, current = [], ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and width_of(candidate, size) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    panels = ARCH_PANELS
    note = ARCH_NOTE

    # horizontal budget (inches)
    margin, gap = 0.03, 0.36
    panel_w = (W - 2 * margin - 4 * gap) / 3
    FS, TITLE_FS, LABEL_FS = 5.9, 6.9, 5.4
    pad_x, bullet_w = 0.07, 0.09
    text_w = panel_w - 2 * pad_x - bullet_w
    wrapped = [[wrap(item, FS, text_w) for item in items] for *_, items in panels]
    n_lines = max(sum(len(lines) for lines in items) for items in wrapped)
    LH = FS * 1.24 / 72
    title_h, pad_top, pad_bottom = TITLE_FS * 1.5 / 72, 0.05, 0.06
    panel_h = pad_top + title_h + n_lines * LH + pad_bottom

    # vertical budget
    NOTE_FS = 5.6
    note_lines = wrap(note, NOTE_FS, W - 0.4)
    note_lh = NOTE_FS * 1.3 / 72
    note_h = len(note_lines) * note_lh
    H = 0.04 + panel_h + 0.08 + note_h + 0.04
    fig.set_size_inches(W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    y_top = H - 0.04
    y_mid = y_top - panel_h / 2

    def rounded(x, y, w, h, fc, ec):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                            fc=fc, ec=ec, lw=0.8, zorder=2))

    def arrow(x0, x1, above=None, below=None, weight="normal", color=MUTED):
        ax.annotate("", xy=(x1, y_mid), xytext=(x0, y_mid),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0,
                                    mutation_scale=6), zorder=4)
        if above:
            ax.text((x0 + x1) / 2, y_mid + 0.035, above, ha="center", va="bottom",
                    fontsize=LABEL_FS, fontweight=weight, color=color, zorder=4)
        if below:
            ax.text((x0 + x1) / 2, y_mid - 0.035, below, ha="center", va="top",
                    fontsize=LABEL_FS, fontweight=weight, color=color, zorder=4)

    # draw left to right
    x = margin
    arrow(x, x + gap, ARCH_LABELS["in_above"], ARCH_LABELS["in_below"], weight="bold", color=INK)
    x += gap
    labels = [None, ARCH_LABELS["belief"], ARCH_LABELS["frontier"]]
    for k, ((title, fc, ec, title_color, _), items) in enumerate(zip(panels, wrapped)):
        rounded(x, y_top - panel_h, panel_w, panel_h, fc, ec)
        ax.text(x + panel_w / 2, y_top - pad_top, title, ha="center", va="top",
                fontsize=TITLE_FS, fontweight="bold", color=title_color, zorder=3)
        y = y_top - pad_top - title_h
        for lines in items:
            for j, line in enumerate(lines):
                if j == 0:
                    ax.text(x + pad_x + 0.02, y, "•", ha="left", va="top", fontsize=FS,
                            color=title_color, zorder=3)
                ax.text(x + pad_x + bullet_w, y, line, ha="left", va="top", fontsize=FS,
                        color=INK, zorder=3)
                y -= LH
        x += panel_w
        if k < 2:
            label = labels[k + 1]
            above, below = label if isinstance(label, tuple) else (label, None)
            arrow(x, x + gap, above, below)
            x += gap
    arrow(x, x + gap, ARCH_LABELS["out_above"], ARCH_LABELS["out_below"], weight="bold", color=INK)

    y = 0.04 + note_h - note_lh / 2
    for line in note_lines:
        ax.text(W / 2, y, line, ha="center", va="center", fontsize=NOTE_FS, color=MUTED,
                style="italic", zorder=3)
        y -= note_lh

    # guard: every text artist must be inside the figure
    fig.canvas.draw()
    for artist in ax.texts:
        bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
        x0, y0 = bb.x0 / fig.dpi, bb.y0 / fig.dpi
        x1, y1 = bb.x1 / fig.dpi, bb.y1 / fig.dpi
        if x0 < -0.005 or x1 > W + 0.005 or y0 < -0.005 or y1 > H + 0.005:
            print(f"  WARNING: text outside figure: {artist.get_text()!r}")
    save(fig, "fig_architecture")


# 2. RQ1, diagnostic accuracy across banks and systems
def fig_rq1():
    systems = [
        ("deterministic", "rq1_{bank}_full_offline.json", BLUE),
        ("+ reasoning model", "rq1_{bank}_full_gpt-oss-120b.json", VIOLET),
        ("model only", "rq1_{bank}_modelonly_gpt-oss-120b.json", ORANGE),
    ]
    metrics = [("top1_accuracy", "top-1 accuracy"), ("micro_f1", "micro-F1"), ("recall_any", "recall (any label)"),
               ("cohen_kappa", "Cohen's κ")]
    data = {}
    for bank in ("dev", "heldout"):
        for label, pattern, colour in systems:
            payload = load(pattern.format(bank=bank))
            if payload is not None and payload.get("degraded_rows", 0) == 0:
                data[(bank, label)] = payload
    if not data:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(TEXT_WIDTH, 1.55), sharey=False)
    for ax, (key, title) in zip(axes, metrics):
        light_grid(ax, "y")
        for b, bank in enumerate(("dev", "heldout")):
            for s, (label, _, colour) in enumerate(systems):
                payload = data.get((bank, label))
                if payload is None:
                    continue
                x = b * 4.4 + s * 1.15
                value = float(payload[key])
                ax.bar(x, value, width=0.9, color=colour, zorder=3, linewidth=0)
                ax.text(x, value + 0.02, f"{value:.2f}".lstrip("0") if value < 1 else "1.0",
                        ha="center", va="bottom", fontsize=5.4, color=INK)
        ax.set_xticks([1.15, 5.55])
        ax.set_xticklabels(["development", "held-out"], fontsize=6.8)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=7.6)
        ax.tick_params(length=0)
        if key != "top1_accuracy":
            ax.set_yticklabels([])
    handles = [patches.Patch(color=c, label=l) for l, _, c in systems if any(k[1] == l for k in data)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), handlelength=1.0, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.18)
    save(fig, "fig_rq1")


def fig_rq1_labels():
    """Per misconception recall on the held-out bank, one dot per system."""
    systems = [
        ("deterministic", "rq1_heldout_full_offline.json", BLUE, "o"),
        ("+ gpt-oss-120b", "rq1_heldout_full_gpt-oss-120b.json", VIOLET, "s"),
        ("+ gpt-oss-20b", "rq1_heldout_full_gpt-oss-20b.json", AQUA, "^"),
        ("gpt-oss-120b only", "rq1_heldout_modelonly_gpt-oss-120b.json", ORANGE, "D"),
    ]
    loaded = [(l, load(p), c, m) for l, p, c, m in systems]
    loaded = [(l, d, c, m) for l, d, c, m in loaded if d is not None and d.get("degraded_rows", 0) == 0]
    if not loaded:
        return
    labels = sorted({k for _, d, _, _ in loaded for k, v in d["per_label"].items() if v["support"]},
                    key=lambda k: (-loaded[0][1]["per_label"].get(k, {"support": 0})["support"], k))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 0.23 * len(labels) + 0.6))
    light_grid(ax, "x")
    step = 0.78 / max(1, len(loaded) - 1)          # dodge so coincident dots stay visible
    for i, label in enumerate(labels):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f3f2ee", lw=0, zorder=0)
        for j, (name, d, colour, marker) in enumerate(loaded):
            v = d["per_label"].get(label)
            if v is None:
                continue
            y = i - 0.39 + j * step
            ax.scatter(v["recall"], y, s=13, color=colour, marker=marker, zorder=3, linewidths=0)
    support = loaded[0][1]["per_label"]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{l}  (n={support.get(l, {'support': 0})['support']})" for l in labels], fontsize=6.4, family="monospace")
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("recall on held-out submissions carrying the label")
    ax.tick_params(length=0)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="", markersize=4, label=l) for l, _, c, m in loaded]
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, handlelength=1.0, columnspacing=1.1)
    save(fig, "fig_rq1_labels")


# 3. external validation on Refactory
def fig_external():
    """Two panels with unequal text load, the right one has long monospace
    belief names, so the layout is constrained (tick labels measured, panels
    cannot collide) and the legend goes under both.
    """
    d = load("refactory_full_offline.json")
    if d is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.25), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.0, 1.02]})
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.06, hspace=0.0)
    questions = ["q1", "q2", "q3", "q4", "q5"]
    names = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    series = [
        ("wrong__counterexample_rate", "rejected program: failing input found", BLUE),
        ("wrong__named_correctness_belief_rate", "rejected program: a belief was named", AQUA),
        ("correct__false_accusation_rate", "accepted program: a belief was named", ORANGE),
    ]
    light_grid(ax1, "y")
    width = 0.27
    for s_, (key, label, colour) in enumerate(series):
        xs = [i + (s_ - 1) * width for i in range(len(questions))]
        ys = [d["by_question"][q][key] or 0.0 for q in questions]
        ax1.bar(xs, ys, width=width - 0.02, color=colour, zorder=3, linewidth=0, label=label)
        for x, y in zip(xs, ys):                      # upright labels, the bars are wide enough
            ax1.text(x, y + 0.02, f"{y:.2f}".lstrip("0") if y < 1 else "1", ha="center", va="bottom",
                     fontsize=5.0, color=MUTED)
    ax1.set_xticks(range(len(questions)))
    ax1.set_xticklabels(names, fontsize=6.6)
    ax1.set_xlim(-0.55, len(questions) - 0.45)
    ax1.set_ylim(0, 1.12)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(["0", ".25", ".50", ".75", "1"], fontsize=6.6)
    ax1.set_ylabel("rate of programs", fontsize=7)
    ax1.tick_params(length=0, pad=2)
    ov = d["overall"]
    ax1.set_title(f"{ov['n_correct']:,} accepted, {ov['n_wrong']:,} rejected programs", fontsize=7.2)
    fig.legend(loc="outside lower center", ncol=3, fontsize=6.0, handlelength=1.0, handleheight=0.8,
               columnspacing=1.6, handletextpad=0.5, borderaxespad=0.1)

    prim = d["primary_on_wrong"]
    top = sorted(prim.items(), key=lambda kv: -kv[1])[:9]
    total = sum(prim.values())
    light_grid(ax2, "x")
    ys = list(range(len(top)))
    ax2.barh(ys, [v / total for _, v in top], color=BLUE, zorder=3, height=0.64, linewidth=0)
    for y, (k, v) in zip(ys, top):
        ax2.text(v / total + 0.004, y, f"{v}", va="center", fontsize=5.6, color=MUTED)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([k for k, _ in top], fontsize=5.9, family="monospace")
    ax2.invert_yaxis()
    ax2.set_ylim(len(top) - 0.4, -0.6)
    ax2.set_xlim(0, max(v for _, v in top) / total * 1.18)
    ax2.set_xticks([0, 0.05, 0.10, 0.15])
    ax2.set_xticklabels(["0", ".05", ".10", ".15"], fontsize=6.6)
    ax2.set_xlabel("share of headline beliefs, rejected programs", fontsize=6.7)
    ax2.tick_params(length=0, pad=2)
    ax2.set_title("what the evaluator names", fontsize=7.2)
    save(fig, "fig_external")


# 4. the gate
def fig_gate():
    adv = load("gate_adversarial_offline.json")
    audits = [("deterministic generator", "gate_audit_offline.json"),
              ("gpt-oss-120b", "gate_audit_gpt-oss-120b.json"),
              ("gpt-oss-20b", "gate_audit_gpt-oss-20b.json")]
    audit_data = [(l, load(p)) for l, p in audits]
    audit_data = [(l, d) for l, d in audit_data if d is not None and d.get("degraded_rows", 0) == 0]
    if adv is None:
        return
    ncols = 1 + (1 if audit_data else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(TEXT_WIDTH, 1.95), gridspec_kw={"width_ratios": [1.15, 1] if ncols == 2 else [1]})
    ax = axes[0] if ncols == 2 else axes
    order = ["full-solution", "reference-line", "reference-line-renamed", "prose-assignment", "prose-repair",
             "student-line", "question", "fallback-rung"]
    pretty = {"full-solution": "whole reference solution", "reference-line": "a reference line",
              "reference-line-renamed": "a reference line, renamed", "prose-assignment": "“set x to …” in prose",
              "prose-repair": "repair in plain words (no code)", "student-line": "the student's own line",
              "question": "a pure question", "fallback-rung": "fixed fallback rung"}
    summary = adv["summary"]
    light_grid(ax, "x")
    ys = list(range(len(order)))
    for y, cat in zip(ys, order):
        s = summary[cat]
        leak = s["should_reject"]
        colour = BLUE if leak else GREY
        ax.barh(y, s["rejection_rate"], color=colour, height=0.62, zorder=3, linewidth=0)
        ax.text(min(s["rejection_rate"], 0.98) + 0.015 if s["rejection_rate"] < 0.95 else 0.60, y,
                f"{s['rejection_rate']:.0%}  (n={s['n']})", va="center", fontsize=6,
                color=INK if s["rejection_rate"] < 0.95 else "white", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty[c] for c in order], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("rejection rate")
    ax.tick_params(length=0)
    ax.set_title("manufactured guidance with known verdicts", fontsize=7.4)
    ax.legend(handles=[patches.Patch(color=BLUE, label="must be rejected"), patches.Patch(color=GREY, label="must be accepted")],
              loc="lower right", fontsize=6.2, handlelength=1.0)

    if audit_data:
        ax2 = axes[1]
        light_grid(ax2, "y")
        sources = [("generated", BLUE), ("sanitised", AQUA), ("template", YELLOW), ("fallback", ORANGE)]
        items = ["hint-1", "hint-2", "hint-3", "challenge"]
        n_models = len(audit_data)
        width = 0.8 / n_models
        for m, (label, d) in enumerate(audit_data):
            for i, item in enumerate(items):
                s = d["summary"].get(item)
                if s is None:
                    continue
                bottom = 0.0
                shares = {"generated": s["reached_student_as_generated"]}
                for k, v in s["replaced"].items():
                    shares[k] = v / s["n"]
                x = i + (m - (n_models - 1) / 2) * width
                for name, colour in sources:
                    share = shares.get(name, 0.0)
                    if share <= 0:
                        continue
                    ax2.bar(x, share, bottom=bottom, width=width - 0.04, color=colour, zorder=3, linewidth=0.4, edgecolor="white")
                    bottom += share
                ax2.text(x, 1.02, label.replace("deterministic generator", "det.").replace("gpt-oss-", "")[:5], ha="center", fontsize=5.4, color=MUTED, rotation=0)
        ax2.set_xticks(range(len(items)))
        ax2.set_xticklabels(["rung 1", "rung 2", "rung 3", "challenge"], fontsize=6.8)
        ax2.set_ylim(0, 1.12)
        ax2.set_ylabel("share of items")
        ax2.tick_params(length=0)
        ax2.set_title("what the student finally saw", fontsize=7.4)
        ax2.legend(handles=[patches.Patch(color=c, label=n) for n, c in sources], loc="lower center",
                   bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=6.0, handlelength=0.9, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig_gate")


# 5. RQ2 design analysis
def fig_design():
    d = load("rq2_design_offline.json")
    if d is None or not d.get("replication"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # (a) sampling distribution of the effect, default vs null model
    ax = axes[0]
    light_grid(ax, "y")
    rep = d["per_seed"]["replication"]
    null = d["per_seed"]["null"]
    import random
    rnd = random.Random(3)
    for x, (rows, key, colour, label) in enumerate([
        (rep, "retention_g", BLUE, "retention"),
        (rep, "repeat_error_g", AQUA, "recurring error"),
        (null, "retention_g", GREY, "retention"),
        (null, "repeat_error_g", GREY, "recurring error"),
    ]):
        vals = [r[key] for r in rows]
        xs = [x + rnd.uniform(-0.18, 0.18) for _ in vals]
        ax.scatter(xs, vals, s=5, color=colour, alpha=0.55, linewidths=0, zorder=3)
        lo, hi = sorted(vals)[int(0.05 * (len(vals) - 1))], sorted(vals)[int(0.95 * (len(vals) - 1))]
        ax.plot([x - 0.28, x + 0.28], [sum(vals) / len(vals)] * 2, color=INK, lw=1.0, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["retention", "recurring\nerror", "retention", "recurring\nerror"], fontsize=5.8)
    ax.set_xlim(-0.6, 3.6)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(ax.get_ylim()[0], ymax * 1.18)
    ax.text(0.5, ymax * 1.14, "default model", ha="center", va="top", fontsize=6.0, color=INK)
    ax.text(2.5, ymax * 1.14, "null model", ha="center", va="top", fontsize=6.0, color=MUTED)
    ax.set_ylabel("Hedges $g$, treatment − control")
    ax.set_title(f"{d['replication']['seeds']} cohorts of {d['replication']['students']}", fontsize=7.4)
    ax.tick_params(length=0)

    # (b) sensitivity
    ax = axes[1]
    light_grid(ax, "y")
    sens = d["sensitivity"]
    xs = [s["strength"] for s in sens]
    ax.plot(xs, [s["retention_g"]["mean"] for s in sens], color=BLUE, lw=1.4, marker="o", ms=3, label="retention $g$")
    ax.plot(xs, [-s["repeat_error_g"]["mean"] for s in sens], color=AQUA, lw=1.4, marker="s", ms=3, label="−(recurring error $g$)")
    ax.plot(xs, [s["on_target_treatment"]["mean"] for s in sens], color=ORANGE, lw=1.4, marker="D", ms=3, label="on-target rate (measured)")
    ax.plot(xs, [s["on_target_ablation"]["mean"] for s in sens], color=ORANGE, lw=1.0, ls="--", marker="D", ms=2.5, label="on-target, no graph")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("assumed Socratic advantage")
    ax.set_xticks(xs)
    ax.set_ylim(-0.08, 1.05)
    ax.tick_params(length=0)
    ax.legend(fontsize=5.3, loc="center left", bbox_to_anchor=(0.0, 0.66), handlelength=1.4, labelspacing=0.25)
    ax.set_title("sensitivity to the response model", fontsize=7.4)

    # (c) power
    ax = axes[2]
    light_grid(ax, "y")
    power = d["power"]
    ns = [p["n_per_arm"] for p in power]
    ax.plot(ns, [p["empirical_power_retention"] for p in power], color=BLUE, marker="o", ms=3, lw=1.4, label="empirical, retention")
    ax.plot(ns, [p["empirical_power_repeat_error"] for p in power], color=AQUA, marker="s", ms=3, lw=1.4, label="empirical, recurring error")
    ax.plot(ns, [p["analytic_power_at_mean_g"] for p in power], color=MUTED, ls="--", lw=1.0, label="analytic at mean $|g|$")
    ax.axhline(0.8, color=GRID, lw=0.8)
    ax.set_xlabel("learners per arm")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ns)
    ax.tick_params(length=0)
    ax.legend(fontsize=5.3, loc="lower right", handlelength=1.4, labelspacing=0.25)
    ax.set_title("power (Holm-corrected)", fontsize=7.4)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig_design")


# 6. RQ3 robustness
def fig_robustness():
    dev = load("rq3_robustness_dev_offline.json")
    held = load("rq3_robustness_heldout_offline.json")
    if dev is None:
        return
    order = ["rename-identifiers", "split-assignments", "expand-augmented", "invert-conditionals", "inject-noise", "all-combined"]
    pretty = {"rename-identifiers": "rename every identifier", "split-assignments": "split tuple assignments",
              "expand-augmented": "expand augmented assignments", "invert-conditionals": "invert conditionals",
              "inject-noise": "inject docstring + dead store", "all-combined": "all five at once"}
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 1.55))
    light_grid(ax, "x")
    ys = list(range(len(order)))
    for y, t in zip(ys, order):
        ax.plot([0, 1], [y, y], color=GRID, lw=0.5)
        ax.scatter(dev["by_transform"][t]["top1_stability"], y, s=22, color=BLUE, zorder=3, linewidths=0, marker="o")
        if held is not None:
            ax.scatter(held["by_transform"][t]["top1_stability"], y, s=22, color=ORANGE, zorder=3, linewidths=0, marker="s")
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty[t] for t in order], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("top-1 diagnosis unchanged under the rewrite")
    ax.tick_params(length=0)
    handles = [plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=4, label=f"development ({int(dev['overall']['n'])} mutants)")]
    if held is not None:
        handles.append(plt.Line2D([], [], color=ORANGE, marker="s", ls="", ms=4, label=f"held-out ({int(held['overall']['n'])} mutants)"))
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, handlelength=1.0)
    save(fig, "fig_robustness")


# 7. worked example: aligned trajectories and the divergence point
def fig_trace():
    """Both executions on one input, sampled at the loop checkpoints."""
    try:
        from deductive_mas import Config
        from deductive_mas.analysis.align import trajectory, locate_divergence
        from deductive_mas.analysis.tracer import GuardedRunner
        from deductive_mas.analysis.varmatch import match_variables
        from deductive_mas.pipeline import _sync_lines
        from deductive_mas.data.problems import PROBLEMS
        from deductive_mas.data.submissions import SUBMISSIONS
    except Exception as exc:
        print("  (package import failed:", exc, ")")
        return
    spec = PROBLEMS["lower_bound"]
    sub = SUBMISSIONS["lb_inclusive_bound"]
    args = ([1, 3, 5, 7], 8)
    runner = GuardedRunner(Config().execution)
    student = runner.run(sub.source, spec.entry_point, [list(args[0]), args[1]])
    reference = runner.run(spec.reference_solution, spec.entry_point, [list(args[0]), args[1]])
    mapping = match_variables(student, reference, student_params=["a", "t"], reference_params=["a", "t"])
    roles = [r for r in ("lo", "hi", "mid") if r in mapping.roles]
    rename = dict(mapping.pairs)
    s_path = trajectory(student, roles, rename, sync_lines=_sync_lines(sub.source, spec.entry_point))
    r_path = trajectory(reference, roles, sync_lines=_sync_lines(spec.reference_solution, spec.entry_point))
    div = locate_divergence(student, reference, mapping, source_lines=sub.lines,
                            student_sync=_sync_lines(sub.source, spec.entry_point),
                            reference_sync=_sync_lines(spec.reference_solution, spec.entry_point))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 1.7))
    light_grid(ax, "y")
    colours = {"lo": BLUE, "hi": ORANGE, "mid": AQUA}
    for role in roles:
        i = roles.index(role)
        ref_vals = [st.values[i] for st in r_path]
        stu_vals = [st.values[i] for st in s_path]
        ax.plot(range(len(ref_vals)), ref_vals, color=colours[role], lw=1.4, marker="o", ms=2.6, label=f"reference {role}")
        ax.plot(range(len(stu_vals)), stu_vals, color=colours[role], lw=1.4, ls="--", marker="s", ms=2.6, label=f"student {role}")
    # first checkpoint where the two disagree
    first = next((k for k, (a, b) in enumerate(zip(s_path, r_path)) if a.values != b.values), None)
    if first is not None:
        ax.axvspan(first - 0.5, max(len(s_path), len(r_path)) - 0.5, color=GRID, alpha=0.35, lw=0, zorder=0)
        ax.axvline(first, color=INK, lw=0.7, ls=":")
        ax.text(first + 0.1, ax.get_ylim()[1], "first disagreement", fontsize=6.0, va="top", color=INK)
    n = max(len(s_path), len(r_path))
    ax.set_xticks(range(1, n))
    ax.set_xticklabels([str(k) for k in range(1, n)])
    ax.set_xlim(0.6, n - 0.6)
    ax.set_xlabel("loop-invariant checkpoint (each evaluation of the loop guard)")
    ax.set_ylabel("value")
    ax.tick_params(length=0)
    ax.legend(fontsize=5.4, ncol=3, loc="lower right", handlelength=1.6, columnspacing=0.8, labelspacing=0.2)
    ax.set_title("lower_bound([1, 3, 5, 7], 8): the student's hi starts at len(a) − 1", fontsize=7.0)
    save(fig, "fig_trace")


# 8. knowledge graph with blame from one diagnosis
def fig_kg():
    """Blamed subgraph after one diagnosis, with frontier and target."""
    try:
        from deductive_mas import DeductiveOrchestrator, problem, submission
        from deductive_mas.knowledge.ontology import knowledge_graph
    except Exception as exc:
        print("  (package import failed:", exc, ")")
        return
    graph = knowledge_graph()
    result = DeductiveOrchestrator().tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
    blame = result.alignment.blame
    frontier = set(result.alignment.deficiency_frontier)
    target = result.intervention.target_concept
    peak = max(blame.values())
    keep = {cid for cid, b in blame.items() if b >= 0.02 * peak}
    depths = {cid: graph.depth(cid) for cid in keep}
    columns: dict = {}
    for cid in sorted(keep, key=lambda c: (depths[c], -blame[c])):
        columns.setdefault(depths[cid], []).append(cid)
    positions = {}
    for depth, members in columns.items():
        n = len(members)
        for i, cid in enumerate(members):
            positions[cid] = (depth, (i - (n - 1) / 2) * 1.7)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.5))
    ax.axis("off")
    for cid in keep:
        x1, y1 = positions[cid]
        for pre in graph.prerequisites(cid):
            if pre in keep:
                x0, y0 = positions[pre]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.7, alpha=0.75, shrinkA=7, shrinkB=7, mutation_scale=6), zorder=1)
    for cid, (x, y) in positions.items():
        b = blame[cid] / peak
        idx = min(len(SEQ) - 1, int(round(b * (len(SEQ) - 1))))
        if cid == target or cid in frontier:
            # ring outside the fill with a white gap so it reads on dark fills
            ax.scatter(x, y, s=270, color=ORANGE if cid == target else INK, linewidths=0, zorder=2)
            ax.scatter(x, y, s=205, color="white", linewidths=0, zorder=2.5)
        ax.scatter(x, y, s=150, color=SEQ[idx], edgecolors=MUTED, linewidths=0.4, zorder=3)
        ax.text(x, y, f"{100 * blame[cid]:.0f}%", ha="center", va="center", fontsize=5.4,
                color="white" if idx >= 3 else INK, zorder=4)
        import textwrap
        ax.text(x, y - 0.36, "\n".join(textwrap.wrap(graph.name(cid), 13)), ha="center", va="top",
                fontsize=5.3, color=INK, zorder=4, linespacing=0.95)
    xs = sorted(columns)
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ys = [p[1] for p in positions.values()]
    ax.set_ylim(min(ys) - 1.25, max(ys) + 0.75)
    for depth in xs:
        ax.text(depth, max(ys) + 0.62, f"depth {depth}", ha="center", fontsize=5.4, color=MUTED)
    ax.text(0.0, -0.02, "share of blame inside each node; arrows point from prerequisite to dependent; "
            "ringed: the deficiency frontier (orange: the concept the intervention targets)",
            transform=ax.transAxes, fontsize=5.6, color=MUTED, style="italic")
    save(fig, "fig_kg")


if __name__ == "__main__":
    print("figures:")
    fig_architecture()
    fig_rq1()
    fig_rq1_labels()
    fig_external()
    fig_gate()
    fig_design()
    fig_robustness()
    fig_trace()
    fig_kg()
