"""Draw every figure in the paper from the json in results/.

    /usr/local/bin/python3 experiments/make_figures.py

Only needs matplotlib (the package itself has no deps, this script is the
one place a third party lib shows up). Figures with missing inputs are
skipped with a note so it can be rerun as results come in.
"""

import json
import sys
import textwrap
from pathlib import Path
from typing import Literal, Optional
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parent / "src"))

# palette. one dark hue carries the system, one warm hue carries whatever it
# is being compared against, the other two only show up when a panel needs a
# third and a fourth line. all four are muted so they print well in grey.
SLATE, BRICK, TEAL, OCHRE = "#2f4858", "#9e4634", "#3a6b63", "#b4883c"
STONE = "#8fa0ad"
GREY, INK, MUTED, RULE = "#8e8c86", "#17181a", "#5b5a56", "#d3d1ca"
SEQ = ["#eef2f5", "#d2dde4", "#b0c1cd", "#8aa3b4", "#64849a", "#436781", "#2a4c63"]

plt.rcParams.update({
    # a list of real families, not the "sans-serif" alias, so a glyph the first
    # font lacks is taken from the next one instead of coming out as a box
    "font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
    "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
    "font.monospace": ["Menlo", "Monaco", "DejaVu Sans Mono"],
    "mathtext.fontset": "stixsans",
    "font.size": 7.6,
    "axes.titlesize": 7.2,
    "axes.titlelocation": "left",
    "axes.titlepad": 4.0,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "legend.fontsize": 6.1,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": RULE,
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "legend.handlelength": 1.0,
    "legend.handletextpad": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})

TEXT_WIDTH = 5.5


def load(name: str) -> Optional[dict[str, Any]]:
    path = RESULTS / name
    if not path.exists():
        print(f"  (missing {name})")
        return None
    with path.open() as handle:
        return json.load(handle)


PAPER_FIGURES = ROOT.parent.parent / "deductive_multi_agent_system_(MAS)_paper" / "figures"


def save(fig: Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=220)
    if PAPER_FIGURES.parent.exists():
        PAPER_FIGURES.mkdir(exist_ok=True)
        fig.savefig(PAPER_FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def light_grid(ax: plt.Axes, axis: Literal["both", "x", "y"] = "y") -> None:
    """Faint rules behind the data, and drop the spine that runs beside them."""
    ax.grid(True, axis=axis, color=RULE, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    if axis == "y":
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    elif axis == "x":
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)


def end_label(ax: plt.Axes, x: float, y: float, text: str, colour: str,
              size: float = 6.0, dx: float = 0.0) -> None:
    """Name a line where it ends, instead of in a legend box."""
    ax.text(x + dx, y, text, color=colour, fontsize=size, va="center", ha="left",
            clip_on=False, zorder=5)


ARCH_PANELS = [
    ("Socratic evaluator", "#f1f4f7", "#b7c4ce", SLATE, [
        "differential testing against the reference; ddmin to a 1-minimal failing input",
        "variables matched by behaviour (DTW + Hungarian), not by name",
        "earliest divergent checkpoint",
        "static rules · asymptotic cost",
        "taxonomy-bound reasoning model",
        "Dempster–Shafer fusion",
    ]),
    ("Knowledge-graph alignment", "#f0f5f3", "#b2c9c3", TEAL, [
        "65-concept prerequisite DAG",
        "blame: personalised PageRank over reversed prerequisites",
        "Bayesian knowledge tracing with soft, graph-coupled evidence",
        "deficiency frontier = unmastered concepts whose prerequisites are secure (the ZPD)",
    ]),
    ("Cognitive intervention", "#faf2ef", "#dab7ac", BRICK, [
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
def fig_architecture() -> None:
    """Pipeline overview. Boxes are sized from measured text extents (data
    units are inches, axes fill the figure) so nothing overflows.
    """
    W = TEXT_WIDTH
    fig = plt.figure(figsize=(W, 2.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    renderer = fig.canvas.get_renderer()

    def width_of(text: str, size: float, weight: str = "normal", style: str = "normal") -> float:
        probe = ax.text(0, 0, text, fontsize=size, fontweight=weight, style=style)
        extent = probe.get_window_extent(renderer=renderer)
        probe.remove()
        return extent.width / fig.dpi * 1.04      # 4% slack for pdf font metrics

    def wrap(text: str, size: float, max_width: float) -> list[str]:
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
    margin, gap = 0.03, 0.37
    panel_w = (W - 2 * margin - 4 * gap) / 3
    FS, TITLE_FS, LABEL_FS = 5.9, 7.0, 5.1
    pad_x, bullet_w = 0.075, 0.085
    text_w = panel_w - 2 * pad_x - bullet_w
    # the longest title ran all the way out to the edge of its box, past the
    # rule underneath it. shrink the titles by as much as the worst one needs
    # and apply that to all three, so they stay the same size as each other
    # and none of them is wider than its rule. measured, not hardcoded: the
    # vietnamese figures swap in longer titles.
    widest_title = max(width_of(title, TITLE_FS) for title, *_ in panels)
    title_fs = min(TITLE_FS, TITLE_FS * (panel_w - 2 * pad_x) / widest_title)
    wrapped = [[wrap(item, FS, text_w) for item in items] for *_, items in panels]
    n_items = max(len(items) for items in wrapped)
    n_lines = max(sum(len(lines) for lines in items) for items in wrapped)
    LH = FS * 1.26 / 72
    ITEM_GAP = LH * 0.3                           # a little air between bullets
    title_h, rule_gap, pad_top, pad_bottom = title_fs * 1.45 / 72, 0.052, 0.062, 0.075
    panel_h = pad_top + title_h + rule_gap + n_lines * LH + (n_items - 1) * ITEM_GAP + pad_bottom

    # vertical budget
    NOTE_FS = 5.6
    note_lines = wrap(note, NOTE_FS, W - 0.4)
    note_lh = NOTE_FS * 1.3 / 72
    note_h = len(note_lines) * note_lh
    H = 0.04 + panel_h + 0.1 + note_h + 0.04
    fig.set_size_inches(W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    y_top = H - 0.04
    y_mid = y_top - panel_h / 2

    def rounded(x: float, y: float, w: float, h: float, fc: str, ec: str) -> None:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.028",
                                            fc=fc, ec=ec, lw=0.6, zorder=2))

    def arrow(x0: float, x1: float, above: Optional[str] = None, below: Optional[str] = None, color: str = MUTED) -> None:
        ax.annotate("", xy=(x1 - 0.015, y_mid), xytext=(x0 + 0.015, y_mid),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7, shrinkA=0, shrinkB=0,
                                    mutation_scale=5), zorder=4)
        if above:
            ax.text((x0 + x1) / 2, y_mid + 0.032, above, ha="center", va="bottom",
                    fontsize=LABEL_FS, color=color, zorder=4)
        if below:
            ax.text((x0 + x1) / 2, y_mid - 0.032, below, ha="center", va="top",
                    fontsize=LABEL_FS, color=color, zorder=4)

    # draw left to right
    x = margin
    arrow(x, x + gap, ARCH_LABELS["in_above"], ARCH_LABELS["in_below"], color=INK)
    x += gap
    labels = [None, ARCH_LABELS["belief"], ARCH_LABELS["frontier"]]
    for k, ((title, fc, ec, title_color, _), items) in enumerate(zip(panels, wrapped)):
        rounded(x, y_top - panel_h, panel_w, panel_h, fc, ec)
        ax.text(x + panel_w / 2, y_top - pad_top, title, ha="center", va="top",
                fontsize=title_fs, color=title_color, zorder=3)
        y_rule = y_top - pad_top - title_h
        ax.plot([x + pad_x, x + panel_w - pad_x], [y_rule, y_rule], color=ec, lw=0.6,
                solid_capstyle="butt", zorder=3)
        y = y_rule - rule_gap
        for lines in items:
            for j, line in enumerate(lines):
                if j == 0:
                    ax.text(x + pad_x + 0.012, y - LH * 0.09, "•", ha="left", va="top",
                            fontsize=FS * 0.82, color=ec, zorder=3)
                ax.text(x + pad_x + bullet_w, y, line, ha="left", va="top", fontsize=FS,
                        color=INK, zorder=3)
                y -= LH
            y -= ITEM_GAP
        x += panel_w
        if k < 2:
            label = labels[k + 1]
            above, below = label if isinstance(label, tuple) else (label, None)
            arrow(x, x + gap, above, below)
            x += gap
    arrow(x, x + gap, ARCH_LABELS["out_above"], ARCH_LABELS["out_below"], color=INK)

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
def fig_rq1() -> None:
    systems = [
        ("deterministic", "rq1_{bank}_full_offline.json", SLATE),
        ("+ reasoning model", "rq1_{bank}_full_gpt-oss-120b.json", STONE),
        ("model only", "rq1_{bank}_modelonly_gpt-oss-120b.json", BRICK),
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
    fig, axes = plt.subplots(1, len(metrics), figsize=(TEXT_WIDTH, 1.6), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        light_grid(ax, "y")
        for b, bank in enumerate(("dev", "heldout")):
            for s, (label, _, colour) in enumerate(systems):
                payload = data.get((bank, label))
                if payload is None:
                    continue
                x = b * 4.4 + s * 1.15
                value = float(payload[key])
                ax.bar(x, value, width=0.92, color=colour, zorder=3, linewidth=0)
                # every bar gets its number. labelling only one group made the
                # two halves look like they were measured differently, and the
                # one place the ordering flips (micro-F1 on development) was
                # too easy to miss.
                # a full "1.00" is wide enough to run into the label on the
                # next bar, so write it "1", the way the axis writes it
                shown = "1" if value > 0.995 else f"{value:.2f}".lstrip("0")
                ax.text(x, value + 0.025, shown, ha="center", va="bottom",
                        fontsize=5.6, color=MUTED)
        ax.set_xticks([1.15, 5.55])
        ax.set_xticklabels(["development", "held-out"])
        # the bars that reach 1 now carry a label on top, so leave room for it
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
        ax.set_title(title)
        ax.tick_params(axis="x", length=0, pad=3)
        # the last bar sits at x = 6.70 and is 0.92 wide, so it ends at 7.16.
        # the limit has to clear that or matplotlib clips it and the bar comes
        # out thinner than the rest. same 0.34 of air on both sides.
        ax.set_xlim(-0.8, 7.5)
    handles = [patches.Patch(color=c, label=l) for l, _, c in systems if any(k[1] == l for k in data)]
    fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.005, -0.12),
               columnspacing=1.6)
    fig.subplots_adjust(wspace=0.14)
    save(fig, "fig_rq1")


def fig_rq1_labels() -> None:
    """Per misconception recall on the held-out bank, one dot per system.
    Dotted leaders carry the eye from the name to the dots.
    """
    systems = [
        ("deterministic", "rq1_heldout_full_offline.json", SLATE, "o", True),
        ("+ gpt-oss-120b", "rq1_heldout_full_gpt-oss-120b.json", SLATE, "o", False),
        ("+ gpt-oss-20b", "rq1_heldout_full_gpt-oss-20b.json", TEAL, "^", True),
        ("gpt-oss-120b only", "rq1_heldout_modelonly_gpt-oss-120b.json", BRICK, "D", True),
    ]
    loaded = [(l, load(p), c, m, f) for l, p, c, m, f in systems]
    loaded = [(l, d, c, m, f) for l, d, c, m, f in loaded if d is not None and d.get("degraded_rows", 0) == 0]
    if not loaded:
        return
    labels = sorted({k for _, d, _, _, _ in loaded for k, v in d["per_label"].items() if v["support"]},
                    key=lambda k: (-loaded[0][1]["per_label"].get(k, {"support": 0})["support"], k))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 0.21 * len(labels) + 0.55))
    step = 0.62 / max(1, len(loaded) - 1)          # dodge so coincident dots stay visible
    for i, label in enumerate(labels):
        ax.plot([0, 1], [i, i], color=RULE, lw=0.5, ls=(0, (1, 2.4)), zorder=1)
        for j, (name, d, colour, marker, filled) in enumerate(loaded):
            v = d["per_label"].get(label)
            if v is None:
                continue
            y = i - 0.31 + j * step
            ax.scatter(v["recall"], y, s=11, zorder=3,
                       facecolors=colour if filled else "white",
                       edgecolors=colour, linewidths=0.7, marker=marker)
    support = loaded[0][1]["per_label"]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{l}  (n={support.get(l, {'support': 0})['support']})" for l in labels],
                       fontsize=6.2, family="monospace")
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlabel("recall on held-out submissions carrying the label")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="", markersize=3.6, label=l,
                          markerfacecolor=c if f else "white", markeredgewidth=0.7)
               for l, _, c, m, f in loaded]
    ax.set_ylim(len(labels) - 0.5, -0.9)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4,
              columnspacing=1.2)
    save(fig, "fig_rq1_labels")


# 3. external validation on Refactory
def fig_external() -> None:
    """Two panels with unequal text load, the right one has long monospace
    belief names. The key sits under the left panel only, because that is the
    panel it describes.
    """
    d = load("refactory_full_offline.json")
    if d is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.35), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.0, 1.08]})
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.10, hspace=0.0)
    questions = ["q1", "q2", "q3", "q4", "q5"]
    names = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    series = [
        ("wrong__counterexample_rate", "rejected program: failing input found", SLATE),
        ("wrong__named_correctness_belief_rate", "rejected program: a belief was named", STONE),
        ("correct__false_accusation_rate", "accepted program: a belief was named", BRICK),
    ]
    light_grid(ax1, "y")
    width = 0.27
    for s_, (key, label, colour) in enumerate(series):
        xs = [i + (s_ - 1) * width for i in range(len(questions))]
        ys = [d["by_question"][q][key] or 0.0 for q in questions]
        ax1.bar(xs, ys, width=width - 0.03, color=colour, zorder=3, linewidth=0, label=label)
    ax1.set_xticks(range(len(questions)))
    ax1.set_xticklabels(names)
    ax1.set_xlim(-0.55, len(questions) - 0.45)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax1.set_ylabel("rate of programs")
    ax1.tick_params(axis="x", length=0, pad=3)
    ov = d["overall"]
    ax1.set_title(f"{ov['n_correct']:,} accepted, {ov['n_wrong']:,} rejected programs")
    fig.legend(loc="outside lower left", ncol=1, labelspacing=0.35, borderaxespad=0.1)

    prim = d["primary_on_wrong"]
    top = sorted(prim.items(), key=lambda kv: -kv[1])[:9]
    total = sum(prim.values())
    ys = list(range(len(top)))
    ax2.barh(ys, [v / total for _, v in top], color=SLATE, zorder=3, height=0.6, linewidth=0)
    for y, (k, v) in zip(ys, top):
        ax2.text(v / total + 0.004, y, f"{v}", va="center", fontsize=5.5, color=MUTED)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([k for k, _ in top], fontsize=5.4, family="monospace")
    ax2.invert_yaxis()
    ax2.set_ylim(len(top) - 0.4, -0.7)
    ax2.set_xlim(0, max(v for _, v in top) / total * 1.2)
    ax2.set_xticks([0, 0.05, 0.10, 0.15])
    ax2.set_xticklabels(["0", ".05", ".10", ".15"])
    ax2.set_xlabel("share of headline beliefs, rejected programs", loc="left", fontsize=6.0)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0, pad=2)
    ax2.set_title("what the evaluator names")
    save(fig, "fig_external")


# 4. the gate
def fig_gate() -> None:
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
        colour = SLATE if leak else STONE
        ax.barh(y, s["rejection_rate"], color=colour, height=0.58, zorder=3, linewidth=0)
        inside = s["rejection_rate"] > 0.5
        ax.text(s["rejection_rate"] - 0.02 if inside else s["rejection_rate"] + 0.015, y,
                f"{s['rejection_rate']:.0%}  (n={s['n']})", va="center", fontsize=6,
                ha="right" if inside else "left",
                color="white" if inside else MUTED, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty[c] for c in order], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlabel("rejection rate")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_title("manufactured guidance with known verdicts")
    # the two blocks are named in the right margin, which saves a legend whose
    # grey patch would point at bars of length zero
    n_reject = sum(1 for c in order if summary[c]["should_reject"])
    for lo, hi, text in [(0, n_reject - 1, "must be\nrejected"),
                         (n_reject, len(order) - 1, "must be\naccepted")]:
        ax.plot([1.03, 1.03], [lo - 0.2, hi + 0.2], color=GREY, lw=0.6, clip_on=False, zorder=5)
        ax.text(1.05, (lo + hi) / 2, text, rotation=-90, ha="left", va="center", fontsize=5.6,
                color=MUTED, clip_on=False, zorder=5, linespacing=1.1, multialignment="center")

    if audit_data:
        ax2 = axes[1]
        light_grid(ax2, "y")
        sources = [("generated", SLATE), ("sanitised", TEAL), ("template", OCHRE), ("fallback", BRICK)]
        items = ["hint-1", "hint-2", "hint-3", "challenge"]
        n_models = len(audit_data)
        width = 0.78 / n_models
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
                    ax2.bar(x, share, bottom=bottom, width=width - 0.035, color=colour, zorder=3,
                            linewidth=0.4, edgecolor="white")
                    bottom += share
                # the model goes inside the foot of its own bar, so the three
                # bars of a group need no second row of tick labels
                ax2.text(x, 0.03, label.replace("deterministic generator", "det.").replace("gpt-oss-", "")[:5],
                         ha="center", va="bottom", rotation=90, fontsize=4.9, color="white", zorder=5)
        ax2.set_xticks(range(len(items)))
        ax2.set_xticklabels(["rung 1", "rung 2", "rung 3", "challenge"], fontsize=5.9)
        ax2.set_ylim(0, 1.1)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_yticklabels(["0", ".25", ".50", ".75", "1"])
        ax2.set_ylabel("share of items")
        ax2.tick_params(axis="x", length=0, pad=3)
        ax2.set_title("what the student finally saw")
        ax2.legend(handles=[patches.Patch(color=c, label=n) for n, c in sources], loc="upper left",
                   bbox_to_anchor=(-0.14, -0.13), ncol=2, columnspacing=1.2, labelspacing=0.3,
                   borderaxespad=0.0)
    fig.subplots_adjust(wspace=0.62, bottom=0.22)
    save(fig, "fig_gate")


# 5. RQ2 design analysis
def fig_design() -> None:
    d = load("rq2_design_offline.json")
    if d is None or not d.get("replication"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.95), gridspec_kw={"width_ratios": [1.4, 1, 1]})

    # (a) sampling distribution of the effect, default vs null model
    ax = axes[0]
    light_grid(ax, "y")
    rep = d["per_seed"]["replication"]
    null = d["per_seed"]["null"]
    import random
    rnd = random.Random(3)
    for x, (rows, key, colour) in enumerate([
        (rep, "retention_g", SLATE),
        (rep, "repeat_error_g", TEAL),
        (null, "retention_g", GREY),
        (null, "repeat_error_g", GREY),
    ]):
        vals = [r[key] for r in rows]
        xs = [x + rnd.uniform(-0.17, 0.17) for _ in vals]
        ax.scatter(xs, vals, s=4, color=colour, alpha=0.45, linewidths=0, zorder=3)
        ordered = sorted(vals)
        lo, hi = ordered[int(0.05 * (len(vals) - 1))], ordered[int(0.95 * (len(vals) - 1))]
        ax.plot([x, x], [lo, hi], color=INK, lw=0.6, zorder=4)         # 5th to 95th
        ax.plot([x - 0.26, x + 0.26], [sum(vals) / len(vals)] * 2, color=INK, lw=1.1, zorder=5)
    ax.axhline(0, color=MUTED, lw=0.5, zorder=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["retention", "recurring\nerror"] * 2, fontsize=4.9, linespacing=1.0)
    ax.set_xlim(-0.6, 3.6)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(ax.get_ylim()[0], ymax * 1.22)
    ax.text(0.5, ymax * 1.18, "default model", ha="center", va="top", fontsize=6.0, color=INK)
    ax.text(2.5, ymax * 1.18, "null model", ha="center", va="top", fontsize=6.0, color=GREY)
    ax.set_ylabel("Hedges $g$, treatment − control")
    ax.set_title(f"{d['replication']['seeds']} cohorts of {d['replication']['students']}")
    ax.tick_params(axis="x", length=0, pad=2)

    # (b) sensitivity, lines named where they end
    ax = axes[1]
    light_grid(ax, "y")
    sens = d["sensitivity"]
    xs = [s["strength"] for s in sens]
    lines = [
        ([s["retention_g"]["mean"] for s in sens], SLATE, "-", "o", 1.3, "retention $g$"),
        ([-s["repeat_error_g"]["mean"] for s in sens], TEAL, "-", "s", 1.3, "−(recurring error $g$)"),
        ([s["on_target_treatment"]["mean"] for s in sens], BRICK, "-", "D", 1.3, "on-target rate (measured)"),
        ([s["on_target_ablation"]["mean"] for s in sens], BRICK, "--", "D", 0.9, "on-target, no graph"),
    ]
    for ys, colour, style, marker, lw, label in lines:
        ax.plot(xs, ys, color=colour, lw=lw, ls=style, marker=marker, ms=2.6, markeredgewidth=0)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xlabel("assumed Socratic advantage")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_ylim(-0.08, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_title("sensitivity to the response model", fontsize=6.7)
    ax.legend(handles=[plt.Line2D([], [], color=c, ls=s, lw=lw, marker=m, ms=2.6, label=t)
                       for _, c, s, m, lw, t in lines],
              fontsize=4.9, loc="upper left", bbox_to_anchor=(-0.02, 0.88), handlelength=1.5,
              labelspacing=0.22, borderaxespad=0.0)

    # (c) power
    ax = axes[2]
    light_grid(ax, "y")
    power = d["power"]
    ns = [p["n_per_arm"] for p in power]
    curves = [
        ([p["empirical_power_retention"] for p in power], SLATE, "-", "o", 1.3, "retention"),
        ([p["empirical_power_repeat_error"] for p in power], TEAL, "-", "s", 1.3, "recurring error"),
        ([p["analytic_power_at_mean_g"] for p in power], GREY, "--", None, 0.9, "analytic"),
    ]
    ends = []
    for ys, colour, style, marker, lw, label in curves:
        ax.plot(ns, ys, color=colour, lw=lw, ls=style, marker=marker, ms=2.6, markeredgewidth=0)
        ends.append((ys[-1], colour, label))
    ends.sort()
    for k, (y, colour, label) in enumerate(ends):      # keep coincident labels apart
        if k and y - ends[k - 1][0] < 0.07:
            y = ends[k - 1][0] + 0.07
            ends[k] = (y, colour, label)
        end_label(ax, ns[-1], y, f" {label}", colour, size=5.0)
    ax.axhline(0.8, color=RULE, lw=0.6)
    ax.set_xlabel("learners per arm")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlim(ns[0] - 3, ns[-1] + (ns[-1] - ns[0]) * 0.85)
    ax.set_xticks(ns)
    ax.set_title("power (Holm-corrected)", fontsize=6.7)
    fig.subplots_adjust(wspace=0.5, bottom=0.2)
    save(fig, "fig_design")


# 6. RQ3 robustness
def fig_robustness() -> None:
    dev = load("rq3_robustness_dev_offline.json")
    held = load("rq3_robustness_heldout_offline.json")
    if dev is None:
        return
    order = ["rename-identifiers", "split-assignments", "expand-augmented", "invert-conditionals", "inject-noise", "all-combined"]
    pretty = {"rename-identifiers": "rename every identifier", "split-assignments": "split tuple assignments",
              "expand-augmented": "expand augmented assignments", "invert-conditionals": "invert conditionals",
              "inject-noise": "inject docstring + dead store", "all-combined": "all five at once"}
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.66, 1.6))
    ys = list(range(len(order)))
    for y, t in zip(ys, order):
        ax.plot([0.5, 1], [y, y], color=RULE, lw=0.5, ls=(0, (1, 2.4)), zorder=1)
        ax.scatter(dev["by_transform"][t]["top1_stability"], y, s=18, color=SLATE, zorder=3,
                   linewidths=0, marker="o")
        if held is not None:
            ax.scatter(held["by_transform"][t]["top1_stability"], y, s=16, zorder=3,
                       facecolors="white", edgecolors=BRICK, linewidths=0.8, marker="o")
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty[t] for t in order], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(len(order) - 0.4, -0.8)
    ax.set_xlabel("top-1 diagnosis unchanged under the rewrite")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    handles = [plt.Line2D([], [], color=SLATE, marker="o", ls="", ms=3.6,
                          label=f"development ({int(dev['overall']['n'])} mutants)")]
    if held is not None:
        handles.append(plt.Line2D([], [], color=BRICK, marker="o", ls="", ms=3.6,
                                  markerfacecolor="white", markeredgewidth=0.8,
                                  label=f"held-out ({int(held['overall']['n'])} mutants)"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              fontsize=6.2, columnspacing=1.4)
    save(fig, "fig_robustness")


# 7. worked example: aligned trajectories and the divergence point
def fig_trace() -> None:
    """One small panel per matched variable, reference against student. Six
    lines on one pair of axes crossed each other too much to read.
    """
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
    n = max(len(s_path), len(r_path))
    first = next((k for k, (a, b) in enumerate(zip(s_path, r_path)) if a.values != b.values), None)
    seen = [v for st in list(r_path) + list(s_path) for v in st.values if v is not None]
    low, high = min(seen), max(seen)
    colours = {"lo": SLATE, "hi": BRICK, "mid": TEAL}
    # each name sits on its own reference line, at a checkpoint where the
    # other two lines are far enough away
    name_at = {"lo": 1.55, "hi": 2.45, "mid": 2.55}
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.66, 1.75), layout="constrained")
    ax.grid(True, axis="y", which="major", color=RULE, linewidth=0.4, zorder=0)
    ax.grid(True, axis="y", which="minor", color="#e9e7e1", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    if first is not None:
        ax.axvline(first, color=GREY, lw=0.5, ls=(0, (1.4, 1.8)), zorder=1)
    for role in roles:
        i = roles.index(role)
        ref_vals = [st.values[i] for st in r_path]
        stu_vals = [st.values[i] for st in s_path]
        # the gap the student carries, drawn at every checkpoint
        for x, (a, b) in enumerate(zip(ref_vals, stu_vals)):
            if a is not None and b is not None and a != b:
                ax.plot([x, x], [b, a], color=GREY, lw=0.5, zorder=2)
        ax.plot(range(len(ref_vals)), ref_vals, color=colours[role], lw=1.2, marker="o", ms=2.8,
                markeredgewidth=0, zorder=3)
        ax.plot(range(len(stu_vals)), stu_vals, color=colours[role], lw=1.1, ls=(0, (2.4, 1.6)),
                marker="o", ms=2.8, markerfacecolor="white", markeredgecolor=colours[role],
                markeredgewidth=0.8, zorder=3)
        known = [j for j, v in enumerate(ref_vals) if v is not None]
        at = name_at.get(role, (known[0] + known[-1]) / 2)
        left = max(j for j in known if j <= at)
        right = min(j for j in known if j >= at)
        y = ref_vals[left] if left == right else (ref_vals[left]
                                                  + (ref_vals[right] - ref_vals[left]) * (at - left))
        ax.text(at, y, role, color=colours[role], fontsize=6.6, family="monospace",
                ha="center", va="center", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax.set_ylim(low - 0.45, high + 0.6)
    ax.set_yticks([t for t in range(low, high + 1) if (t - low) % 2 == 0])
    ax.set_yticks([t for t in range(low, high + 1) if (t - low) % 2], minor=True)
    ax.set_xticks(range(1, n))
    ax.set_xticklabels([str(k) for k in range(1, n)])
    ax.set_xlim(0.78, n - 1 + 0.35)
    ax.set_xlabel("loop-invariant checkpoint")
    ax.set_ylabel("value")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0, pad=2)
    if first is not None:
        ax.text(first + 0.08, high + 0.58, "first disagreement", fontsize=6.0, va="top", ha="left",
                color=MUTED)
    # colour says which variable, line style says whose run it is
    key_x, key_y = n - 1 - 0.66, low + 0.92
    for k, (style, label) in enumerate([("-", "reference"), ((0, (2.4, 1.6)), "student")]):
        y = key_y - k * 0.42
        ax.plot([key_x, key_x + 0.2], [y, y], color=GREY, lw=1.0, ls=style, zorder=4)
        ax.text(key_x + 0.26, y, label, fontsize=6.0, color=MUTED, va="center", ha="left")
    save(fig, "fig_trace")


# 8. knowledge graph with blame from one diagnosis
def fig_kg() -> None:
    """Blamed subgraph after one diagnosis. Blame sets both the size and the
    fill of a node, so the shape of the answer reads before the numbers do.
    """
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
            positions[cid] = (depth, (i - (n - 1) / 2) * 2.0)

    def area(cid: str) -> float:
        return 125 + 215 * (blame[cid] / peak)

    def radius(cid: str) -> float:
        return (area(cid) / 3.14159) ** 0.5

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.45))
    ax.axis("off")
    for cid in keep:
        x1, y1 = positions[cid]
        for pre in graph.prerequisites(cid):
            if pre in keep:
                x0, y0 = positions[pre]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.65, alpha=0.8,
                                            connectionstyle="arc3,rad=0.06",
                                            shrinkA=radius(pre) + 1.5, shrinkB=radius(cid) + 3.0,
                                            mutation_scale=5), zorder=1)
    for cid, (x, y) in positions.items():
        b = blame[cid] / peak
        idx = min(len(SEQ) - 1, int(round(b * (len(SEQ) - 1))))
        if cid == target or cid in frontier:
            # a ring outside the fill, with a white gap so it reads on dark fills
            ax.scatter(x, y, s=area(cid) * 1.95, color=BRICK if cid == target else INK,
                       linewidths=0, zorder=2)
            ax.scatter(x, y, s=area(cid) * 1.45, color="white", linewidths=0, zorder=2.5)
        ax.scatter(x, y, s=area(cid), color=SEQ[idx], edgecolors=RULE, linewidths=0.4, zorder=3)
        ax.text(x, y, f"{100 * blame[cid]:.0f}%", ha="center", va="center", fontsize=5.5,
                color="white" if idx >= 4 else INK, zorder=4)
        # the knockout box keeps an arrow from running through a name
        ax.text(x, y - radius(cid) / 72 - 0.24, "\n".join(textwrap.wrap(graph.name(cid), 13)),
                ha="center", va="top", fontsize=5.4, color=INK, zorder=4, linespacing=0.95,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    xs = sorted(columns)
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.62, max(xs) + 0.62)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 0.78)
    for depth in xs:
        ax.text(depth, max(ys) + 0.74, f"depth {depth}", ha="center", va="top", fontsize=5.4,
                color=GREY)
    # a two-item key in the empty strip, instead of a sentence across the foot
    key_x, key_y = min(xs) - 0.3, min(ys) - 1.2
    for dx, (colour, label) in enumerate([(INK, "deficiency frontier"), (BRICK, "intervention target")]):
        cx = key_x + dx * 1.9
        ax.scatter(cx, key_y, s=46, color=colour, linewidths=0, zorder=3)
        ax.scatter(cx, key_y, s=30, color="white", linewidths=0, zorder=3.5)
        ax.scatter(cx, key_y, s=20, color=SEQ[2], linewidths=0, zorder=4)
        ax.text(cx + 0.13, key_y, label, ha="left", va="center", fontsize=5.6, color=MUTED)
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
