"""Terminal widgets: rules, panels, tables, trees, bars, sparklines, source
listings. Each returns a list of lines instead of printing so reports can be
composed, nested and tested.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..util.text import truncate, wrap
from .ansi import BOLD, PALETTE, RESET, Style, display_width

# box drawing chars: rounded corners for panels, light lines for tables
TL, TR, BL, BR = "╭", "╮", "╰", "╯"
H, V = "─", "│"
TEE_L, TEE_R = "├", "┤"
BRANCH, LAST, PIPE, SPACE = "├─ ", "╰─ ", "│  ", "   "

BLOCKS = " ▁▂▃▄▅▆▇█"
BAR_FULL, BAR_EMPTY = "█", "░"

# the mock-ups in images/ card every section behind a 3px coloured bar; in a
# terminal that is one column of ▌, and the colour says which agent is talking
GUTTER = "▌"
DOT = "●"
CARET = "❯"
CELL_FULL, CELL_EMPTY = "▰", "▱"


def pad(text: str, width: int, align: str = "left") -> str:
    """Pad to width printable columns, ignoring escape codes."""
    deficit = width - display_width(text)
    if deficit <= 0:
        return text
    if align == "right":
        return " " * deficit + text
    if align == "center":
        left = deficit // 2
        return " " * left + text + " " * (deficit - left)
    return text + " " * deficit


def rule(style: Style, label: str = "", width: Optional[int] = None, *,
         colour: str = "primary") -> str:
    """Horizontal rule with an optional label. The colour names the stage, so
    a long report reads in the same three colours the session does.
    """
    width = width or style.width
    if not label:
        return style.paint(H * width, "dim")
    prefix = style.paint(H * 2 + " ", colour)
    text = style.paint(label, colour, bold=True)
    used = 3 + display_width(label) + 1
    return prefix + text + " " + style.paint(H * max(0, width - used), "dim")


def banner(style: Style, title: str, subtitle: str = "", width: Optional[int] = None) -> List[str]:
    """Report header: title block with a full width underline."""
    width = width or style.width
    lines = [style.heading(truncate(title, width))]
    if subtitle:
        lines.append(style.muted(truncate(subtitle, width)))
    lines.append(style.muted(H * width))
    return lines


def panel(
    style: Style,
    body: Sequence[str],
    *,
    title: str = "",
    width: Optional[int] = None,
    colour: str = "muted",
) -> List[str]:
    """Rounded box around already rendered lines."""
    width = width or style.width
    inner = width - 4
    if title:
        label = f" {title} "
        top = (
            style.paint(TL + H, colour)
            + style.paint(label, colour, bold=True)
            + style.paint(H * max(0, width - 3 - display_width(label)) + TR, colour)
        )
    else:
        top = style.paint(TL + H * (width - 2) + TR, colour)
    out = [top]
    for line in body:
        for piece in (line.split("\n") if "\n" in line else [line]):
            out.append(
                style.paint(V, colour) + " " + pad(truncate_styled(piece, inner), inner)
                + " " + style.paint(V, colour)
            )
    out.append(style.paint(BL + H * (width - 2) + BR, colour))
    return out


def truncate_styled(text: str, width: int) -> str:
    """Truncate to width printable columns without cutting an escape code."""
    if display_width(text) <= width:
        return text
    out = []
    used = 0
    index = 0
    while index < len(text):
        if text[index] == "\x1b":
            end = text.find("m", index)
            if end == -1:
                break
            out.append(text[index : end + 1])
            index = end + 1
            continue
        if used >= width - 1:
            out.append("…")
            break
        out.append(text[index])
        used += 1
        index += 1
    return "".join(out) + "\x1b[0m" if "\x1b" in text else "".join(out)


@dataclass
class Column:
    header: str
    align: str = "left"
    width: Optional[int] = None
    colour: Optional[str] = None
    # never shrink below this many columns, use it for identifiers (unreadable
    # once cut) rather than prose
    min_width: int = 0
    # columns get dropped lowest priority first when even the minimum widths do
    # not fit. Dropping one is better than overflowing (wrapping breaks every
    # other column too).
    priority: int = 50


def table(
    style: Style,
    columns: Sequence[Column],
    rows: Sequence[Sequence[str]],
    *,
    width: Optional[int] = None,
    indent: str = "",
) -> List[str]:
    """Aligned table with a dim header rule, no vertical borders."""
    if not rows:
        return [indent + style.muted("(no rows)")]

    budget = (width or style.width) - len(indent)
    columns = list(columns)
    # drop the least important columns until the mandatory widths fit
    while len(columns) > 1:
        floors = [c.width or max(c.min_width, 6, display_width(c.header)) for c in columns]
        if sum(floors) + 2 * (len(columns) - 1) <= budget:
            break
        victim = min(range(len(columns)), key=lambda i: (columns[i].priority, -i))
        keep = [i for i in range(len(columns)) if i != victim]
        columns = [columns[i] for i in keep]
        rows = [[row[i] for i in keep if i < len(row)] for row in rows]

    widths: List[int] = []
    for index, column in enumerate(columns):
        natural = max(
            [display_width(column.header)]
            + [display_width(str(row[index])) for row in rows if index < len(row)]
        )
        widths.append(column.width or natural)

    available = budget - 2 * (len(columns) - 1)
    # shrink whichever column has the most slack above its own header, one at
    # a time, so a narrow id column is never sacrificed for a wide prose one
    floors = [
        column.width or max(column.min_width, 6, display_width(column.header))
        for column in columns
    ]
    guard = 0
    while sum(widths) > available and guard < 4096:
        guard += 1
        slack = [widths[i] - floors[i] for i in range(len(widths))]
        if max(slack) <= 0:
            break
        target = max(range(len(widths)), key=lambda i: slack[i])
        widths[target] -= 1

    header = "  ".join(
        style.muted(pad(column.header, widths[i], column.align), bold=True)
        for i, column in enumerate(columns)
    )
    total = sum(widths) + 2 * (len(columns) - 1)
    ruler = min(total, budget)
    out = [indent + header, indent + style.muted(H * ruler)]
    for row in rows:
        cells = []
        for index, column in enumerate(columns):
            value = str(row[index]) if index < len(row) else ""
            cells.append(pad(truncate_styled(value, widths[index]), widths[index], column.align))
        out.append(indent + "  ".join(cells))
    # last guarantee: nothing overflows whatever the caller asked for
    return [line if display_width(line) <= budget + len(indent)
            else indent + truncate_styled(line[len(indent):], budget)
            for line in out]


def key_values(style: Style, pairs: Sequence[Tuple[str, str]], *, indent: str = "", gap: int = 2) -> List[str]:
    """Two column key: value list with right aligned keys."""
    if not pairs:
        return []
    width = max(display_width(k) for k, _ in pairs)
    return [
        indent + style.muted(pad(key, width, "right")) + " " * gap + value
        for key, value in pairs
    ]


# the mastery bar in images/session-terminal.svg fills teal into green
MASTERY_STOPS = ((45, 212, 191), (134, 239, 172))


def bar(style: Style, value: float, *, width: int = 18, colour: Optional[str] = None) -> str:
    """Proportional bar for a value in [0, 1]. Left unqualified it runs the
    mastery ramp, which reads as one quantity rather than three bands.
    """
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    full = (style.paint(BAR_FULL * filled, colour) if colour
            else style.gradient(BAR_FULL * filled, MASTERY_STOPS))
    return full + style.paint(BAR_EMPTY * (width - filled), "dim")


def sparkline(style: Style, values: Sequence[float], *, colour: str = "primary") -> str:
    """One line distribution sketch."""
    if not values:
        return ""
    low, high = min(values), max(values)
    span = high - low
    if span <= 1e-12:
        return style.paint(BLOCKS[4] * len(values), colour)
    scaled = [int((v - low) / span * (len(BLOCKS) - 1)) for v in values]
    return style.paint("".join(BLOCKS[i] for i in scaled), colour)


def histogram(style: Style, values: Sequence[float], *, bins: int = 12, width: int = 12) -> str:
    """Compact sparkline histogram of a sample."""
    if not values:
        return ""
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return sparkline(style, [1.0] * min(bins, width))
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / (high - low) * bins))
        counts[index] += 1
    return sparkline(style, [float(c) for c in counts])


def tree(
    style: Style,
    root: str,
    children: Callable[[str], Sequence[str]],
    *,
    label: Callable[[str], str],
    depth: int = 3,
    indent: str = "",
) -> List[str]:
    """Labelled tree with box drawing connectors."""
    out = [indent + label(root)]

    def walk(node: str, prefix: str, level: int) -> None:
        if level >= depth:
            return
        kids = list(children(node))
        for position, child in enumerate(kids):
            last = position == len(kids) - 1
            connector = LAST if last else BRANCH
            out.append(indent + style.muted(prefix + connector) + label(child))
            walk(child, prefix + (SPACE if last else PIPE), level + 1)

    walk(root, "", 0)
    return out


def source_listing(
    style: Style,
    source: str,
    *,
    highlight: Optional[int] = None,
    context: int = 3,
    note: str = "",
    indent: str = "",
) -> List[str]:
    """Numbered listing focused on a highlighted line."""
    lines = source.splitlines()
    if not lines:
        return []
    if highlight is None:
        start, stop = 1, len(lines)
    else:
        start = max(1, highlight - context)
        stop = min(len(lines), highlight + context)
    gutter = len(str(stop))
    out: List[str] = []
    for number in range(start, stop + 1):
        text = lines[number - 1].rstrip()
        marker = style.danger("▶") if number == highlight else " "
        number_cell = style.muted(pad(str(number), gutter, "right"))
        rendered = style.code(text) if number != highlight else style.paint(text, "code", bold=True)
        out.append(f"{indent}{marker} {number_cell} {style.muted(V)} {rendered}")
        if number == highlight and note:
            caret_pad = " " * (len(indent) + 2 + gutter + 2)
            out.append(caret_pad + style.danger("└─ " + note))
    return out


def inline(
    style: Style,
    parts: Sequence[Tuple[str, str]],
    *,
    indent: str = "  ",
    separator: str = "  ·  ",
    width: Optional[int] = None,
) -> List[str]:
    """Put labelled facts on one line, or stack them if they don't fit.
    parts is a list of (plain, styled) pairs. Status lines are the usual
    source of overflow because their length depends on the data, so measure
    the plain form first and fall back to one fact per line.
    """
    limit = (width or style.width) - len(indent)
    # measure the rendered form: a badge has its own padding so the plain label
    # understates the width by exactly what overflows
    rendered_width = sum(display_width(rendered) for _, rendered in parts)
    rendered_width += display_width(separator) * (len(parts) - 1)
    if rendered_width <= limit:
        return [indent + separator.join(rendered for _, rendered in parts)]
    out: List[str] = []
    for text, rendered in parts:
        if display_width(rendered) <= limit:
            out.append(indent + rendered)
        else:
            out.extend(indent + line for line in wrap(text, limit))
    return out


def badge(style: Style, text: str, colour: str = "primary") -> str:
    return style.paint(f" {text} ", colour, bold=True)


def bullet(style: Style, text: str, *, indent: str = "  ", colour: str = "muted", width: Optional[int] = None) -> List[str]:
    """Wrapped bullet point."""
    width = (width or style.width) - len(indent) - 2
    wrapped = wrap(text, width)
    if not wrapped:
        return []
    out = [indent + style.paint("• ", colour) + wrapped[0]]
    out.extend(indent + "  " + line for line in wrapped[1:])
    return out


def paragraph(style: Style, text: str, *, indent: str = "  ", width: Optional[int] = None,
              colour: Optional[str] = None) -> List[str]:
    width = (width or style.width) - len(indent)
    return [indent + (style.paint(line, colour) if colour else line) for line in wrap(text, width)]


def block(style: Style, lines: Sequence[str], *, colour: str = "sky",
          indent: str = "  ") -> List[str]:
    """Lines sharing one coloured gutter: the terminal form of the cards in
    images/*.svg. The colour is the meaning, so a reader can tell evidence
    from a prerequisite from a question without reading a word.
    """
    bar = style.paint(GUTTER, colour)
    return [indent + bar + " " + line for line in lines]


def lede(style: Style, label: str, text: str, *, colour: str = "amber") -> str:
    """A badged headline: the pill, then what it is about."""
    return style.pill(label, colour) + "  " + style.paint(text, "text", bold=True)


def segments(style: Style, done: int, total: int, *, colour: str = "amber",
             cap: int = 24) -> str:
    """The check summary as a row of cells rather than a fraction."""
    if total <= 0:
        return ""
    cells = min(total, cap)
    filled = int(round(done / total * cells))
    return (style.paint(CELL_FULL * filled, colour)
            + style.paint(CELL_EMPTY * (cells - filled), "dim"))


def brand_box(style: Style, headline: str, body: Sequence[str],
              width: Optional[int] = None) -> List[str]:
    """A titled box for a standing report such as /status."""
    lines = [style.gradient(headline, bold=True), ""]
    lines += ["  " + line for line in body]
    return panel(style, lines, width=width or style.width, colour="dim")


def prompt_string(style: Style, label: str = "dmas") -> str:
    """The prompt, drawn the way images/session-terminal.svg draws it: the
    gutter, the tool's name, a caret.

    Colour is only safe under GNU readline, which is told with \x01/\x02
    which bytes take no columns. libedit (the macOS build) ignores those
    markers and reorders the escapes, so there the prompt is plain and
    correctly aligned rather than coloured and one character adrift.
    """
    plain = f"  {GUTTER} {label} {CARET} "
    if not style.enabled:
        return plain
    try:
        import readline
    except ImportError:
        return plain
    if getattr(readline, "backend", "") != "readline":
        return plain

    def tint(text: str, colour: str, bold: bool = False) -> str:
        opening = PALETTE[colour].fg(style.truecolour) + (BOLD if bold else "")
        return "\x01" + opening + "\x02" + text + "\x01" + RESET + "\x02"

    return ("  " + tint(GUTTER, "cyan") + " " + tint(label, "cyan", bold=True)
            + " " + tint(CARET, "dim") + " ")


def footnote(style: Style, text: str) -> str:
    """The dim line under the input box."""
    return "  " + style.muted(text)
