"""The wordmark the session opens with.

images/DMAS_Logo.png is a bold rounded DMAS running teal to blue to amber,
under three pills naming the agents. A terminal cannot do rounded, but it can
do the letters and it can do the ramp, so the banner is drawn from block
characters and coloured by column. The three pills below it carry the same
colours the rest of the interface uses for the same three things.
"""

from typing import List, Optional, Sequence, Tuple

from .ansi import LOGO_STOPS, Style

# one entry per letter, every row the same width so a column maps to one
# position on the colour ramp
LETTERS = {
    "D": (
        "██████ ",
        "██   ██",
        "██   ██",
        "██   ██",
        "██████ ",
    ),
    "M": (
        "███   ███",
        "████ ████",
        "██ ███ ██",
        "██  █  ██",
        "██     ██",
    ),
    "A": (
        " █████ ",
        "██   ██",
        "███████",
        "██   ██",
        "██   ██",
    ),
    "S": (
        " ██████",
        "██     ",
        " █████ ",
        "     ██",
        "██████ ",
    ),
}
HEIGHT = 5
KERN = " "

# the agents, in the order the pipeline runs them, with the colour each one
# keeps everywhere else in the interface
AGENTS: Tuple[Tuple[str, str], ...] = (
    ("Socratic Evaluator", "amber"),
    ("Knowledge Graph", "sky"),
    ("Cognitive Intervention", "teal"),
)
DOT = "●"


def wordmark(style: Style, word: str = "DMAS",
             stops: Sequence = LOGO_STOPS) -> List[str]:
    """The word drawn in block letters, ramped left to right."""
    rows = [""] * HEIGHT
    for position, char in enumerate(word.upper()):
        glyph = LETTERS.get(char)
        if glyph is None:
            continue
        for row in range(HEIGHT):
            rows[row] += (KERN if position else "") + glyph[row]
    return [style.gradient(row, stops) for row in rows]


def agent_row(style: Style, width: Optional[int] = None) -> str:
    """The three pills from the logo, as dots and names on one line."""
    parts = [
        style.paint(DOT, colour) + " " + style.paint(name, "text")
        for name, colour in AGENTS
    ]
    separator = style.paint("   ", "dim")
    line = separator.join(parts)
    from .ansi import display_width

    if width and display_width(line) > width:
        # a narrow terminal gets the dots without the names rather than a wrap
        line = separator.join(style.paint(DOT, colour) for _name, colour in AGENTS)
    return line
