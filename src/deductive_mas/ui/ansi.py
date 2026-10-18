"""ANSI colours and the colour theme.

The terminal is the interface so a few rules: colour means something
(severity, belief, pass/fail), nothing else is coloured. NO_COLOR, a pipe,
a dumb terminal or --color never all fall back to plain text with the same
layout. Every widget measures width with display_width which ignores
escape codes, so columns line up.
"""

import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from typing import Optional, Protocol as _Protocol, Sequence


class _TTYStream(_Protocol):
    def isatty(self) -> bool:
        ...


class _InputStream(_Protocol):
    def readline(self) -> str:
        ...


class _OutputStream(_Protocol):
    def write(self, text: str, /) -> object:
        ...

    def flush(self) -> object:
        ...


_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m|\x1b\]8;;[^\x07]*\x07")

RESET = "\x1b[0m"


# capability detection
def supports_colour(mode: str = "auto", stream: Optional[_TTYStream] = None) -> bool:
    """Should we emit escape codes at all."""
    if mode == "never":
        return False
    if mode == "always":
        return True
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") in ("dumb", ""):
        return False
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def supports_truecolour() -> bool:
    return os.environ.get("COLORTERM", "") in ("truecolor", "24bit")


def terminal_width(default: int = 96, maximum: int = 108) -> int:
    try:
        width = shutil.get_terminal_size((default, 24)).columns
    except OSError:
        width = default
    return max(52, min(width - 1, maximum))


def display_width(text: str) -> int:
    """Printable width, ignoring escapes, wide glyphs count as two."""
    stripped = _ESCAPE_RE.sub("", text)
    width = 0
    for char in stripped:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def strip(text: str) -> str:
    return _ESCAPE_RE.sub("", text)


# theme
@dataclass(frozen=True)
class Colour:
    """A colour with a 24 bit and a 256 colour version."""

    rgb: tuple
    xterm: int

    def fg(self, truecolour: bool) -> str:
        if truecolour:
            r, g, b = self.rgb
            return f"\x1b[38;2;{r};{g};{b}m"
        return f"\x1b[38;5;{self.xterm}m"

    def bg(self, truecolour: bool) -> str:
        if truecolour:
            r, g, b = self.rgb
            return f"\x1b[48;2;{r};{g};{b}m"
        return f"\x1b[48;5;{self.xterm}m"


# The palette is the project's own, lifted from images/DMAS_Logo.png and the
# terminal mock-ups in images/*.svg, and it carries the same meaning there as
# here: amber is evidence, blue is the prerequisite it points at, mint is the
# question the learner gets back. Violet is a rung of the ladder.
PALETTE = {
    "amber":    Colour((251, 191, 36), 214),   # evidence, the diagnosis
    "sky":      Colour((56, 189, 248), 39),    # the knowledge graph
    "teal":     Colour((94, 234, 212), 86),    # the question, a repaired belief
    "violet":   Colour((167, 139, 250), 141),  # one rung of the ladder
    "cyan":     Colour((103, 232, 249), 87),   # the prompt
    "mint":     Colour((153, 246, 228), 122),
    "blue":     Colour((147, 197, 253), 111),
    "lilac":    Colour((196, 181, 253), 183),  # file names, source listings
    "green":    Colour((134, 239, 172), 114),
    "red":      Colour((255, 107, 122), 210),
    "text":     Colour((216, 227, 243), 189),
    "muted":    Colour((141, 165, 197), 110),
    "dim":      Colour((77, 102, 136), 60),
}

# the names the reports were written against, kept so meaning stays one word
PALETTE["primary"] = PALETTE["sky"]
PALETTE["accent"] = PALETTE["violet"]
PALETTE["success"] = PALETTE["green"]
PALETTE["warning"] = PALETTE["amber"]
PALETTE["danger"] = PALETTE["red"]
PALETTE["code"] = PALETTE["lilac"]

# the wordmark runs through these left to right, the way the logo does
LOGO_STOPS = ((45, 212, 191), (56, 189, 248), (129, 173, 255), (251, 191, 36))

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"
UNDERLINE = "\x1b[4m"


class Style:
    """Renders styled text, or plain text when colour is off."""

    def __init__(self, mode: str = "auto", stream: Optional[_TTYStream] = None) -> None:
        self.enabled = supports_colour(mode, stream)
        self.truecolour = self.enabled and supports_truecolour()
        self.width = terminal_width()

    # basics
    def paint(self, text: str, colour: Optional[str] = None, *, bold: bool = False,
              dim: bool = False, italic: bool = False, underline: bool = False) -> str:
        if not self.enabled or not text:
            return text
        prefix = ""
        if colour and colour in PALETTE:
            prefix += PALETTE[colour].fg(self.truecolour)
        if bold:
            prefix += BOLD
        if dim:
            prefix += DIM
        if italic:
            prefix += ITALIC
        if underline:
            prefix += UNDERLINE
        return f"{prefix}{text}{RESET}" if prefix else text

    # drawing
    def rgb(self, r: int, g: int, b: int, *, background: bool = False) -> str:
        """An escape for one exact colour, used by the gradient."""
        if not self.enabled:
            return ""
        if self.truecolour:
            return f"\x1b[{48 if background else 38};2;{r};{g};{b}m"
        # the 6x6x6 cube is close enough for a wordmark on a 256 colour terminal
        cube = [round(c / 255 * 5) for c in (r, g, b)]
        return f"\x1b[{48 if background else 38};5;{16 + 36 * cube[0] + 6 * cube[1] + cube[2]}m"

    def gradient(self, text: str, stops: Sequence = LOGO_STOPS, *, bold: bool = False) -> str:
        """Run a colour ramp left to right across the text.
        Blanks are skipped so a drawn letter keeps the ramp of its own column
        rather than of the gap before it.
        """
        if not self.enabled or not text:
            return text
        span = max(1, len(text) - 1)
        out = []
        for index, char in enumerate(text):
            if char == " ":
                out.append(char)
                continue
            position = index / span * (len(stops) - 1)
            low = min(int(position), len(stops) - 2)
            ratio = position - low
            channels = [
                round(stops[low][c] + (stops[low + 1][c] - stops[low][c]) * ratio)
                for c in range(3)
            ]
            out.append(self.rgb(*channels) + (BOLD if bold else "") + char)
        return "".join(out) + RESET

    def pill(self, text: str, colour: str) -> str:
        """A label on its own ground, the way the mock-ups badge a verdict."""
        label = f" {text} "
        if not self.enabled:
            return label
        shade = PALETTE[colour].rgb
        dark = tuple(round(c * 0.22) for c in shade)
        return self.rgb(*dark, background=True) + PALETTE[colour].fg(self.truecolour) + BOLD + label + RESET

    # shortcuts
    def primary(self, text: str, **kwargs: bool) -> str:
        return self.paint(text, "primary", **kwargs)

    def accent(self, text: str, **kwargs: bool) -> str:
        return self.paint(text, "accent", **kwargs)

    def success(self, text: str, **kwargs: bool) -> str:
        return self.paint(text, "success", **kwargs)

    def warning(self, text: str, **kwargs: bool) -> str:
        return self.paint(text, "warning", **kwargs)

    def danger(self, text: str, **kwargs: bool) -> str:
        return self.paint(text, "danger", **kwargs)

    def muted(self, text: str, **kwargs: bool) -> str:
        return self.paint(text, "muted", **kwargs)

    def code(self, text: str, **kwargs: bool) -> str:
        return self.paint(text, "code", **kwargs)

    def heading(self, text: str) -> str:
        return self.paint(text, "primary", bold=True)

    # extras
    def severity(self, level: str) -> str:
        colour = {
            "critical": "danger", "major": "danger",
            "minor": "warning", "info": "muted",
        }.get(level.lower(), "muted")
        return self.paint(level.upper(), colour, bold=level.lower() in ("critical", "major"))

    def belief(self, value: float) -> str:
        """Colour a confidence by how much weight it carries."""
        colour = "success" if value >= 0.66 else "warning" if value >= 0.4 else "muted"
        return self.paint(f"{value:.2f}", colour, bold=value >= 0.66)

    def link(self, text: str, url: str) -> str:
        if not self.enabled:
            return text
        return f"\x1b]8;;{url}\x07{text}\x1b]8;;\x07"
