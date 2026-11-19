"""Shared helpers: load macros.yml, expand macros, find maths spans in Markdown."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


MACROS_FILE = Path(__file__).resolve().parent.parent / "references" / "macros.yml"


@dataclass(frozen=True)
class Macro:
    name: str
    body: str

    @property
    def n_args(self) -> int:
        found = [int(n) for n in re.findall(r"#(\d)", self.body)]
        return max(found, default=0)


def load_macros(path: Path = MACROS_FILE) -> list[Macro]:
    """Read macros.yml into a list of macros, in file order."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Macro(name=name, body=body) for name, body in raw.items()]


def _read_group(text: str, start: int) -> tuple[str, int] | None:
    """Return the brace group starting at text[start] == '{' and the index after it."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return None


def expand(text: str, macros: list[Macro]) -> str:
    """Replace every \\name{arg} with its definition; repeat until nothing changes."""
    for _ in range(10):
        changed = False
        for macro in macros:
            pattern = re.compile(r"\\" + re.escape(macro.name) + r"(?![A-Za-z])")
            out, pos = [], 0
            for match in pattern.finditer(text):
                if match.start() < pos:
                    continue
                args, end = [], match.end()
                for _ in range(macro.n_args):
                    group = _read_group(text, end)
                    if group is None:
                        break
                    args.append(group[0])
                    end = group[1]
                if len(args) != macro.n_args:
                    continue
                body = macro.body
                for n, arg in enumerate(args, start=1):
                    body = body.replace(f"#{n}", arg)
                out.append(text[pos : match.start()])
                out.append(body)
                pos = end
                changed = True
            out.append(text[pos:])
            text = "".join(out)
        if not changed:
            break
    return text


FENCE = re.compile(r"^\s*(```+|~~~+)\s*(\S*)")
INLINE = re.compile(r"(?<![\\$])\$(`?)(?!\$)(.+?)\1(?<![\\$])\$(?!\$)")
TICK_MATH = re.compile(r"\$`(?!\s)(.+?)(?<!\s)`\$")
# MyST allows a label after the closing dollars: `$$ … $$ (eq-name)`.
LABEL = r"(?:\s*\([^()\s]+\))?"
DOLLARS_LINE = re.compile(r"^\s*\$\$(?!\$)(.+?)\$\$" + LABEL + r"\s*$")
DOLLARS_CLOSE = re.compile(r"^\s*\$\$" + LABEL + r"\s*$")


# A code span may run over several lines of one paragraph, never over a blank line.
CODE_SPAN = re.compile(r"(`+)(?!`)(?:(?!\n[ \t]*\n).)*?(?<!`)\1(?!`)", re.DOTALL)


def _blank(match: re.Match) -> str:
    return re.sub(r"[^\n]", "\0", match.group(0))


def backtick_inline(markdown: str) -> str:
    """Rewrite each plain inline ``$…$`` as ``$`…`$``, the form GitHub's Markdown
    leaves alone. Fenced blocks, code spans and maths already in backticks are
    not touched."""
    fenced = fenced_lines(markdown)
    lines = markdown.split("\n")
    # Code spans are found over the whole document, so one that runs over
    # several lines still protects the maths inside it.
    body = "\n".join("" if n + 1 in fenced else line for n, line in enumerate(lines))
    masked_lines = CODE_SPAN.sub(_blank, TICK_MATH.sub(_blank, body)).split("\n")
    for index, line in enumerate(lines):
        if index + 1 in fenced or "$" not in line:
            continue
        masked = masked_lines[index]
        pieces: list[str] = []
        end = 0
        for match in INLINE.finditer(masked):
            if match.group(1):
                continue
            start, stop = match.span(2)
            pieces.append(line[end : match.start()] + "$`" + line[start:stop] + "`$")
            end = match.end()
        lines[index] = "".join(pieces) + line[end:]
    return "\n".join(lines)


def _opens(match: re.Match) -> tuple[str, str]:
    """The kind of block a fence opens (math or code) and the fence itself."""
    kind = "math" if match.group(2) in {"math", "{math}"} else "code"
    return kind, match.group(1)


def _closes(match: re.Match | None, opening: str) -> bool:
    """A fence closes a block when it has the same character, is at least as long
    as the opening fence, and carries no info string."""
    if match is None or match.group(2):
        return False
    fence = match.group(1)
    return fence[0] == opening[0] and len(fence) >= len(opening)


def fenced_lines(markdown: str) -> set[int]:
    """The numbers of the lines inside a code or math fence, delimiters included."""
    inside: set[int] = set()
    fence: tuple[str, str] | None = None
    for number, line in enumerate(markdown.splitlines(), start=1):
        match = FENCE.match(line)
        if fence is None:
            if match:
                fence = _opens(match)
                inside.add(number)
            continue
        inside.add(number)
        if _closes(match, fence[1]):
            fence = None
    return inside


@dataclass(frozen=True)
class Span:
    line: int
    text: str
    display: bool
    in_table: bool


def math_spans(markdown: str) -> list[Span]:
    """Find inline $…$, $`…`$, ```math, ```{math} and $$…$$ spans, with line numbers."""
    spans: list[Span] = []
    lines = markdown.splitlines()
    fence: tuple[str, str] | None = None
    block: list[str] = []
    block_start = 0
    in_dollars = False
    for number, line in enumerate(lines, start=1):
        match = FENCE.match(line)
        if fence is None and match and not in_dollars:
            fence = _opens(match)
            block, block_start = [], number + 1
            continue
        if fence is not None:
            if _closes(match, fence[1]):
                if fence[0] == "math":
                    body = "\n".join(b for b in block if not b.strip().startswith(":"))
                    spans.append(Span(block_start, body, True, False))
                fence = None
            else:
                block.append(line)
            continue
        if in_dollars:
            if DOLLARS_CLOSE.match(line):
                spans.append(Span(block_start, "\n".join(block), True, False))
                in_dollars = False
            else:
                block.append(line)
            continue
        if line.strip() == "$$":
            block, block_start = [], number + 1
            in_dollars = True
            continue
        one_line = DOLLARS_LINE.match(line)
        if one_line:
            spans.append(Span(number, one_line.group(1).strip(), True, False))
            continue
        in_table = line.lstrip().startswith("|")
        stripped = re.sub(r"``.+?``", lambda m: " " * len(m.group(0)), line)
        for tick in TICK_MATH.finditer(stripped):
            spans.append(Span(number, tick.group(1), False, in_table))
        stripped = TICK_MATH.sub(lambda m: " " * len(m.group(0)), stripped)
        stripped = re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), stripped)
        for inline in INLINE.finditer(stripped):
            spans.append(Span(number, inline.group(2), False, in_table))
    return spans


def expand_markdown(markdown: str, macros: list[Macro]) -> str:
    """Expand macros outside code: fenced blocks and `inline code` are untouched."""
    out: list[str] = []
    fence: tuple[str, str] | None = None
    for line in markdown.splitlines(keepends=True):
        match = FENCE.match(line)
        if fence is None and match:
            fence = _opens(match)
            out.append(line)
            continue
        if fence is not None:
            if _closes(match, fence[1]):
                fence = None
                out.append(line)
            else:
                out.append(expand(line, macros) if fence[0] == "math" else line)
            continue
        protected: list[str] = []

        def hold_text(text: str, protected: list[str] = protected) -> str:
            protected.append(text)
            return f"\x00{len(protected) - 1}\x00"

        def hold(m: re.Match) -> str:
            return hold_text(m.group(0))

        held = re.sub(r"``.+?``", hold, line)
        held = TICK_MATH.sub(
            lambda m: hold_text("$`" + expand(m.group(1), macros) + "`$"), held
        )
        held = re.sub(r"`[^`]*`", hold, held)
        expanded = expand(held, macros)
        out.append(
            re.sub(
                r"\x00(\d+)\x00",
                lambda m, protected=protected: protected[int(m.group(1))],
                expanded,
            )
        )
    return "".join(out)
