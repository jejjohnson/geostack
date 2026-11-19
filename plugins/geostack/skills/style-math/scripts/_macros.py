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
    fence: str | None = None
    block: list[str] = []
    block_start = 0
    in_dollars = False
    for number, line in enumerate(lines, start=1):
        match = FENCE.match(line)
        if fence is None and match:
            fence = "math" if match.group(2) in {"math", "{math}"} else "code"
            block, block_start = [], number + 1
            continue
        if fence is not None:
            if match and not match.group(2):
                if fence == "math":
                    body = "\n".join(b for b in block if not b.strip().startswith(":"))
                    spans.append(Span(block_start, body, True, False))
                fence = None
            else:
                block.append(line)
            continue
        if line.strip() == "$$":
            if in_dollars:
                spans.append(Span(block_start, "\n".join(block), True, False))
            else:
                block, block_start = [], number + 1
            in_dollars = not in_dollars
            continue
        if in_dollars:
            block.append(line)
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
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        match = FENCE.match(line)
        if fence is None and match:
            fence = "math" if match.group(2) in {"math", "{math}"} else "code"
            out.append(line)
            continue
        if fence is not None:
            if match and not match.group(2):
                fence = None
                out.append(line)
            else:
                out.append(expand(line, macros) if fence == "math" else line)
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
