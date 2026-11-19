"""The maths-span parser of the style-math skill's scripts.

`check_math.py` and `expand_macros.py` both read Markdown through
`plugins/geostack/skills/style-math/scripts/_macros.py`; a span the parser
misses is maths nobody checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "geostack"
    / "skills"
    / "style-math"
    / "scripts"
)
pytest.importorskip("yaml")
sys.path.insert(0, str(SCRIPTS))

import _macros as mc


def texts(markdown: str) -> list[str]:
    return [span.text for span in mc.math_spans(markdown)]


def test_a_one_line_dollar_block_is_a_display_span() -> None:
    spans = mc.math_spans("before\n\n$$\\vect{x}^T$$\n\nafter $y$\n")
    assert [(s.line, s.text, s.display) for s in spans] == [
        (3, "\\vect{x}^T", True),
        (5, "y", False),
    ]


def test_a_labelled_one_line_block_keeps_the_label_out_of_the_span() -> None:
    assert texts("$$ a = b $$ (eq-name)\n") == ["a = b"]


def test_a_labelled_closing_delimiter_ends_the_block() -> None:
    page = "$$\na = b\n$$ (eq-first)\n\ntext $c$\n\n$$\nd = e\n$$\n"
    assert texts(page) == ["a = b", "c", "d = e"]


def test_an_inner_shorter_fence_does_not_close_the_outer_one() -> None:
    page = "````markdown\n```math\n\\vect{x}\n```\n````\n\n$\\vect{y}$\n"
    assert texts(page) == ["\\vect{y}"]


def test_a_tilde_fence_is_not_closed_by_backticks() -> None:
    page = "~~~text\n```\n$a$\n~~~\n\n$b$\n"
    assert texts(page) == ["b"]


def test_expansion_resumes_after_a_nested_fence() -> None:
    page = "````markdown\n```math\n\\vect{x}\n```\n````\n\n$\\vect{y}$\n"
    expanded = mc.expand_markdown(page, mc.load_macros())
    assert expanded.endswith("$\\mathbf{y}$\n")
    assert "\\vect{x}" in expanded
