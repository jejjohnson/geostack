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


def test_fenced_lines_cover_a_block_and_its_delimiters() -> None:
    page = "a\n```python\nx = 1\n```\nb\n"
    assert mc.fenced_lines(page) == {2, 3, 4}


def test_dollars_in_a_code_fence_are_not_a_display_finding(tmp_path) -> None:
    pytest.importorskip("cyclopts")
    pytest.importorskip("loguru")
    import check_math as cm

    page = tmp_path / "page.md"
    page.write_text(
        '```python\ntext = """\n$$ not mathematics\n"""\n```\n\n$$ a = b\nmore\n',
        encoding="utf-8",
    )
    findings = cm.check_file(page, cm.Target.github, [])
    assert [(f.line, f.rule) for f in findings if f.rule == "display"] == [
        (7, "display")
    ]


def test_backtick_inline_wraps_plain_inline_maths_only() -> None:
    source = "a $x_n$ and $`y_n`$ and `$z$` code\n\n```math\nx_n\n```\n$$x$$\n"
    expected = "a $`x_n`$ and $`y_n`$ and `$z$` code\n\n```math\nx_n\n```\n$$x$$\n"
    assert mc.backtick_inline(source) == expected


def test_underscores_that_pair_across_inline_maths_are_an_error(tmp_path) -> None:
    pytest.importorskip("cyclopts")
    pytest.importorskip("loguru")
    import check_math

    page = tmp_path / "body.md"
    page.write_text("The mean $\\boldsymbol{\\mu}_x$ and the size $N_{b}$ pair up.\n")
    found = check_math.check_file(page, check_math.Target.github, [])
    assert [f.rule for f in found] == ["inline-emphasis"]
    assert check_math.check_file(page, check_math.Target.myst, []) == []


def test_backticked_inline_maths_is_not_an_emphasis_error(tmp_path) -> None:
    pytest.importorskip("cyclopts")
    pytest.importorskip("loguru")
    import check_math

    page = tmp_path / "body.md"
    page.write_text("The size $`N_{b}`$ and the mean $`\\boldsymbol{\\mu}_x`$.\n")
    assert check_math.check_file(page, check_math.Target.github, []) == []
    lone = tmp_path / "lone.md"
    lone.write_text("One subscript, $\\boldsymbol{\\mu}_x$, pairs with nothing.\n")
    assert check_math.check_file(lone, check_math.Target.github, []) == []


def test_backtick_inline_leaves_a_code_span_over_several_lines_alone() -> None:
    source = "see `first line\n$x_n$\nlast line` and $y_n$\n\nthen $z_n$\n"
    expected = "see `first line\n$x_n$\nlast line` and $`y_n`$\n\nthen $`z_n`$\n"
    assert mc.backtick_inline(source) == expected


def test_emphasis_is_not_paired_across_list_items_or_a_heading(tmp_path) -> None:
    pytest.importorskip("cyclopts")
    pytest.importorskip("loguru")
    import check_math

    page = tmp_path / "list.md"
    page.write_text(
        "# The mean $\\boldsymbol{\\mu}_x$\n"
        "Sizes are $N_{b}$ here.\n"
        "- $\\boldsymbol{\\mu}_x$\n"
        "- $N_{b}$\n"
        "1. $\\boldsymbol{\\Sigma}_x$\n"
    )
    assert check_math.check_file(page, check_math.Target.github, []) == []
    item = tmp_path / "item.md"
    item.write_text("- the mean $\\boldsymbol{\\mu}_x$ and\n  the size $N_{b}$\n")
    found = check_math.check_file(item, check_math.Target.github, [])
    assert [f.rule for f in found] == ["inline-emphasis"]


def test_a_quoted_paragraph_over_several_lines_is_one_paragraph(tmp_path) -> None:
    pytest.importorskip("cyclopts")
    pytest.importorskip("loguru")
    import check_math

    quoted = tmp_path / "quoted.md"
    quoted.write_text("> The mean $\\boldsymbol{\\mu}_x$ and\n> the size $N_{b}$.\n")
    found = check_math.check_file(quoted, check_math.Target.github, [])
    assert [f.rule for f in found] == ["inline-emphasis"]
    apart = tmp_path / "apart.md"
    apart.write_text(
        "The mean $\\boldsymbol{\\mu}_x$.\n"
        "> The size $N_{b}$.\n"
        ">\n"
        "> The noise $\\sigma_y$.\n"
        "> > The step $\\alpha_k$.\n"
    )
    assert check_math.check_file(apart, check_math.Target.github, []) == []


def test_backtick_inline_leaves_an_indented_code_block_alone() -> None:
    source = (
        'Run:\n\n    echo "$HOME$PATH"\n\nthen $x_n$.\n\n'
        "- an item\n\n    its second paragraph, $y_n$\n"
    )
    expected = (
        'Run:\n\n    echo "$HOME$PATH"\n\nthen $`x_n`$.\n\n'
        "- an item\n\n    its second paragraph, $`y_n`$\n"
    )
    assert mc.backtick_inline(source) == expected


def test_strong_emphasis_runs_across_inline_maths_are_an_error(tmp_path) -> None:
    pytest.importorskip("cyclopts")
    pytest.importorskip("loguru")
    import check_math

    page = tmp_path / "strong.md"
    page.write_text("The blocks $x_{**}$ and $y_{**}$ pair up.\n")
    found = check_math.check_file(page, check_math.Target.github, [])
    assert [f.rule for f in found] == ["inline-emphasis"]
    code = tmp_path / "code.md"
    code.write_text("Run:\n\n    a_b = c_d * e_f * g\n")
    assert check_math.check_file(code, check_math.Target.github, []) == []


def test_backtick_inline_leaves_a_fence_inside_a_block_quote_alone() -> None:
    source = '> ~~~sh\n> echo "$HOME$PATH"\n> ~~~\n> and $x_n$\n\nthen $y_n$\n'
    expected = '> ~~~sh\n> echo "$HOME$PATH"\n> ~~~\n> and $`x_n`$\n\nthen $`y_n`$\n'
    assert mc.backtick_inline(source) == expected
    unclosed = "> ```\n> $a_n$\n\nafter the quote, $b_n$\n"
    after = "> ```\n> $a_n$\n\nafter the quote, $`b_n`$\n"
    assert mc.backtick_inline(unclosed) == after


def test_nested_emphasis_does_not_hide_an_opener_that_pairs_in_maths(tmp_path) -> None:
    pytest.importorskip("cyclopts")
    pytest.importorskip("loguru")
    import check_math

    page = tmp_path / "nested.md"
    page.write_text("*Note **important** $x_{*}$\n")
    found = check_math.check_file(page, check_math.Target.github, [])
    assert [f.rule for f in found] == ["inline-emphasis"]
    fine = tmp_path / "fine.md"
    fine.write_text("*Note **important** here*, then $x_{b}$ alone.\n")
    assert check_math.check_file(fine, check_math.Target.github, []) == []


def test_backtick_inline_leaves_a_pre_block_alone() -> None:
    source = '<pre>\necho "$HOME$PATH"\n</pre>\n\nthen $x_n$ and <pre>$a$</pre>\n'
    expected = '<pre>\necho "$HOME$PATH"\n</pre>\n\nthen $`x_n`$ and <pre>$`a`$</pre>\n'
    assert mc.backtick_inline(source) == expected
    one_line = "<pre>$a_n$ and $b_n$</pre>\nnext $c_n$\n"
    assert mc.backtick_inline(one_line) == "<pre>$a_n$ and $b_n$</pre>\nnext $`c_n`$\n"


def test_runs_that_the_rule_of_three_keeps_apart_are_not_an_error(tmp_path) -> None:
    pytest.importorskip("cyclopts")
    pytest.importorskip("loguru")
    import check_math

    kernels = tmp_path / "kernels.md"
    kernels.write_text("The blocks $K_{*X}$ and $K_{**}$ of the kernel matrix.\n")
    assert check_math.check_file(kernels, check_math.Target.github, []) == []
    stars = tmp_path / "stars.md"
    stars.write_text("The blocks $K_{*X}$ and $K_{X*}$ of the kernel matrix.\n")
    found = check_math.check_file(stars, check_math.Target.github, [])
    assert [f.rule for f in found] == ["inline-emphasis"]
