# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest
from awkward_docs.docstrings import process_docstring
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.parametrize(
    ("docstring", "expected"),
    [
        ("", ""),
        ("Plain text.", "Plain text."),
        # Inline literals
        ("Use `axis=None`.", "Use ``axis=None``."),
        ("Use ``write(str)``.", "Use ``write(str)``."),
        ("`a` and `b`", "``a`` and ``b``"),
        ("```", "```"),
        # Hyperlink targets
        ("See <<<nested projection>>>.", "See `nested projection`_."),
        ("<<<>>>", "<<<>>>"),
        # Cross-references to ak.*
        ("See #ak.flatten.", "See :py:obj:`ak.flatten`."),
        ("See #ak.contents.Content.", "See :py:obj:`ak.contents.Content`."),
        ("See #ak.str.is_alnum, too", "See :py:obj:`ak.str.is_alnum`, too"),
        ("#ak.", "#ak."),
        ("#ak", "#ak"),
        # Markdown links
        ("[docs](https://awkward-array.org)", "`docs <https://awkward-array.org>`__"),
        ("[docs](http://example.com/a#b)", "`docs <http://example.com/a#b>`__"),
        (
            "[named axes](../../user-guide/how-to-array-properties-named-axis.html)",
            "`named axes <../../user-guide/how-to-array-properties-named-axis.html>`__",
        ),
        ("[here](./page.html)", "`here <./page.html>`__"),
        ("[guide](../user-guide/)", "`guide <../user-guide/>`__"),
        ("[data](./data.json)", "`data <./data.json>`__"),
        ("[here](ak.Array.html#nested)", "`here <ak.Array.html#nested>`__"),
        # Code that looks like a Markdown link stays as it is
        ("class GenProto[T](Protocol):", "class GenProto[T](Protocol):"),
        ("x[i](builder)", "x[i](builder)"),
        ("[a](page.htm)", "[a](page.htm)"),
        ("[a](my page.html)", "[a](my page.html)"),
        # Each line on its own
        ("`a\nb`", "``a\nb``"),
        ("[a\n](https://x)", "[a\n](https://x)"),
    ],
)
def test_process_docstring(docstring, expected):
    assert process_docstring(docstring, (), "ak.Array") == expected


@pytest.mark.parametrize(
    ("docstring", "expected"),
    [
        ("See #to_list.", "See :py:meth:`to_list <ak.Array.to_list>`."),
        (
            "See #mask and #to_list",
            "See :py:meth:`mask <ak.Array.mask>` and :py:meth:`to_list <ak.Array.to_list>`",
        ),
        ("See #other.", "See #other."),
        ("See #ak.flatten.", "See :py:obj:`ak.flatten`."),
    ],
)
def test_process_docstring_member_links(docstring, expected):
    assert process_docstring(docstring, {"to_list", "mask"}, "ak.Array") == expected


def test_process_docstring_member_links_need_members():
    assert process_docstring("See #to_list.", set(), "ak.Array") == "See #to_list."


# Text without the characters the markup uses
plain_text = st.text(
    alphabet=st.characters(blacklist_characters="`<#[", blacklist_categories=("Cs",))
)


@given(plain_text, st.sets(st.from_regex(r"[a-z_]{1,5}", fullmatch=True)))
def test_process_docstring_keeps_plain_text(docstring, members):
    assert process_docstring(docstring, members, "ak.Array") == docstring


@given(st.text(), st.sets(st.from_regex(r"[a-z_]{1,5}", fullmatch=True)))
def test_process_docstring_keeps_the_lines(docstring, members):
    assert process_docstring(docstring, members, "ak.Array").count(
        "\n"
    ) == docstring.count("\n")
