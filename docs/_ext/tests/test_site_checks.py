# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import zlib

import pytest
from awkward_docs.site_checks import (
    BUILTIN_TYPES,
    broken_links,
    builtin_type_links,
    link_target,
    missing_awkward_aliases,
    signatures_with_empty_parameters,
)
from hypothesis import given
from hypothesis import strategies as st
from strategies import dotted_names, identifiers


@pytest.mark.parametrize(
    ("page", "href", "expected"),
    [
        ("index.html", "user-guide/index.html", ("user-guide/index.html", "")),
        ("user-guide/a.html", "b.html", ("user-guide/b.html", "")),
        (
            "user-guide/a.html",
            "../reference/x.html#ak.flatten",
            ("reference/x.html", "ak.flatten"),
        ),
        ("user-guide/a.html", "#section", ("user-guide/a.html", "section")),
        ("user-guide/a.html", "b.html?v=1#s", ("user-guide/b.html", "s")),
        ("user-guide/a.html", "../", ("index.html", "")),
        ("index.html", "user-guide/", ("user-guide/index.html", "")),
        ("index.html", "./", ("index.html", "")),
        ("a.html", "my%20page.html#a%20b", ("my page.html", "a b")),
        ("a.html", "https://docs.python.org/3/", None),
        ("a.html", "mailto:someone@example.com", None),
        ("a.html", "//cdn.example.com/x.js", None),
        ("a.html", "#", None),
        ("a.html", "", None),
    ],
)
def test_link_target(page, href, expected):
    assert link_target(page, href) == expected


@given(st.lists(identifiers, min_size=1, max_size=3), identifiers)
def test_link_target_of_a_fragment_is_the_page(dirs, anchor):
    page = "/".join(dirs) + ".html"
    assert link_target(page, "#" + anchor) == (page, anchor)


@given(st.lists(identifiers, min_size=0, max_size=3), identifiers)
def test_link_target_of_a_sibling_is_in_the_same_directory(dirs, name):
    page = "/".join([*dirs, "page.html"])
    path, _ = link_target(page, f"{name}.html")
    assert path == "/".join([*dirs, f"{name}.html"])


def html(body: str) -> str:
    return f"<html><body>{body}</body></html>"


def test_broken_links():
    pages = {
        "index.html": html(
            '<a href="a.html">a</a><a href="a.html#x">x</a><a href="a.html#y">y</a>'
        ),
        "a.html": html(
            '<h1 id="x">X</h1><a href="missing.html">m</a><a href="_static/s.css">s</a>'
        ),
    }
    assert broken_links(pages, frozenset({"_static/s.css"})) == {
        "a.html#y": "index.html",
        "missing.html": "a.html",
    }


def test_broken_links_reports_the_first_page():
    pages = {
        "a.html": html('<a href="gone.html">g</a>'),
        "b.html": html('<a href="gone.html">g</a>'),
    }
    assert broken_links(pages) == {"gone.html": "a.html"}


def test_broken_links_reports_the_first_page_for_an_anchor():
    pages = {
        "a.html": html('<a href="c.html#gone">g</a>'),
        "b.html": html('<a href="c.html#gone">g</a>'),
        "c.html": html("<p>c</p>"),
    }
    assert broken_links(pages) == {"c.html#gone": "a.html"}


def test_broken_links_only_follows_a_elements():
    pages = {
        "a.html": html(
            '<link rel="stylesheet" href="missing.css"><area href="gone.html">'
        )
    }
    assert broken_links(pages) == {}


def test_broken_links_ignores_anchors_into_other_files():
    pages = {"a.html": html('<a href="_static/doxygen/x.html#abc">x</a>')}
    assert broken_links(pages, frozenset({"_static/doxygen/x.html"})) == {}


def test_broken_links_of_a_consistent_site():
    pages = {
        "index.html": html(
            '<a href="#top">t</a><div id="top"></div><a href="user-guide/">u</a>'
        ),
        "user-guide/index.html": html(
            '<a href="../index.html#top">i</a><a href="https://x.org/#gone">x</a>'
        ),
    }
    assert broken_links(pages) == {}


TYPE_LINK = '<a class="reference internal" href="{href}"><em>{text}</em></a>'
METHOD_LINK = (
    '<a class="reference internal" href="{href}">'
    '<code class="xref py py-meth docutils literal notranslate"><span class="pre">{text}</span></code></a>'
)


def test_builtin_type_links():
    page = html(
        TYPE_LINK.format(
            href="ak.prettyprint.FormatterOptions.html#ak.prettyprint.FormatterOptions.bool",
            text="bool",
        )
        + '<a class="reference internal" href="ak.Array.html#ak.Array.type"><span class="pre">type</span></a>'
        + METHOD_LINK.format(href="#ak.ArrayBuilder.list", text="list")
        + TYPE_LINK.format(href="ak.Array.html", text="ak.Array")
        + '<a class="reference external" href="https://docs.python.org/3/library/functions.html#int"><em>int</em></a>'
        + '<a href="other.html">str</a>'
    )
    assert builtin_type_links(page) == [
        (
            "bool",
            "ak.prettyprint.FormatterOptions.html#ak.prettyprint.FormatterOptions.bool",
        ),
        ("type", "ak.Array.html#ak.Array.type"),
    ]


def test_builtin_type_links_text_ends_with_the_link():
    page = html(TYPE_LINK.format(href="x.html", text="bool") + " and more text")
    assert builtin_type_links(page) == [("bool", "x.html")]


def test_builtin_type_links_strips_the_text():
    page = html('<a class="reference internal" href="x.html"><em> bool </em></a>')
    assert builtin_type_links(page) == [("bool", "x.html")]


@given(st.sampled_from(sorted(BUILTIN_TYPES)), dotted_names)
def test_builtin_type_links_finds_each_builtin_type(name, target):
    page = html(TYPE_LINK.format(href=f"{target}.html", text=name))
    assert builtin_type_links(page) == [(name, f"{target}.html")]


@given(st.sampled_from(sorted(BUILTIN_TYPES)), dotted_names)
def test_builtin_type_links_ignores_method_links(name, target):
    assert (
        builtin_type_links(html(METHOD_LINK.format(href=f"#{target}", text=name))) == []
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("#### *class* ak.Array(data, , behavior=None)", True),
        ("#### ak.flatten(, array)", True),
        ("#### ak.flatten(array, )", True),
        ("#### ak.f(a,  , b)", True),
        ("#### ak.f(a,, b)", False),
        (
            "#### \\_str_parameters_exclude *: tuple[str, ...]* *= ('_\\_categorical_\\_',)*",
            False,
        ),
        ("#### *class* ak.Array(data, \\*, behavior=None)", False),
        ("#### ak.flatten(array, axis=1, \\*, highlevel=True)", False),
        ("#### ak.to_list(array, /)", False),
        ("#### ak.ArrayBuilder()", False),
        ("ak.flatten(array, , axis)", False),
        ("Some text (a, , b)", False),
    ],
)
def test_signatures_with_empty_parameters(line, expected):
    markdown = f"intro\n{line}\noutro\n"
    assert signatures_with_empty_parameters(markdown) == ([line] if expected else [])


HEADER = (
    b"# Sphinx inventory version 2\n# Project: Awkward Array\n# Version: 2.15\n"
    b"# The remainder of this file is compressed using zlib.\n"
)


def inventory(names: list[str]) -> bytes:
    body = "".join(f"{name} py:function 1 x.html#$ -\n" for name in names)
    return HEADER + zlib.compress(body.encode("utf-8"))


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ([], []),
        (["ak.flatten", "awkward.flatten"], []),
        (["ak.flatten"], ["ak.flatten"]),
        (["ak.flatten", "ak.Array", "awkward.Array", "index"], ["ak.flatten"]),
        (["awkward.extra", "numpy.ndarray"], []),
    ],
)
def test_missing_awkward_aliases(names, expected):
    assert missing_awkward_aliases(inventory(names)) == expected


@given(st.sets(dotted_names, max_size=10), st.sets(dotted_names, max_size=10))
def test_missing_awkward_aliases_are_the_ak_names_without_an_alias(
    ak_names, alias_names
):
    names = [f"ak.{n}" for n in ak_names] + [f"awkward.{n}" for n in alias_names]
    expected = sorted(f"ak.{n}" for n in ak_names - alias_names)
    assert missing_awkward_aliases(inventory(names)) == expected
