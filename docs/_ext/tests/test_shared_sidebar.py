# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import json
import posixpath

import pytest
from awkward_docs.shared_sidebar import (
    file_name,
    find_script_tag,
    mark,
    relative_uri,
    script,
    script_tag,
    tree_from_script,
    unmark,
)
from hypothesis import given
from hypothesis import strategies as st
from strategies import identifiers

TOGGLE = (
    '<summary><span class="toctree-toggle" role="presentation">'
    '<i class="fa-solid fa-chevron-down"></i></span></summary>'
)

# The sidebar of reference/generated/ak.flatten.html, shortened
FLATTEN_SIDEBAR = (
    '<p aria-level="2" class="caption" role="heading"><span class="caption-text">'
    "High-level data types</span></p>\n"
    '<ul class="nav bd-sidenav">\n'
    '<li class="toctree-l1"><a class="reference internal" href="ak.Array.html">'
    "ak.Array</a></li>\n"
    "</ul>\n"
    '<ul class="current nav bd-sidenav">\n'
    '<li class="toctree-l1 current active"><a aria-current="page" '
    'class="current reference internal" href="#">ak.flatten</a></li>\n'
    '<li class="toctree-l1"><a class="reference internal" href="../ak.behavior.html">'
    "ak.behavior</a></li>\n"
    "</ul>\n"
)

FLATTEN_TREE = (
    '<p aria-level="2" class="caption" role="heading"><span class="caption-text">'
    "High-level data types</span></p>\n"
    '<ul class="nav bd-sidenav">\n'
    '<li class="toctree-l1"><a class="reference internal" '
    'href="reference/generated/ak.Array.html">ak.Array</a></li>\n'
    "</ul>\n"
    '<ul class="nav bd-sidenav">\n'
    '<li class="toctree-l1"><a class="reference internal" '
    'href="reference/generated/ak.flatten.html">ak.flatten</a></li>\n'
    '<li class="toctree-l1"><a class="reference internal" '
    'href="reference/ak.behavior.html">ak.behavior</a></li>\n'
    "</ul>\n"
)

# The sidebar of user-guide/how-to-convert-numpy.html, shortened
NUMPY_SIDEBAR = (
    '<ul class="current nav bd-sidenav">\n'
    '<li class="toctree-l1 current active has-children"><a class="reference internal" '
    f'href="how-to-convert.html">Converting arrays</a><details open="open">{TOGGLE}'
    '<ul class="current">\n'
    '<li class="toctree-l2 current active"><a aria-current="page" '
    'class="current reference internal" href="#">NumPy</a></li>\n'
    '<li class="toctree-l2"><a class="reference internal" '
    'href="how-to-convert-json.html">JSON</a></li>\n'
    "</ul>\n"
    "</details></li>\n"
    '<li class="toctree-l1 has-children"><a class="reference internal" '
    f'href="how-to-create.html">Creating new arrays</a><details>{TOGGLE}<ul>\n'
    '<li class="toctree-l2"><a class="reference internal" '
    'href="how-to-create-lists.html">Arrays of lists</a></li>\n'
    "</ul>\n"
    "</details></li>\n"
    "</ul>\n"
)


def test_unmark_flat_tree():
    assert unmark(FLATTEN_SIDEBAR, "reference/generated/ak.flatten.html") == (
        FLATTEN_TREE
    )


def test_mark_flat_tree():
    assert mark(FLATTEN_TREE, "reference/generated/ak.flatten.html") == (
        FLATTEN_SIDEBAR
    )


def test_mark_page_not_in_tree():
    # reference/index.html is not an entry of its own sidebar
    assert mark(FLATTEN_TREE, "reference/index.html") == (
        FLATTEN_TREE.replace(
            'href="reference/generated/ak.Array.html"',
            'href="generated/ak.Array.html"',
        )
        .replace(
            'href="reference/generated/ak.flatten.html"',
            'href="generated/ak.flatten.html"',
        )
        .replace('href="reference/ak.behavior.html"', 'href="ak.behavior.html"')
    )


def test_nested_tree():
    page = "user-guide/how-to-convert-numpy.html"
    tree = unmark(NUMPY_SIDEBAR, page)
    assert "current" not in tree
    assert "active" not in tree
    assert "open" not in tree
    assert 'href="user-guide/how-to-convert-numpy.html"' in tree
    assert mark(tree, page) == NUMPY_SIDEBAR


def test_mark_entry_with_children():
    # The <details> of the current page's own entry is open, but the list of
    # its children is not marked
    tree = unmark(NUMPY_SIDEBAR, "user-guide/how-to-convert-numpy.html")
    sidebar = mark(tree, "user-guide/how-to-convert.html")
    assert (
        '<li class="toctree-l1 current active has-children"><a aria-current="page" '
        'class="current reference internal" href="#">Converting arrays</a>'
        f'<details open="open">{TOGGLE}<ul>\n'
    ) in sidebar
    assert unmark(sidebar, "user-guide/how-to-convert.html") == tree


@pytest.mark.parametrize(
    "href",
    [
        "https://awkward-array.org/",
        "mailto:someone@example.org",
        "/doc/main/index.html",
    ],
)
def test_links_out_of_the_site_are_kept(href):
    sidebar = f'<ul class="nav bd-sidenav"><li class="toctree-l1"><a class="reference external" href="{href}">x</a></li></ul>'
    tree = unmark(sidebar, "a/b.html")
    assert tree == sidebar
    assert mark(tree, "a/b.html") == sidebar


def test_fragments():
    sidebar = (
        '<ul class="current"><li class="toctree-l2 current active">'
        '<a aria-current="page" class="current reference internal" href="#section">'
        "x</a></li>"
        '<li class="toctree-l2"><a class="reference internal" href="c.html#other">'
        "y</a></li></ul>"
    )
    tree = unmark(sidebar, "a/b.html")
    assert 'href="a/b.html#section"' in tree
    assert 'href="a/c.html#other"' in tree
    assert mark(tree, "a/b.html") == sidebar


def test_escaped_links():
    sidebar = '<ul><li><a href="x&amp;y.html">x</a></li></ul>'
    tree = unmark(sidebar, "a/b.html")
    assert 'href="a/x&amp;y.html"' in tree
    assert mark(tree, "a/b.html") == sidebar


def test_other_markup_is_kept():
    # Anchors without a link, void and self-closing elements, comments,
    # attributes without a value, and a stray end tag
    sidebar = (
        '<!-- a comment --><ul><li><a name="x">x</a><br><img src="i.png"/>'
        '<a href="b.html" hidden>b</a></li></span></ul>'
    )
    tree = unmark(sidebar, "a/b.html")
    assert tree == sidebar.replace(
        '<a href="b.html" hidden>', '<a hidden="" href="a/b.html">'
    )
    assert mark(tree, "a/c.html") == sidebar.replace(
        '<a href="b.html" hidden>', '<a hidden="" href="b.html">'
    )


@pytest.mark.parametrize(
    ("base", "to", "expected"),
    [
        ("a/b.html", "a/c.html", "c.html"),
        ("a/b/c.html", "a/d.html", "../d.html"),
        ("a/b.html", "a/b/c.html", "b/c.html"),
        ("a/b.html", "c/d.html", "../c/d.html"),
        ("index.html", "a/b.html", "a/b.html"),
        ("a/b.html", "a/b.html", ""),
        ("a/b.html#x", "a/c.html#y", "c.html"),
        ("a.html", "", "./"),
    ],
)
def test_relative_uri(base, to, expected):
    assert relative_uri(base, to) == expected


pages = st.lists(identifiers, min_size=1, max_size=4).map(
    lambda parts: "/".join(parts) + ".html"
)


@given(pages, pages)
def test_relative_uri_resolves_to_target(base, to):
    link = relative_uri(base, to)
    if link == "":
        assert base == to
    else:
        assert posixpath.normpath(posixpath.join(posixpath.dirname(base), link)) == to


# -- Sidebars written as pydata-sphinx-theme writes them ---------------------

titles = st.text(alphabet="abcXYZ &", min_size=1, max_size=8)
entries = st.tuples(pages, titles)
sections = st.lists(
    st.tuples(
        st.one_of(st.none(), titles),
        st.lists(
            st.tuples(pages, titles, st.lists(entries, max_size=3)),
            min_size=1,
            max_size=4,
        ),
    ),
    min_size=1,
    max_size=3,
)


def escape(text):
    return text.replace("&", "&amp;")


def render(tree, page):
    """Write the sidebar of ``page`` for a tree of (caption, entries) sections."""

    def anchor(path, title):
        if path == page:
            return (
                '<a aria-current="page" class="current reference internal" '
                f'href="#">{escape(title)}</a>'
            )
        href = escape(relative_uri(page, path))
        return f'<a class="reference internal" href="{href}">{escape(title)}</a>'

    def li(level, current, extra=""):
        marker = " current active" if current else ""
        return f'<li class="toctree-l{level}{marker}{extra}">'

    out = []
    for caption, items in tree:
        if caption is not None:
            out.append(
                '<p aria-level="2" class="caption" role="heading">'
                f'<span class="caption-text">{escape(caption)}</span></p>\n'
            )
        in_section = any(
            path == page or any(p == page for p, _ in children)
            for path, _, children in items
        )
        out.append(f'<ul class="{"current " * in_section}nav bd-sidenav">\n')
        for path, title, children in items:
            child_current = any(p == page for p, _ in children)
            current = path == page or child_current
            if not children:
                out.append(f"{li(1, current)}{anchor(path, title)}</li>\n")
                continue
            details = '<details open="open">' if current else "<details>"
            ul = '<ul class="current">' if child_current else "<ul>"
            out.append(
                f"{li(1, current, ' has-children')}{anchor(path, title)}"
                f"{details}{TOGGLE}{ul}\n"
            )
            for p, t in children:
                out.append(f"{li(2, p == page)}{anchor(p, t)}</li>\n")
            out.append("</ul>\n</details></li>\n")
        out.append("</ul>\n")
    return "".join(out)


@st.composite
def trees_and_pages(draw):
    tree = draw(sections)
    in_tree = [path for _, items in tree for path, _, _ in items] + [
        p for _, items in tree for _, _, children in items for p, _ in children
    ]
    page = st.one_of(st.sampled_from(in_tree), pages)
    return tree, draw(page), draw(page)


@given(trees_and_pages())
def test_mark_restores_the_sidebar(args):
    tree, page, _ = args
    sidebar = render(tree, page)
    assert mark(unmark(sidebar, page), page) == sidebar


@given(trees_and_pages())
def test_pages_share_the_tree(args):
    tree, page, other = args
    assert unmark(render(tree, page), page) == unmark(render(tree, other), other)


def test_script():
    text = script(FLATTEN_TREE)
    assert "__AWKWARD_SIDEBAR_TREE__" not in text
    assert text.endswith("})(" + json.dumps(FLATTEN_TREE) + ");\n")
    assert tree_from_script(text) == FLATTEN_TREE


@given(st.text())
def test_script_round_trip(tree):
    assert tree_from_script(script(tree)) == tree


def test_tree_from_script_rejects_other_files():
    with pytest.raises(ValueError, match="not a sidebar script"):
        tree_from_script("console.log(1);")


def test_file_name():
    name = file_name(script(FLATTEN_TREE))
    assert name.endswith(".js")
    assert len(name) == len("0123456789abcdef.js")
    assert name == file_name(script(FLATTEN_TREE))
    assert name != file_name(script(NUMPY_SIDEBAR))


def test_script_tag():
    tag = script_tag("../../_static/sidebar/0123456789abcdef.js", "a/b&c.html")
    assert tag == (
        '<script data-page="a/b&amp;c.html" '
        'src="../../_static/sidebar/0123456789abcdef.js"></script>'
    )
    page_html = f'<div class="bd-toc-item navbar-nav">{tag}</div>'
    assert find_script_tag(page_html) == (
        "a/b&c.html",
        "../../_static/sidebar/0123456789abcdef.js",
    )


def test_find_script_tag_without_tag():
    assert find_script_tag('<script src="_static/other.js"></script>') is None
