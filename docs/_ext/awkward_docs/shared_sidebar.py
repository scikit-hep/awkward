# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Sidebar navigation stored once per tree instead of in every page.

pydata-sphinx-theme writes the whole navigation tree of a section into the
sidebar of each page of the section. The pages differ only in the markers of
the current page: the ``current`` and ``active`` classes on its entry and on
the entries above it, the open ``<details>`` of those entries, and the link of
its entry, ``href="#"``. In the API reference, 245 pages carry the same 38 KB
tree.

``unmark`` removes the markers from the sidebar of a page and writes the links
relative to the root of the site, which gives the same tree for every page of
the section. The tree is stored in one script file (``script``), which each page
loads (``script_tag``). The script inserts the tree into the page and adds the
markers back, as ``mark`` does. Pages are built with the shared tree only when
``mark`` restores their sidebar exactly.
"""

import hashlib
import html
import json
import pathlib
import posixpath
import re

# The tags and text of an HTML fragment
_TOKEN = re.compile(r"<[^>]*>|[^<]+")
_TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9-]*)((?:\s[^>]*)?)>")
_ATTRIBUTE = re.compile(r'([^\s=/>]+)(?:="([^"]*)")?')
_VOID = frozenset(
    "area base br col embed hr img input link meta source track wbr".split()
)
# Links that do not go to a page of the site: other sites, absolute paths,
# and anchors
_NOT_RELATIVE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:|/|#")

_PLACEHOLDER = "__AWKWARD_SIDEBAR_TREE__"
_SCRIPT = (pathlib.Path(__file__).parent / "sidebar.js").read_text(encoding="utf-8")
_PREFIX, _SUFFIX = _SCRIPT.split(_PLACEHOLDER)
_SCRIPT_TAG = re.compile(
    r'<script data-page="([^"]*)" src="([^"]*_static/sidebar/[0-9a-f]+\.js)"></script>'
)


class _Element:
    """A start tag, with its attributes in the order they appear."""

    def __init__(self, token):
        match = _TAG.fullmatch(token)
        assert match is not None
        self.name = match.group(2).lower()
        self.attributes = dict(_ATTRIBUTE.findall(match.group(3)))

    def classes(self):
        return self.attributes.get("class", "").split()

    def set_classes(self, classes):
        if classes:
            self.attributes["class"] = " ".join(classes)
        else:
            self.attributes.pop("class", None)

    def __str__(self):
        # pydata-sphinx-theme writes the attributes in alphabetical order
        attributes = "".join(
            f' {name}="{value}"' for name, value in sorted(self.attributes.items())
        )
        return f"<{self.name}{attributes}>"


def _parse(fragment):
    """Split ``fragment`` into tokens and find the parent element of each token.

    Returns the tokens, a list in which the start tags are ``_Element`` objects
    and other tokens are strings, and a list with the index of the parent
    element of each token, ``None`` at the top level.
    """
    tokens = []
    parents = []
    open_elements = []
    for token in _TOKEN.findall(fragment):
        match = _TAG.fullmatch(token)
        parents.append(open_elements[-1] if open_elements else None)
        if match is None:
            tokens.append(token)
        elif match.group(1):
            tokens.append(token)
            name = match.group(2).lower()
            while open_elements and tokens[open_elements.pop()].name != name:
                pass
        else:
            element = _Element(token)
            tokens.append(element)
            if element.name not in _VOID and not token.endswith("/>"):
                open_elements.append(len(tokens) - 1)
    return tokens, parents


def _join(tokens, changed, original):
    """Write the tokens back, keeping the text of unchanged tags."""
    return "".join(
        str(token) if i in changed else original[i] for i, token in enumerate(tokens)
    )


def _escape(value):
    """Escape an attribute value as pydata-sphinx-theme does (BeautifulSoup)."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _split_fragment(href):
    path, hash_sign, fragment = href.partition("#")
    return path, hash_sign + fragment


def relative_uri(base, to):
    """Return the link from the page ``base`` to ``to``, as Sphinx writes it.

    Both are paths relative to the root of the site. This follows
    ``sphinx.util.osutil.relative_uri``.
    """
    b = base.split("#")[0].split("/")
    t = to.split("#")[0].split("/")
    while len(b) > 1 and len(t) > 1 and b[0] == t[0]:
        b.pop(0)
        t.pop(0)
    if b == t:
        return ""
    if len(b) == 1 and t == [""]:
        return "./"
    return "../" * (len(b) - 1) + "/".join(t)


def unmark(sidebar, page):
    """Remove the markers of the current page from the sidebar of ``page``.

    ``page`` is the path of the page relative to the root of the site, e.g.,
    ``reference/generated/ak.flatten.html``. The links in the result are relative
    to the root of the site.
    """
    tokens, _ = _parse(sidebar)
    original = _TOKEN.findall(sidebar)
    changed = set()
    directory = posixpath.dirname(page)
    for i, token in enumerate(tokens):
        if not isinstance(token, _Element):
            continue
        if token.name in ("li", "ul", "a"):
            classes = token.classes()
            kept = [c for c in classes if c not in ("current", "active")]
            if kept != classes:
                token.set_classes(kept)
                changed.add(i)
        if token.name == "details" and "open" in token.attributes:
            del token.attributes["open"]
            changed.add(i)
        if token.name == "a":
            if token.attributes.pop("aria-current", None) is not None:
                changed.add(i)
            href = token.attributes.get("href")
            if href is None:
                continue
            target = html.unescape(href)
            if target == "#":
                # The entry of the page itself
                target = page
            elif target.startswith("#"):
                target = page + target
            elif not _NOT_RELATIVE.match(target):
                path, fragment = _split_fragment(target)
                target = posixpath.normpath(posixpath.join(directory, path)) + fragment
            else:
                continue
            token.attributes["href"] = _escape(target)
            changed.add(i)
    return _join(tokens, changed, original)


def mark(tree, page):
    """Add the markers of the current page to a tree from ``unmark``.

    The inverse of ``unmark``: the links are written relative to ``page``, the
    entry of ``page`` links to ``#``, and it and the entries above it are
    marked as current. ``sidebar.js`` does the same in the browser.
    """
    tokens, parents = _parse(tree)
    original = _TOKEN.findall(tree)
    changed = set()
    marked = set()
    for i, token in enumerate(tokens):
        if not (isinstance(token, _Element) and token.name == "a"):
            continue
        href = token.attributes.get("href")
        if href is None:
            continue
        target = html.unescape(href)
        if _NOT_RELATIVE.match(target):
            continue
        path, fragment = _split_fragment(target)
        changed.add(i)
        if path != page:
            token.attributes["href"] = _escape(relative_uri(page, path) + fragment)
            continue
        token.attributes["href"] = _escape(fragment or "#")
        token.attributes["aria-current"] = "page"
        token.set_classes(["current", *token.classes()])
        parent = parents[i]
        while parent is not None:
            if parent not in marked:
                marked.add(parent)
                element = tokens[parent]
                classes = [
                    c for c in element.classes() if c not in ("current", "active")
                ]
                if element.name == "li":
                    element.set_classes(
                        [*classes[:1], "current", "active", *classes[1:]]
                    )
                    changed.add(parent)
                elif element.name == "ul":
                    element.set_classes(["current", *classes])
                    changed.add(parent)
            parent = parents[parent]
    # The <details> of each marked entry is open, including the one with the
    # children of the current page
    for i, token in enumerate(tokens):
        if (
            isinstance(token, _Element)
            and token.name == "details"
            and parents[i] in marked
            and tokens[parents[i]].name == "li"
        ):
            token.attributes["open"] = "open"
            changed.add(i)
    return _join(tokens, changed, original)


def script(tree):
    """Return the script that inserts ``tree`` in place of itself."""
    return _PREFIX + json.dumps(tree) + _SUFFIX


def tree_from_script(text):
    """Return the tree that a script from ``script`` inserts."""
    if not (text.startswith(_PREFIX) and text.endswith(_SUFFIX)):
        raise ValueError("not a sidebar script")
    return json.loads(text[len(_PREFIX) : len(text) - len(_SUFFIX)])


def file_name(text):
    """Return the file name of a script, which changes with its content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] + ".js"


def script_tag(src, page):
    """Return the tag that loads the script at ``src`` into ``page``."""
    return f'<script data-page="{_escape(page)}" src="{_escape(src)}"></script>'


def find_script_tag(page_html):
    """Return the page and source of the sidebar script tag, or ``None``."""
    match = _SCRIPT_TAG.search(page_html)
    if match is None:
        return None
    return html.unescape(match.group(1)), html.unescape(match.group(2))
