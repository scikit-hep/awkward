# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Checks on the built HTML site that a successful build does not guarantee."""

import posixpath
import re
import zlib
from collections.abc import Iterator, Mapping
from html.parser import HTMLParser
from urllib.parse import unquote, urldefrag

# Names that, as parameter or return types, should link to the Python
# documentation, not to an attribute with the same name in these docs
BUILTIN_TYPES = frozenset(
    {
        "None",
        "bool",
        "bytearray",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "object",
        "set",
        "str",
        "tuple",
        "type",
    }
)


class _Page(HTMLParser):
    """The element ids and the links of an HTML page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        # Internal links: [href, text, whether the link wraps a <code>]
        self.internal: list[list] = []
        self._open: list | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.add(element_id)
        href = attributes.get("href")
        if tag == "a" and href is not None:
            self.hrefs.append(href)
            if "internal" in (attributes.get("class") or "").split():
                self._open = [href, "", False]
                self.internal.append(self._open)
        elif tag == "code" and self._open is not None:
            self._open[2] = True

    def handle_endtag(self, tag):
        if tag == "a":
            self._open = None

    def handle_data(self, data):
        if self._open is not None:
            self._open[1] += data


def _parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    page.close()
    return page


def link_target(page: str, href: str) -> tuple[str, str] | None:
    """Return the file and anchor that a link on ``page`` points to.

    ``page`` is the path of the page within the site, e.g.
    ``user-guide/how-to-convert.html``. Links to other sites, and links
    without a file and anchor, return ``None``.
    """
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:|^//", href):
        return None
    path, fragment = urldefrag(href)
    path = path.split("?", 1)[0]
    if not path:
        return (page, unquote(fragment)) if fragment else None
    resolved = posixpath.join(posixpath.dirname(page), unquote(path))
    if path.endswith("/"):
        resolved = posixpath.join(resolved, "index.html")
    return posixpath.normpath(resolved), unquote(fragment)


def broken_links(
    pages: Mapping[str, str], other_files: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Return the link targets in ``pages`` that do not exist.

    ``pages`` maps the path of each HTML page to its content; ``other_files``
    holds the paths of the other files in the site. The result maps each
    missing target, ``file`` or ``file#anchor``, to a page that links to it.
    """
    parsed = {path: _parse(html) for path, html in pages.items()}
    missing: dict[str, str] = {}
    for page, content in parsed.items():
        for href in content.hrefs:
            target = link_target(page, href)
            if target is None:
                continue
            path, anchor = target
            if path in parsed:
                if anchor and anchor not in parsed[path].ids:
                    missing.setdefault(f"{path}#{anchor}", page)
            elif path not in other_files:
                missing.setdefault(path, page)
    return missing


def builtin_type_links(html: str) -> list[tuple[str, str]]:
    """Return the internal links whose text is a built-in type name.

    Links that wrap a ``<code>`` element are cross-references written in the
    text, e.g. to the method ``ak.ArrayBuilder.list``, and are not returned.
    """
    return [
        (text.strip(), href)
        for href, text, has_code in _parse(html).internal
        if text.strip() in BUILTIN_TYPES and not has_code
    ]


def signatures_with_empty_parameters(markdown: str) -> list[str]:
    """Return the signature headings in a Markdown page with an empty parameter.

    An empty parameter, e.g. ``(data, , behavior=None)``, is what remains
    when the ``*`` or ``/`` separator is dropped. A comma right before a
    closing parenthesis, as in the default value ``('x',)``, is not one.
    """
    return [
        line
        for line in markdown.splitlines()
        if line.startswith("#") and re.search(r"\(\s*,|,\s+,|,\s+\)", line)
    ]


def _inventory_names(inventory: bytes) -> Iterator[str]:
    *_, compressed = inventory.split(b"\n", 4)
    for line in zlib.decompress(compressed).decode("utf-8").splitlines():
        yield line.split(" ", 1)[0]


def missing_awkward_aliases(inventory: bytes) -> list[str]:
    """Return the ``ak.*`` names in ``objects.inv`` without an ``awkward.*`` alias."""
    names = set(_inventory_names(inventory))
    return sorted(
        name
        for name in names
        if name.startswith("ak.") and "awkward." + name[3:] not in names
    )
