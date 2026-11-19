# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Check a built documentation site for problems that a successful build does not rule out.

Usage: python docs/_ext/check_site.py docs/_build/html

Exits with status 1 and lists the problems if any check fails. The checks are
in ``awkward_docs.site_checks``. Link targets that are already broken are
listed in ``known-broken-links.txt`` next to this script; only other broken
targets fail the check.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from awkward_docs import site_checks

_NOT_CHECKED = ("_static/doxygen/", "_downloads/")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    root = pathlib.Path(argv[0])
    files = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    pages = {
        rel: path.read_text(encoding="utf-8", errors="replace")
        for rel, path in files.items()
        if rel.endswith(".html") and not rel.startswith(_NOT_CHECKED)
    }
    other_files = frozenset(rel for rel in files if rel not in pages)
    known = {
        line.strip()
        for line in (pathlib.Path(__file__).parent / "known-broken-links.txt")
        .read_text()
        .splitlines()
        if line.strip() and not line.startswith("#")
    }

    problems = []
    broken = site_checks.broken_links(pages, other_files)
    for target, page in sorted(broken.items()):
        if target not in known:
            problems.append(f"{page}: link to a missing target: {target}")
    for target in sorted(known - set(broken)):
        problems.append(
            f"known-broken-links.txt: no longer broken, remove it: {target}"
        )
    for rel, html in pages.items():
        for text, href in site_checks.builtin_type_links(html):
            problems.append(f"{rel}: the type {text} links within the site, to {href}")
    for rel, path in files.items():
        if rel.endswith(".html.md"):
            for line in site_checks.signatures_with_empty_parameters(
                path.read_text(encoding="utf-8")
            ):
                problems.append(f"{rel}: signature with an empty parameter: {line}")
    for name in site_checks.missing_awkward_aliases(
        (root / "objects.inv").read_bytes()
    ):
        problems.append(f"objects.inv: no awkward.* alias for {name}")

    for problem in problems:
        print(problem)
    print(f"{len(pages)} pages checked; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
