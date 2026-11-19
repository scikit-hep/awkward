# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Check in a browser that each page shows the sidebar it was built with.

Usage: python docs/_ext/check_sidebar.py docs/_build/html

Pages load their sidebar from a shared script (``awkward_docs.shared_sidebar``).
The build keeps this only for pages for which ``shared_sidebar.mark`` restores
the sidebar exactly. This script opens each such page in headless Chromium
(Playwright) and compares the sidebar that the script inserts with the result
of ``shared_sidebar.mark``. Requests to other hosts are blocked.

Exits with status 1 and lists the pages if any sidebar differs.
"""

import functools
import http.server
import pathlib
import posixpath
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from awkward_docs import shared_sidebar
from playwright.sync_api import sync_playwright

# Compare the sidebar with the expected HTML, parsed by the browser
_COMPARE = """
expected => {
  const sidebar = document.querySelector(".bd-docs-nav .bd-toc-item");
  if (sidebar === null) {
    return "no sidebar";
  }
  if (sidebar.querySelector("script") !== null) {
    return "the script is still in the sidebar";
  }
  const template = document.createElement("div");
  template.className = sidebar.className;
  template.innerHTML = expected;
  if (sidebar.isEqualNode(template)) {
    return null;
  }
  const a = sidebar.innerHTML, b = template.innerHTML;
  let i = 0;
  while (i < a.length && a[i] === b[i]) {
    i++;
  }
  return "differs at: " + a.slice(Math.max(0, i - 80), i + 80);
}
"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    root = pathlib.Path(argv[0]).resolve()
    expected = {}
    for path in sorted(root.rglob("*.html")):
        rel = path.relative_to(root).as_posix()
        found = shared_sidebar.find_script_tag(path.read_text(encoding="utf-8"))
        if found is None:
            continue
        page, src = found
        script = root / posixpath.normpath(posixpath.join(posixpath.dirname(rel), src))
        tree = shared_sidebar.tree_from_script(script.read_text(encoding="utf-8"))
        expected[rel] = (page, shared_sidebar.mark(tree, page))

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/"

    problems = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        context.route(
            "**/*",
            lambda route: (
                route.continue_()
                if route.request.url.startswith(base)
                else route.abort()
            ),
        )
        browser_page = context.new_page()
        errors = []
        browser_page.on(
            "pageerror", lambda error: errors.append(error.stack or str(error))
        )
        for rel, (page, sidebar) in expected.items():
            if page != rel:
                problems.append(f"{rel}: the sidebar script is for {page}")
                continue
            errors.clear()
            browser_page.goto(base + rel, wait_until="domcontentloaded")
            result = browser_page.evaluate(_COMPARE, sidebar)
            if result is not None:
                problems.append(f"{rel}: {result}")
            problems.extend(
                f"{rel}: error in the sidebar script: {error}"
                for error in errors
                if "_static/sidebar/" in error
            )
        browser.close()
    server.shutdown()

    for problem in problems:
        print(problem)
    print(f"{len(expected)} pages checked; {len(problems)} problems")
    return 1 if problems or not expected else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
