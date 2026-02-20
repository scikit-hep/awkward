from __future__ import annotations

collect_ignore_glob = []

try:
    import hypothesis_awkward  # noqa: F401
except ModuleNotFoundError:
    collect_ignore_glob.append("**/test_*.py")
