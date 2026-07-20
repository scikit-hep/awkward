from __future__ import annotations

import os

collect_ignore_glob = []

try:
    import hypothesis_awkward  # noqa: F401
except ModuleNotFoundError:
    collect_ignore_glob.append("**/test_*.py")
else:
    from hypothesis import settings

    # Settings not given here fall back to the profile active at registration
    # time. On GitHub Actions, hypothesis auto-loads its built-in "ci" profile
    # (derandomize=True, deadline=None, database=None, print_blob=True).
    settings.register_profile("default", max_examples=200)
    settings.register_profile(
        "nightly",
        max_examples=10_000,
        deadline=None,
        print_blob=True,
        # The inherited "ci" value (True) would test the same examples nightly.
        derandomize=False,
    )
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
