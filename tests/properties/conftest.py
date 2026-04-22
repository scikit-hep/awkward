from __future__ import annotations

collect_ignore_glob = []

try:
    import hypothesis_awkward  # noqa: F401
except ModuleNotFoundError:
    collect_ignore_glob.append("**/test_*.py")
else:
    from hypothesis import Phase, settings

    # Disable the `explain` phase in Hypothesis testing.
    #
    # This phase has an issue
    # https://github.com/HypothesisWorks/hypothesis/issues/4708.
    #
    # We encountered errors because of this issue, e.g.,
    # https://github.com/scikit-hep/awkward/pull/3891#issuecomment-4055221565
    #
    # We can remove these lines when the issue is resolved.
    phases = set(Phase) - {Phase.explain}
    settings.register_profile("default", phases=phases)
    settings.load_profile("default")
