# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Shared no-op fallback for the optional ``nvtx`` profiling annotations."""

from __future__ import annotations

try:
    import nvtx
except ImportError:

    class nvtx:
        @staticmethod
        def annotate(*args, **kwargs):
            def deco(fn):
                return fn

            return deco


__all__ = ("nvtx",)
