# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Backward-compatible re-export; the implementation is backend-neutral and
lives in ``awkward._connect.lazy._lazy_impl``."""

from __future__ import annotations

from awkward._connect.lazy._lazy_impl import LazyAwkwardArray, lazy

__all__ = ("LazyAwkwardArray", "lazy")
