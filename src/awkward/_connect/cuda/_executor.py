# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Backward-compatible re-export; the implementation is backend-neutral and
lives in ``awkward._connect.lazy._executor``."""

from __future__ import annotations

from awkward._connect.lazy._executor import IRExecutor

__all__ = ("IRExecutor",)
