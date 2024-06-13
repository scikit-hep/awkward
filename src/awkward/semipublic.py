# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""
This module provides access to various functions that were originally internal,
but that have proven to be useful for external projects. The functions here are
not necessarily well-documented, but will be supported by future (non-major) releases.

These functions are not available in the top-level `awkward` namespace.
"""

from __future__ import annotations

from awkward import prettyprint as prettyprint
from awkward._do import remove_structure

__all__ = ["remove_structure", "prettyprint"]
