# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Backend-neutral lazy-evaluation machinery for Awkward Arrays.

This package contains everything that does not depend on a particular
compute backend:

- ``core``: the minimal IR node protocol (``IRNode.lower``), demand-driven
  memoized execution, and graph traversal utilities.
- ``_ir``: the operator-level IR (``OpType`` nodes) built by
  ``LazyAwkwardArray``.
- ``_executor``: an interpreter for the operator-level IR that dispatches
  to eager Awkward operations (which themselves dispatch to the array's
  backend).
- ``_lazy_impl``: the user-facing ``LazyAwkwardArray`` wrapper and
  ``lazy()`` entry point.
- ``_layout``: layout utilities (``empty_like``, ``reconstruct_with_offsets``)
  shared by the backend-specific fast paths.

Backend-specific nodes and kernels live in ``awkward._connect.cuda``
(CCCL / cuda.compute) and ``awkward._connect.cpu`` (NumPy).
"""

from __future__ import annotations

from awkward._connect.lazy._lazy_impl import LazyAwkwardArray, lazy
from awkward._connect.lazy.core import (
    Input,
    IRNode,
    compute,
    reset_cache,
    topological_order,
    walk,
)

__all__ = (
    "IRNode",
    "Input",
    "LazyAwkwardArray",
    "compute",
    "lazy",
    "reset_cache",
    "topological_order",
    "walk",
)
