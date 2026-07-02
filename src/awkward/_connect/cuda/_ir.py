# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Backward-compatible re-export; the implementation is backend-neutral and
lives in ``awkward._connect.lazy._ir``."""

from __future__ import annotations

from awkward._connect.lazy._ir import (
    BinaryOpNode,
    CombinationsNode,
    ComparisonNode,
    ConstantNode,
    FilterNode,
    GetItemNode,
    InputNode,
    IRNode,
    MapNode,
    OpType,
    ReduceNode,
    SelectListsNode,
)

__all__ = (
    "BinaryOpNode",
    "CombinationsNode",
    "ComparisonNode",
    "ConstantNode",
    "FilterNode",
    "GetItemNode",
    "IRNode",
    "InputNode",
    "MapNode",
    "OpType",
    "ReduceNode",
    "SelectListsNode",
)
