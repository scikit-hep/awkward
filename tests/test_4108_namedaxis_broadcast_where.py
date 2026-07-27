# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward._broadcasting import broadcast_and_apply
from awkward._namedaxis import NAMED_AXIS_KEY, NamedAxesWithDims, _prettify_named_axes
from awkward.contents import ByteMaskedArray
from awkward.index import Index8


def _option_of_regular(values):
    """Build an ``option[N * ...]`` layout: an option node wrapping a regular array."""
    regular = ak.to_regular(ak.Array(values)).layout
    mask = Index8(np.ones(regular.length, dtype=np.int8))
    return ak.Array(ByteMaskedArray(mask, regular, valid_when=True))


def test_where_regular_2d_option_of_regular():
    # Crashed with: ValueError: zip() argument 2 is shorter than argument 1
    cond = ak.to_regular(ak.Array([[True, False], [False, True]]))
    x = _option_of_regular([[1, 2], [3, 4]])
    assert str(x.type) == "2 * option[2 * int64]"
    y = ak.to_regular(ak.Array([[10, 20], [30, 40]]))

    result = ak.where(cond, x, y)
    assert result.to_list() == [[1, 20], [30, 4]]


def test_prepare_contexts_independent():
    depth_context, lateral_context = NamedAxesWithDims.prepare_contexts(
        [ak.Array([1, 2, 3]), ak.Array([4, 5, 6])]
    )
    depth = depth_context[NAMED_AXIS_KEY]
    lateral = lateral_context[NAMED_AXIS_KEY]
    # mutate depth; lateral must be unaffected
    depth[0] = ({"mutated": 0}, 1)
    assert lateral[0] == ({}, 1)
    assert depth[0] == ({"mutated": 0}, 1)
    # lists themselves must be distinct objects
    assert depth.named_axis is not lateral.named_axis
    assert depth.ndims is not lateral.ndims


def test_broadcast_empty_records():
    a = ak.to_layout(ak.Array([{}, {}]))
    b = ak.to_layout(ak.Array([{}, {}]))

    def action(inputs, **kwargs):
        # generic action: only act on leaves, recurse through records
        if all(x.is_numpy for x in inputs):
            return (inputs[0],)
        return None

    depth_context, lateral_context = NamedAxesWithDims.prepare_contexts([a, b])
    out = broadcast_and_apply(
        (a, b),
        action,
        depth_context=depth_context,
        lateral_context=lateral_context,
    )
    assert len(out) == 1
    assert out[0].to_list() == [{}, {}]


def test_prettify_non_identifier():
    # Names that are not valid identifiers must be JSON-quoted.
    assert _prettify_named_axes({"x y": 0}) == '"x y":0'
    assert _prettify_named_axes({"x": 0, "y": 1}) == "x:0, y:1"
    assert _prettify_named_axes({"$": 0}) == '"$":0'
