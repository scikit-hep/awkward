# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._experimental import experimental
from awkward._layout import HighLevelContext, ensure_same_backend, maybe_posaxis
from awkward._namedaxis import NAMED_AXIS_KEY, NamedAxesWithDims
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._regularize import regularize_axis
from awkward.errors import AxisError

__all__ = ("broadcast_slice",)

np = NumpyMetadata.instance()


@high_level_function()
@experimental
def broadcast_slice(
    array, indices, axis=1, *, highlevel=True, behavior=None, attrs=None
):
    """Select from each collection using arbitrarily nested local integer indices.

    Args:
        array: Array-like data containing the collections to select from.
        indices: Nested integer indices, optionally containing missing values.
        axis (int): Axis of `array` to index. Dimensions before this axis align
            with `indices`; all remaining index dimensions describe selections
            from the same collection. Negative axes count from the innermost
            dimension of `array`.
        highlevel (bool): Return an #ak.Array if True, otherwise a low-level layout.
        behavior (None or dict): Custom behaviors for the output.
        attrs (None or dict): Custom attributes for the output.

    This explicitly broadcasts each collection over the extra nesting in the
    indices. It does not change ordinary #ak.Array.__getitem__ slicing rules.
    Negative indices count from the end of their own collection. An out-of-bounds
    index raises an error, even if another collection contains that index.
    CUDA kernel errors are reported through the usual deferred error checking.
    Boolean masks are not accepted; use integer indices instead.

    For example, each event has a collection of clusters, and each muon has a
    list of indices into the clusters in its own event:

        >>> clusters = ak.Array([[10, 20, 30], [40, 50]])
        >>> indices = ak.Array([[[2, 0], [1]], [[1], []]])
        >>> ak.broadcast_slice(clusters, indices).to_list()
        [[[30, 10], [20]], [[50], []]]

    The output retains the index nesting followed by the structure of the
    selected values (including records or lists). At axis=0, indices select from
    the entire array. Otherwise the outer lengths must match; deeper prefix
    dimensions follow Awkward's left-broadcasting rules. Missing values in either
    input propagate to the output. Named axis metadata is not propagated.
    """
    yield array, indices
    return _impl(array, indices, axis, highlevel, behavior, attrs)


def _impl(array, indices, axis, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout, index = ensure_same_backend(
            ctx.unwrap(array, allow_record=False, primitive_policy="error"),
            ctx.unwrap(indices, allow_record=False, primitive_policy="error"),
        )

    axis = regularize_axis(axis, none_allowed=False)
    axis = maybe_posaxis(layout, axis, 1)
    if axis is None:
        raise ValueError(
            "negative axis is ambiguous for an array with differing depths"
        )
    if axis < 0 or axis >= layout.minmax_depth[0]:
        raise AxisError(f"axis={axis} exceeds the depth of this array")
    if (
        axis > 0
        and layout.length is not unknown_length
        and index.length is not unknown_length
        and layout.length != index.length
    ):
        raise ValueError("array and indices must have the same outer length")

    def validate(node, **kwargs):
        if node.is_record or node.parameter("__array__") in ("string", "bytestring"):
            raise TypeError("indices must contain integers, not records or strings")
        if node.is_numpy and node.dtype.kind not in "iu":
            raise TypeError(
                "indices must contain integers, not booleans or other types"
            )

    ak._do.recursively_apply(index, validate, return_array=False)

    def normalize_unknown(node, **kwargs):
        if node.is_unknown:
            return node.to_NumpyArray(np.dtype(np.int64))
        return None

    index = ak._do.recursively_apply(index, normalize_unknown)

    def gather(inputs, **kwargs):
        collections, positions = inputs
        if collections.is_record and positions.is_regular:
            singleton = ak.contents.RegularArray(
                collections, 1, zeros_length=collections.length
            )
            return (broadcast([singleton, positions], gather),)
        if collections.is_record and positions.is_numpy and len(positions.shape) == 1:
            # Each integer is paired with its own collection. A jagged singleton
            # index selects one item per collection using ordinary slicing.
            singleton = ak.contents.RegularArray(
                positions, 1, zeros_length=positions.length
            )
            selected = collections.content(0)[singleton.to_ListOffsetArray64(True)]
            return (selected[:, 0],)
        return None

    def broadcast(inputs, action):
        depth_context, lateral_context = NamedAxesWithDims.prepare_contexts(inputs)
        return ak._broadcasting.broadcast_and_apply(
            inputs,
            action,
            depth_context=depth_context,
            lateral_context=lateral_context,
            right_broadcast=False,
            numpy_to_regular=True,
            function_name="ak.broadcast_slice",
        )[0]

    def select(collections, positions):
        if collections.is_regular:
            collections = collections.to_ListOffsetArray64(True)
        # Hide collection structure from broadcasting. Carrying this record only
        # repeats list references; its contents remain ordinary sliceable lists.
        wrapped = ak.contents.RecordArray([collections], ["collection"])
        return broadcast([wrapped, positions], gather)

    def action(inputs, depth, **kwargs):
        if depth == axis and inputs[0].is_list:
            return (select(*inputs),)
        return None

    if axis == 0:
        collection = ak.contents.RegularArray(layout, layout.length, zeros_length=1)
        out = select(collection, index)
    else:
        out = broadcast([layout, index], action)
    return ctx.without_attr(NAMED_AXIS_KEY).wrap(out, highlevel=highlevel)
