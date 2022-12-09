# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def num(array, axis=1, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an array of integers specifying the number of elements at a
    particular level.

    For instance, given the following doubly nested `array`,

        >>> array = ak.Array([[[1.1, 2.2, 3.3],
        ...                    [],
        ...                    [4.4, 5.5],
        ...                    [6.6]
        ...                   ],
        ...                   [],
        ...                   [[7.7],
        ...                    [8.8, 9.9]]
        ...                   ])

    The number of elements in `axis=1` is

        >>> ak.num(array, axis=1)
        <Array [4, 0, 2] type='3 * int64'>

    and the number of elements at the next level down, `axis=2`, is

        >>> ak.num(array, axis=2)
        <Array [[3, 0, 2, 1], [], [1, 2]] type='3 * var * int64'>

    The `axis=0` case is special: it returns a scalar, the length of the array.

        >>> ak.num(array, axis=0)
        3

    This function is useful for ensuring that slices do not raise errors. For
    instance, suppose that we want to select the first element from each
    of the outermost nested lists of `array`. One of these lists is empty, so
    selecting the first element (`0`) would raise an error. However, if our
    first selection is `ak.num(array) > 0`, we are left with only those lists
    that *do* have a first element:

        >>> array[ak.num(array) > 0, 0]
        <Array [[1.1, 2.2, 3.3], [7.7]] type='2 * var * float64'>

    To keep a placeholder (None) in each place we do not want to select,
    consider using #ak.mask instead of a #ak.Array.__getitem__.

        >>> array.mask[ak.num(array) > 0][:, 0]
        <Array [[1.1, 2.2, 3.3], None, [7.7]] type='3 * option[var * float64]'>
    """
    with ak._errors.OperationErrorContext(
        "ak.num",
        dict(array=array, axis=axis, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, axis, highlevel, behavior)


def _impl(array, axis, highlevel, behavior):
    layout = ak.operations.to_layout(array)
    behavior = ak._util.behavior_of(array, behavior=behavior)

    if not ak._util.is_integer(axis):
        raise ak._errors.wrap_error(
            TypeError(f"'axis' must be an integer, not {axis!r}")
        )

    if ak._util.maybe_posaxis(layout, axis, 1) == 0:
        if isinstance(layout, ak.record.Record):
            return 1
        else:
            return layout.length

    def action(layout, depth, **kwargs):
        posaxis = ak._util.maybe_posaxis(layout, axis, depth)

        if posaxis == depth and layout.is_list:
            return ak.contents.NumpyArray(layout.stops.data - layout.starts.data)

        elif layout.is_leaf:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )

    out = ak._do.recursively_apply(layout, action, behavior, numpy_to_regular=True)

    return ak._util.wrap(out, behavior, highlevel)
