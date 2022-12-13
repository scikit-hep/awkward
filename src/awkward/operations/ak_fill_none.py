# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numbers

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()
cpu = ak._backends.NumpyBackend.instance()


def fill_none(array, value, axis=-1, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        value: Data with which to replace None.
        axis (None or int): If None, replace all None values in the array
            with the given value; if an int, The dimension at which this
            operation is applied. The outermost dimension is `0`, followed
            by `1`, etc., and negative values count backward from the
            innermost: `-1` is the innermost  dimension, `-2` is the next
            level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Replaces missing values (None) with a given `value`.

    For example, in the following

        >>> array = ak.Array([[1.1, None, 2.2], [], [None, 3.3, 4.4]])

    The None values could be replaced with `0` by

        >>> ak.fill_none(array, 0)
        <Array [[1.1, 0, 2.2], [], [0, 3.3, 4.4]] type='3 * var * float64'>

    The replacement value doesn't strictly need the same type as the
    surrounding data. For example, the None values could also be replaced
    by a string.

        >>> ak.fill_none(array, "hi")
        <Array [[1.1, 'hi', 2.2], [], ['hi', ...]] type='3 * var * union[float64, s...'>

    The list content now has a union type:

        >>> ak.fill_none(array, "hi").type.show()
        3 * var * union[
            float64,
            string
        ]

    The values could be floating-point numbers or strings.
    """
    with ak._errors.OperationErrorContext(
        "ak.fill_none",
        dict(
            array=array, value=value, axis=axis, highlevel=highlevel, behavior=behavior
        ),
    ):
        return _impl(array, value, axis, highlevel, behavior)


def _impl(array, value, axis, highlevel, behavior):
    arraylayout = ak.operations.to_layout(array, allow_record=True, allow_other=False)
    behavior = ak._util.behavior_of(array, behavior=behavior)
    backend = ak._backends.backend_of(arraylayout, default=cpu)

    # Convert value type to appropriate layout
    if (
        isinstance(value, np.ndarray)
        and issubclass(value.dtype.type, (np.bool_, np.number))
        and len(value.shape) != 0
    ):
        valuelayout = ak.operations.to_layout(
            backend.nplike.asarray(value)[np.newaxis],
            allow_record=False,
            allow_other=False,
        )
    elif isinstance(value, (bool, numbers.Number, np.bool_, np.number)) or (
        isinstance(value, np.ndarray)
        and issubclass(value.dtype.type, (np.bool_, np.number))
    ):
        valuelayout = ak.operations.to_layout(
            backend.nplike.asarray(value), allow_record=False, allow_other=False
        )
    elif (
        ak._util.is_sized_iterable(value)
        and not (isinstance(value, (str, bytes)))
        or isinstance(value, (ak.highlevel.Record, ak.record.Record))
    ):
        valuelayout = ak.operations.to_layout(
            value, allow_record=True, allow_other=False
        )
        if isinstance(valuelayout, ak.record.Record):
            valuelayout = valuelayout.array[valuelayout.at : valuelayout.at + 1]
        elif len(valuelayout) == 0:
            offsets = ak.index.Index64(
                backend.index_nplike.array([0, 0], dtype=np.int64)
            )
            valuelayout = ak.contents.ListOffsetArray(offsets, valuelayout)
        else:
            valuelayout = ak.contents.RegularArray(valuelayout, len(valuelayout), 1)
    else:
        valuelayout = ak.operations.to_layout(
            [value], allow_record=False, allow_other=False
        )

    def maybe_fillna(layout):
        if layout.is_option:
            return ak._do.fill_none(layout, valuelayout)
        else:
            return layout

    if axis is None:

        def action(layout, continuation, **kwargs):
            return maybe_fillna(continuation())

    else:

        def action(layout, depth, **kwargs):
            posaxis = ak._util.maybe_posaxis(layout, axis, depth)
            if posaxis is not None and posaxis + 1 < depth:
                return layout
            elif posaxis is not None and posaxis + 1 == depth:
                return maybe_fillna(layout)
            elif layout.is_leaf:
                raise ak._errors.wrap_error(
                    np.AxisError(
                        f"axis={axis} exceeds the depth of this array ({depth})"
                    )
                )

    out = ak._do.recursively_apply(arraylayout, action, behavior)

    return ak._util.wrap(out, behavior, highlevel)
