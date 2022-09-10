# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("where")
def where(condition, *args, **kwargs):
    """
    Args:
        condition (np.ndarray or rectilinear #ak.Array of booleans): In the
            three-argument form of this function (`condition`, `x`, `y`),
            True values in `condition` select values from `x` and False
            values in `condition` select values from `y`.
        x: Data with the same length as `condition`.
        y: Data with the same length as `condition`.
        mergebool (bool, default is True): If True, boolean and numeric data
            can be combined into the same buffer, losing information about
            False vs `0` and True vs `1`; otherwise, they are kept in separate
            buffers with distinct types (using an #ak.layout.UnionArray8_64).
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    This function has a one-argument form, `condition` without `x` or `y`, and
    a three-argument form, `condition`, `x`, and `y`. In the one-argument form,
    it is completely equivalent to NumPy's
    [nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html)
    function.

    In the three-argument form, it acts as a vectorized ternary operator:
    `condition`, `x`, and `y` must all have the same length and

        output[i] = x[i] if condition[i] else y[i]

    for all `i`. The structure of `x` and `y` do not need to be the same; if
    they are incompatible types, the output will have #ak.type.UnionType.
    """
    mergebool, highlevel, behavior = ak._v2._util.extra(
        (), kwargs, [("mergebool", True), ("highlevel", True), ("behavior", None)]
    )

    if len(args) == 0:
        with ak._v2._util.OperationErrorContext(
            "ak._v2.where",
            dict(condition=condition, mergebool=mergebool, highlevel=highlevel),
        ):
            return _impl1(condition, mergebool, highlevel, behavior)

    elif len(args) == 1:
        raise ak._v2._util.error(
            ValueError("either both or neither of x and y should be given")
        )

    elif len(args) == 2:
        x, y = args
        with ak._v2._util.OperationErrorContext(
            "ak._v2.where",
            dict(
                condition=condition, x=x, y=y, mergebool=mergebool, highlevel=highlevel
            ),
        ):
            return _impl3(condition, x, y, mergebool, highlevel, behavior)

    else:
        raise ak._v2._util.error(
            TypeError(
                "where() takes from 1 to 3 positional arguments but {} were "
                "given".format(len(args) + 1)
            )
        )


def _impl1(condition, mergebool, highlevel, behavior):
    akcondition = ak._v2.operations.to_layout(
        condition, allow_record=False, allow_other=False
    )
    nplike = ak.nplike.of(akcondition)

    akcondition = ak._v2.contents.NumpyArray(ak._v2.operations.to_numpy(akcondition))
    out = nplike.nonzero(ak._v2.operations.to_numpy(akcondition))
    if highlevel:
        return tuple(
            ak._v2._util.wrap(
                ak._v2.contents.NumpyArray(x),
                ak._v2._util.behavior_of(condition, behavior=behavior),
            )
            for x in out
        )
    else:
        return tuple(ak._v2.contents.NumpyArray(x) for x in out)


def _impl3(condition, x, y, mergebool, highlevel, behavior):
    akcondition = ak._v2.operations.to_layout(
        condition, allow_record=False, allow_other=False
    )

    left = ak._v2.operations.to_layout(x, allow_record=False, allow_other=True)
    right = ak._v2.operations.to_layout(y, allow_record=False, allow_other=True)

    good_arrays = [akcondition]
    if isinstance(left, ak._v2.contents.Content):
        good_arrays.append(left)
    if isinstance(right, ak._v2.contents.Content):
        good_arrays.append(right)
    nplike = ak.nplike.of(*good_arrays)

    def action(inputs, **kwargs):
        akcondition, left, right = inputs
        if isinstance(akcondition, ak._v2.contents.NumpyArray):
            npcondition = nplike.index_nplike.asarray(akcondition)
            tags = ak._v2.index.Index8((npcondition == 0).view(np.int8))
            index = ak._v2.index.Index64(
                nplike.index_nplike.arange(len(tags), dtype=np.int64), nplike=nplike
            )
            if not isinstance(left, ak._v2.contents.Content):
                left = ak._v2.contents.NumpyArray(nplike.repeat(left, len(tags)))
            if not isinstance(right, ak._v2.contents.Content):
                right = ak._v2.contents.NumpyArray(nplike.repeat(right, len(tags)))
            tmp = ak._v2.contents.UnionArray(tags, index, [left, right])
            return (tmp.simplify_uniontype(mergebool=mergebool),)
        else:
            return None

    behavior = ak._v2._util.behavior_of(condition, x, y, behavior=behavior)
    out = ak._v2._broadcasting.broadcast_and_apply(
        [akcondition, left, right],
        action,
        behavior,
        numpy_to_regular=True,
    )

    return ak._v2._util.wrap(out[0], behavior, highlevel)
